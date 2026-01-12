"""RAG Engine Helper Methods - query 메서드 분리"""
import logging
import re
import base64
import json
from google.genai import types
from prompts import PROMPT_TEMPLATES
from config import constants as C

logger = logging.getLogger(__name__)


class QueryHelperMixin:
    """query() 메서드에서 사용하는 헬퍼 메서드 모음"""

    # ========================================
    # 특정 질문 타입별 처리 메서드
    # ========================================

    def _handle_general_conversation(self, question: str) -> dict:
        """일반 대화 처리 (검색 없이 LLM 직접 응답)"""
        logger.info("💬 일반 대화 감지 (검색 생략)")
        prompt = self._build_general_conversation_prompt(question)
        response = self.genai_client.models.generate_content(
            model=self.model_name_pro,
            contents=prompt
        )
        return {
            "answer": response.text,
            "sources": [],
            "question": question
        }

    def _handle_conversation_history_query(self, question: str, session_id: str) -> dict:
        """대화 이력 조회 처리"""
        logger.info("💬 대화 이력 조회 질문 감지")
        if not session_id:
            return {
                "answer": "세션 ID가 없습니다. 다시 접속 후 질문해주세요.",
                "sources": [],
                "question": question
            }

        history_summary = self.get_conversation_history_summary(session_id=session_id)

        if history_summary.get('total_conversations', 0) == 0:
            return {
                "answer": "이 세션에서 저장된 대화 이력이 없습니다.",
                "sources": [],
                "question": question
            }

        # 응답 포맷팅
        answer_lines = [
            f"**세션 대화 이력 ({session_id})**",
            f"- 대화 수: {history_summary['total_conversations']}개"
        ]

        if history_summary['sessions']:
            session_info = history_summary['sessions'][0]
            answer_lines.append(f"- 기간: {session_info['first_timestamp'][:19]} ~ {session_info['last_timestamp'][:19]}")
            answer_lines.append("")
            answer_lines.append("**질문 목록:**")
            for j, conv in enumerate(session_info['conversations'], 1):
                answer_lines.append(f"  {j}. {conv['question']}")
            answer_lines.append("")

        return {
            "answer": "\n".join(answer_lines),
            "sources": [],
            "question": question
        }

    def _handle_crf_metadata_query(self, question: str, hospital_code: str = None, chat_history: list = None) -> dict:
        """CRF 메타데이터 질문 처리"""
        logger.info("🗂️ CRF 메타데이터 질문 감지 → 전체 현황 요약")

        where_filter = {"hospital": hospital_code} if hospital_code else None
        data = self.collection.get(where=where_filter, include=["metadatas"])
        metadatas = data.get("metadatas") or []

        if not metadatas:
            return {
                "answer": "CRF 메타데이터를 찾을 수 없습니다.",
                "sources": [],
                "question": question
            }

        dataset_meta = self.get_dataset_metadata(metadatas)
        formatted_meta = self.format_metadata_for_llm(dataset_meta)

        return {
            "answer": formatted_meta,
            "sources": [],
            "question": question,
            "document_count": dataset_meta.get("total_records", 0)
        }

    def _handle_crf_statistics_query(self, question: str, hospital_code: str = None) -> dict:
        """CRF 통계 질문 처리 (차트 생성 포함)"""
        logger.info("📊 통계 질문 감지 → Python 직접 계산 + 차트 생성")

        where_filter = {"hospital": hospital_code} if hospital_code else None
        data = self.collection.get(where=where_filter, include=["metadatas", "documents"])
        logger.info(f"  📦 데이터 로드: {len(data['documents'])}개")

        # 통계 계산
        stats = self.calculate_crf_statistics(
            data['documents'],
            data['metadatas'],
            hospital_code
        )
        stats_text = self.format_statistics_for_llm(stats)
        hospital_name = stats['hospital_name']

        # 핵심 필드만 추출 (Rate Limit 회피 + LLM 정확도 향상)
        essential_fields = [
            # 바이오마커 (가장 중요)
            'Ki-67 LI (%)', 'ER_IHC', 'PR_IHC', 'HER2_IHC',
            'ER (-/+)', 'PR (-/+)', 'HER2 (-/+)',
            # 환자 정보
            '나이 (진단시)', '병원명',
            # 종양 정보
            '암 size (mm)_장경', 'T category', 'N category', 'M category',
            'NG (1/2/3)', 'HG (1/2/3/4)',
            '진단명 (histologic type',  # 조직학적 타입
            # 치료 및 예후
            '수술명 (partial/total)', '림프절 전이여부_수술당시',
            '폐경 여부', 'Stage', '재발 여부'
        ]

        raw_metadata_list = []

        # documents에서 파싱 (현재 데이터는 documents에 텍스트로 저장됨)
        for i, doc in enumerate(data['documents']):
            record = {'병원': data['metadatas'][i].get('hospital', '')}

            # "필드명: 값" 형식 파싱
            for line in doc.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()

                    # 핵심 필드만 추출
                    if key in essential_fields:
                        record[key] = value

            raw_metadata_list.append(record)

        # JSON 직렬화 (ensure_ascii=False로 한글 유지)
        raw_metadata_json = json.dumps(raw_metadata_list, ensure_ascii=False, indent=2)
        logger.info(f"  📋 원본 메타데이터 크기: {len(raw_metadata_json):,} 문자 ({len(raw_metadata_list):,}개 레코드)")
        logger.info(f"  🔑 추출된 필드 수: {len(essential_fields)}개 (전체 136개 중)")

        # 통계/메타데이터가 토큰 한도를 넘지 않도록 청크 단위로 Code Execution 호출
        # 요약 통계만 전달 (raw_metadata는 LLM에 보내지 않음)
        stat_chunks = self._chunk_statistics_text(stats_text, max_chars=80000)
        total_parts = len(stat_chunks)
        chunked_prompts = []
        for idx, stat_chunk in enumerate(stat_chunks, start=1):
            chunked_prompts.append({
                "statistics": f"[통계 파트 {idx}/{total_parts}]\n{stat_chunk}",
                "raw_metadata": ""  # 메타데이터 미전송
            })

        chart_images = []
        text_responses = []

        # 차트 실행은 너무 많아지지 않도록 앞부분 일부 청크만 사용
        chart_chunk_limit = len(chunked_prompts)

        for idx, chunk_info in enumerate(chunked_prompts[:chart_chunk_limit], start=1):
            if C.DEBUG_CONFIG.get("crf_chunk_logging"):
                logger.info(
                    f"📊 청크 호출 {idx}/{chart_chunk_limit} | "
                    f"통계 길이: {len(chunk_info['statistics'])} / 메타데이터 길이: {len(chunk_info['raw_metadata'])}"
                )
            logger.info(f"📊 차트 생성 중 (Code Execution) 파트 {idx}/{chart_chunk_limit}...")
            prompt = PROMPT_TEMPLATES["crf_statistics"].format(
                statistics=chunk_info["statistics"],
                raw_metadata=chunk_info["raw_metadata"],
                question=question,
                hospital_name=hospital_name
            )

            code_execution_tool = types.Tool(code_execution={})
            config = types.GenerateContentConfig(tools=[code_execution_tool])
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_responses.append(part.text)
                            elif hasattr(part, 'inline_data') and part.inline_data:
                                image_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                                chart_images.append({
                                    'mime_type': part.inline_data.mime_type,
                                    'data': image_data
                                })
                                logger.info(f"  📈 차트 이미지 생성됨: {part.inline_data.mime_type}")
                            elif hasattr(part, 'executable_code') and part.executable_code:
                                logger.info(f"  🐍 LLM 실행 코드:\n{part.executable_code.code[:500]}...")
                            elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                                logger.info(f"  ✅ 코드 실행 결과: {part.code_execution_result.outcome}")
            elif hasattr(response, 'parts'):
                for part in response.parts:
                    if hasattr(part, 'text') and part.text:
                        text_responses.append(part.text)
                    elif hasattr(part, 'inline_data') and part.inline_data:
                        image_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                        chart_images.append({
                            'mime_type': part.inline_data.mime_type,
                            'data': image_data
                        })
                        logger.info(f"  📈 차트 이미지 생성됨: {part.inline_data.mime_type}")

        return {
            "answer": "\n\n".join(text_responses) if text_responses else "차트가 생성되었습니다.",
            "sources": [],
            "question": question,
            "document_count": len(data['documents']) if data else 0,
            "charts": chart_images
        }

    # ========================================
    # 직접 조회 메서드
    # ========================================

    def _perform_direct_lookup(self, question: str):
        """이슈 번호 또는 CRF record_id 직접 조회"""
        direct_results = None

        if self.use_case == "redmine":
            issue_ids = self._extract_issue_ids(question)
            if issue_ids:
                logger.info(f"🔎 이슈 번호 감지 → {issue_ids}")
                issue_ids_str = [str(i) for i in issue_ids]
                issue_ids_int = [int(i) for i in issue_ids_str if i.isdigit()]

                # 문자열 매칭
                direct_results = self.collection.get(
                    where={"issue_id": {"$in": issue_ids_str}},
                    include=["metadatas", "documents", "embeddings"]
                )

                # 없으면 정수 매칭
                if not direct_results.get("documents") and issue_ids_int:
                    logger.info("  ➡️ 문자열 매칭 실패 → 정수형 매칭 재시도")
                    direct_results = self.collection.get(
                        where={"issue_id": {"$in": issue_ids_int}},
                        include=["metadatas", "documents", "embeddings"]
                    )

                found = len(direct_results.get("documents", []))
                if found:
                    logger.info(f"  ✅ 이슈 번호 매칭 성공: {found}건")
                else:
                    logger.info("  ⚠️ 해당 이슈를 찾지 못했습니다. 일반 검색으로 폴백합니다.")

        elif self.use_case == "crf":
            record_ids = self._extract_crf_record_ids(question)
            if record_ids:
                logger.info(f"🔎 CRF record_id 감지 → {record_ids}")
                direct_results = self.collection.get(
                    where={"record_id": {"$in": record_ids}},
                    include=["metadatas", "documents", "embeddings"]
                )

                found = len(direct_results.get("documents", []))
                if found:
                    logger.info(f"  ✅ CRF record_id 매칭 성공: {found}건")
                else:
                    logger.info("  ⚠️ 해당 record_id를 찾지 못했습니다. 일반 검색으로 폴백합니다.")

        return direct_results

    # ========================================
    # Top-K 결정 메서드
    # ========================================

    def _determine_top_k(self, question: str, top_k: int, recent_query: bool) -> int:
        """적응형 top_k 결정"""
        if top_k is not None:
            if recent_query and top_k < C.DEFAULT_TOP_K['recent']:
                top_k = C.DEFAULT_TOP_K['recent']
                logger.info("📅 최신 질문 감지 → top_k=100")
            return top_k

        # 자동 설정
        if self.use_case == "crf":
            top_k = C.DEFAULT_TOP_K['crf']
            logger.info(f"🧬 CRF 질문 → top_k={top_k}")
        elif self._is_version_or_comparison_query(question):
            top_k = C.DEFAULT_TOP_K['version']
            logger.info(f"📊 버전/비교 질문 감지 → top_k={top_k}")
        elif self._is_specific_technical_query(question):
            top_k = C.DEFAULT_TOP_K['technical']
            logger.info(f"🔧 기술 검색 질문 감지 → top_k={top_k}")
        else:
            top_k = C.DEFAULT_TOP_K['general']
            logger.info(f"📝 일반 질문 → top_k={top_k}")

        if recent_query and top_k < C.DEFAULT_TOP_K['recent']:
            top_k = C.DEFAULT_TOP_K['recent']
            logger.info("📅 최신 질문 감지 → top_k=100")

        return top_k

    # ========================================
    # 출처 생성 메서드
    # ========================================

    def _generate_sources(self, documents: list, metadatas: list, distances: list, answer: str) -> list:
        """use_case에 맞는 출처 정보 생성"""
        if self.use_case == "redmine":
            return self._generate_redmine_sources(documents, metadatas, distances, answer)
        elif self.use_case == "crf":
            return self._generate_crf_sources(documents, metadatas, distances)
        else:
            return self._generate_document_sources(documents, metadatas, distances)

    def _generate_redmine_sources(self, documents: list, metadatas: list, distances: list, answer: str) -> list:
        """Redmine 출처 생성"""
        # 답변에서 언급된 이슈 번호 추출
        mentioned_issues = set()
        for match in re.finditer(r'#(\d+)', answer):
            issue_num = int(match.group(1))
            mentioned_issues.add(issue_num)

        # 모든 검색된 이슈를 sources로 생성
        all_sources = [
            {
                "issue_id": meta.get("issue_id", "N/A"),
                "subject": meta.get("subject", "N/A"),
                "distance": float(dist),
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "url": f"{self.redmine_url}/issues/{meta.get('issue_id', '')}" if meta.get("issue_id") else None
            }
            for meta, dist, doc in zip(metadatas, distances, documents)
        ]

        # 답변에 언급된 이슈가 있으면 그것만, 없으면 상위 N개
        if mentioned_issues:
            filtered_sources = [
                src for src in all_sources
                if src["issue_id"] != "N/A" and int(src["issue_id"]) in mentioned_issues
            ]
            filtered_sources.sort(key=lambda x: int(x["issue_id"]))
            logger.info(f"  📌 답변에 언급된 이슈: {len(filtered_sources)}개 (전체 검색: {len(documents)}개)")
        else:
            top_n = min(5, len(all_sources))
            filtered_sources = [src for src in all_sources[:top_n] if src["issue_id"] != "N/A"]
            logger.info(f"  📌 참조 이슈 (언급 없음): {len(filtered_sources)}개 (전체 검색: {len(documents)}개)")

        return filtered_sources

    def _generate_crf_sources(self, documents: list, metadatas: list, distances: list) -> list:
        """CRF 출처 생성"""
        top_n = min(5, len(documents))
        filtered_sources = []

        for meta, dist, doc in zip(metadatas[:top_n], distances[:top_n], documents[:top_n]):
            # 문서에서 병리번호 추출
            path_no_match = re.search(r'병리번호:\s*([^\n]+)', doc)
            path_no = path_no_match.group(1).strip() if path_no_match else "N/A"

            filtered_sources.append({
                "record_id": meta.get("record_id", "N/A"),
                "hospital": meta.get("hospital", "N/A"),
                "path_no": path_no,
                "sheet": meta.get("sheet", "N/A"),
                "row_index": meta.get("row_index", 0),
                "distance": float(dist),
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc
            })

        logger.info(f"  📌 참조 CRF 데이터: {len(filtered_sources)}개 (전체 검색: {len(documents)}개)")
        return filtered_sources

    def _generate_document_sources(self, documents: list, metadatas: list, distances: list) -> list:
        """일반 문서 출처 생성"""
        top_n = min(5, len(documents))
        filtered_sources = [
            {
                "filename": meta.get("filename", "Unknown"),
                "file_type": meta.get("file_type", "N/A"),
                "doc_category": meta.get("doc_category", "N/A"),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 1),
                "distance": float(dist),
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc
            }
            for meta, dist, doc in zip(metadatas[:top_n], distances[:top_n], documents[:top_n])
        ]

        logger.info(f"  📌 참조 문서: {len(filtered_sources)}개 (전체 검색: {len(documents)}개)")
        return filtered_sources

    # ========================================
    # 대화 히스토리 포맷팅
    # ========================================

    def _format_history_text(self, chat_history: list, relevant_history: list) -> str:
        """대화 히스토리 포맷팅"""
        if not chat_history and not relevant_history:
            return ""

        history_text = "\n<대화_히스토리>\n"

        # 최근 대화
        if chat_history:
            history_text += "[최근 대화]\n"
            for i, turn in enumerate(chat_history[-C.CHAT_HISTORY_CONFIG['max_recent_turns']:], 1):
                history_text += f"- 사용자: {turn['question']}\n"
                history_text += f"  어시스턴트: {turn['answer']}\n"

        # 관련 과거 대화
        if relevant_history:
            history_text += "\n[관련 과거 대화]\n"
            for hist in relevant_history[:C.CHAT_HISTORY_CONFIG['max_relevant_history']]:
                history_text += f"- 사용자: {hist['question']}\n"
                history_text += f"  어시스턴트: {hist['answer']}\n"

        history_text += "</대화_히스토리>\n"
        return history_text

    # ========================================
    # 문서 검색 및 후처리 메서드
    # ========================================

    def _search_documents(self, question: str, chat_history: list, direct_results, top_k: int):
        """문서 검색 (직접 조회 또는 벡터 검색)"""
        documents, metadatas, distances = [], [], []

        if direct_results and direct_results.get("documents"):
            # 직접 조회 결과 사용
            documents = direct_results.get("documents", [])
            metadatas = direct_results.get("metadatas", [])
            distances = [0.0 for _ in documents]
            logger.info(f"  ✅ 직접 조회 결과: {len(documents)}건")
        else:
            # 벡터 검색
            search_query = self._build_search_query(question, chat_history)
            logger.info(f"🔍 검색 중... (top_k={top_k})")
            query_embedding = self._embed(search_query, "RETRIEVAL_QUERY")

            # CRF: 병원 필터링
            hospital_code = None
            if self.use_case == "crf":
                hospital_code = self._extract_hospital_code_from_question(question)

            # Vector DB 검색
            where_filter = {"hospital": hospital_code} if hospital_code else None
            results = self.collection.query(
                query_embeddings=[query_embedding],
                where=where_filter,
                n_results=top_k
            )

            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]

        logger.info(f"  ✅ 검색된 문서: {len(documents)}개")
        return documents, metadatas, distances

    def _post_process_documents(self, documents: list, metadatas: list, distances: list,
                                question: str, recent_query: bool):
        """문서 후처리 (재정렬, 필터링 등)"""
        # 1. 키워드 보강 검색 (Redmine 전용)
        if self.use_case == "redmine":
            keywords = self._extract_model_keywords(question)
            if keywords:
                documents, metadatas, distances = self._augment_with_keyword_matches(
                    documents, metadatas, distances, keywords
                )
                documents, metadatas, distances = self._filter_by_keywords(
                    documents, metadatas, distances, keywords
                )

        # 2. 버전 토큰 재정렬
        version_tokens = self._extract_version_tokens(question)
        if version_tokens and distances:
            logger.info(f"  🎯 버전 토큰 감지 → 재정렬: {version_tokens}")
            scored = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                base_score = 1 - float(dist)
                boost = 0
                subject = str(meta.get("subject", "")).lower()
                version_meta = str(meta.get("version", "")).lower()
                doc_lower = str(doc).lower()
                for token in version_tokens:
                    if token in subject:
                        boost += 1
                    if token in version_meta:
                        boost += 1
                    if token in doc_lower:
                        boost += 1
                scored.append((base_score + 0.3 * boost, doc, meta, dist))

            scored.sort(key=lambda x: x[0], reverse=True)
            documents = [s[1] for s in scored]
            metadatas = [s[2] for s in scored]
            distances = [s[3] for s in scored]

        # 3. 최신순 재정렬
        if recent_query:
            logger.info("  📅 최신 데이터 요청 감지 → 날짜순 재정렬")
            documents, metadatas, distances = self._sort_by_recency(documents, metadatas, distances)

        return documents, metadatas, distances

    def _generate_answer(self, question: str, documents: list, metadatas: list, distances: list,
                        chat_history: list, relevant_history: list) -> dict:
        """컨텍스트 구성 및 LLM 답변 생성"""
        # 1. 컨텍스트 제한 결정
        if self.use_case == "crf":
            context_limit = len(documents)  # CRF는 전체 사용
            logger.info(f"  🧬 CRF → 검색된 문서 전체 사용: {context_limit}개")
        else:
            context_limit = min(C.CONTEXT_LIMITS.get(self.use_case, 15), len(documents))

        # 2. 컨텍스트 포맷팅
        context = self._format_context(documents, metadatas, context_limit)
        logger.info(f"  📄 컨텍스트에 사용된 문서: {context_limit}개")

        # 3. 대화 히스토리 포맷팅
        history_text = self._format_history_text(chat_history, relevant_history)

        # 4. 프롬프트 생성
        prompt = self._build_prompt(context, history_text, question)

        # 5. LLM 답변 생성
        logger.info("💬 답변 생성 중...")
        response = self.genai_client.models.generate_content(
            model=self.model_name_pro,
            contents=prompt
        )
        answer = response.text

        # 6. 출처 생성
        sources = self._generate_sources(documents, metadatas, distances, answer)

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "document_count": len(documents)
        }
