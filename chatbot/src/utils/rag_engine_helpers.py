"""
RAG Engine Helper Methods - query 메서드 분리

[역할]
이 파일은 RAG 엔진의 query() 메서드에서 사용하는 **고수준 비즈니스 로직**을 담당합니다.
질문 타입별 처리 워크플로우, 검색 실행, 답변 생성 등 query() 메서드의 핵심 실행 흐름을 구현합니다.

[주요 기능]
1. 특정 질문 타입별 처리 (_handle_*)
   - 일반 대화, 대화 이력 조회, CRF 메타데이터, CRF 통계(차트) 처리
2. 직접 조회 (_perform_direct_lookup)
   - 이슈 번호/CRF record_id로 직접 DB 조회
3. Top-K 결정 (_determine_top_k)
   - 질문 타입에 따라 적응형 top_k 자동 설정
4. 출처 생성 (_generate_sources)
   - Redmine/CRF/일반 문서별 출처 포맷팅
5. 문서 검색 및 후처리 (_search_documents, _post_process_documents)
   - 벡터 검색 실행, 키워드 보강, 버전 토큰 재정렬, 최신순 정렬
6. 답변 생성 (_generate_answer)
   - 컨텍스트 구성 및 LLM 답변 생성

[vs rag_utils.py]
- rag_utils.py는 범용 유틸리티 함수들(추출, 판별, 변환 등)을 제공
- 이 파일은 rag_utils.py의 함수들을 조합하여 query() 메서드의 실행 흐름을 구현
"""
import logging
import re
import base64
import json as _json
from google.genai import types
from prompts import PROMPT_TEMPLATES
from config import constants as C

logger = logging.getLogger(__name__)


class QueryHelperMixin:
    """
    query() 메서드에서 사용하는 헬퍼 메서드 모음

    이 클래스는 RAG 엔진의 핵심 비즈니스 로직을 담당합니다.
    질문 타입별 처리, 검색 실행, 답변 생성 등 query() 메서드의 실행 워크플로우를 구현합니다.
    """

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

        # 통계 텍스트를 청크 단위로 Code Execution 호출
        stat_chunks = self._chunk_statistics_text(stats_text, max_chars=C.CRF_STATISTICS_CONFIG["max_chunk_chars"])
        total_parts = len(stat_chunks)
        chunked_prompts = []
        for idx, stat_chunk in enumerate(stat_chunks, start=1):
            chunked_prompts.append({
                "statistics": f"[통계 파트 {idx}/{total_parts}]\n{stat_chunk}",
                "raw_metadata": ""
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

            texts, images = self._extract_response_parts(response)
            text_responses.extend(texts)
            chart_images.extend(images)

        return {
            "answer": "\n\n".join(text_responses) if text_responses else "차트가 생성되었습니다.",
            "sources": [],
            "question": question,
            "document_count": len(data['documents']) if data else 0,
            "charts": chart_images
        }

    def _extract_response_parts(self, response) -> tuple[list, list]:
        """genai response에서 텍스트와 이미지(base64) 파트를 추출한다.

        Returns:
            (text_parts, image_parts) 튜플
        """
        texts = []
        images = []

        if hasattr(response, 'candidates') and response.candidates:
            parts = [
                part
                for candidate in response.candidates
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts')
                for part in candidate.content.parts
            ]
        elif hasattr(response, 'parts'):
            parts = response.parts
        else:
            return texts, images

        for part in parts:
            if hasattr(part, 'text') and part.text:
                texts.append(part.text)
            elif hasattr(part, 'inline_data') and part.inline_data:
                image_data = base64.b64encode(part.inline_data.data).decode('utf-8')
                images.append({'mime_type': part.inline_data.mime_type, 'data': image_data})
                logger.info(f"  📈 차트 이미지 생성됨: {part.inline_data.mime_type}")
            elif hasattr(part, 'executable_code') and part.executable_code:
                logger.info(f"  🐍 LLM 실행 코드:\n{part.executable_code.code[:500]}...")
            elif hasattr(part, 'code_execution_result') and part.code_execution_result:
                logger.info(f"  ✅ 코드 실행 결과: {part.code_execution_result.outcome}")

        return texts, images

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
    # Candidate-K 결정 및 유사도 컷오프 메서드
    # ========================================

    def _determine_candidate_k(self, question: str) -> int:
        """
        검색할 후보 문서 개수 결정 (candidate_k)

        비교/버전 질문이면 더 많이 검색, 그 외는 기본값
        실제 사용할 문서는 유사도 컷오프로 결정
        """
        if self._is_version_or_comparison_query(question):
            candidate_k = C.CANDIDATE_K['comparison']
            logger.info(f"📊 비교/버전 질문 → candidate_k={candidate_k}")
        elif self._is_recent_query(question) or self._is_group_query(question):
            candidate_k = C.CANDIDATE_K['recent']
            logger.info(f"📅 최신/단체 쿼리 → candidate_k={candidate_k}")
        else:
            candidate_k = C.CANDIDATE_K['default']
            logger.info(f"📝 일반 질문 → candidate_k={candidate_k}")

        return candidate_k

    def _is_group_query(self, question: str) -> bool:
        """단체(전체 사원) 쿼리 여부 판단"""
        return any(kw in question for kw in C.GROUP_QUERY_KEYWORDS)

    def _apply_similarity_cutoff(self, documents: list, metadatas: list,
                                  distances: list, distance_threshold: float = 0.65,
                                  min_docs: int = 3, max_docs: int = 50) -> tuple:
        """
        유사도 기반 컷오프 적용 (개별 distance 필터 방식)

        RRF 하이브리드 병합 후에는 문서 순서가 RRF score 순이므로
        distance가 단조증가하지 않을 수 있음.
        따라서 순서 기반이 아닌, 개별 distance 필터링으로 변경.

        Args:
            documents: 검색된 문서 리스트
            metadatas: 메타데이터 리스트
            distances: 거리(유사도) 리스트 (낮을수록 유사)
            distance_threshold: 절대 거리 임계값 (기본 0.6, Gemini embedding 기준)
            min_docs: 최소 반환 문서 수 (기본 3, 유사도 낮아도 최소 3개 반환)
            max_docs: 최대 반환 문서 수 (기본 50)

        Returns:
            (documents, metadatas, distances) 컷오프 적용된 튜플
        """
        if not documents or not distances:
            return documents, metadatas, distances

        # 개별 distance 필터: threshold 이하인 문서만 통과
        filtered = [
            (doc, meta, dist) for doc, meta, dist in zip(documents, metadatas, distances)
            if dist <= distance_threshold
        ]

        passed_count = len(filtered)

        # min_docs 보장: 필터 통과 문서가 부족하면 원본 순서대로 채움
        if passed_count < min_docs:
            filtered = list(zip(documents, metadatas, distances))[:min_docs]
            logger.info(f"  ✂️ 임계값 통과 {passed_count}개 < min_docs={min_docs} → {len(filtered)}개로 보장")
        else:
            logger.info(f"  ✂️ 임계값({distance_threshold}) 통과: {passed_count}개")

        # max_docs 제한
        filtered = filtered[:max_docs]

        logger.info(f"  📄 유사도 컷오프: {len(documents)}개 → {len(filtered)}개")

        if not filtered:
            return documents[:min_docs], metadatas[:min_docs], distances[:min_docs]

        result_docs, result_metas, result_dists = zip(*filtered)
        return list(result_docs), list(result_metas), list(result_dists)

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
        all_sources = []
        for meta, dist, doc in zip(metadatas, distances, documents):
            try:
                att_ids = _json.loads(meta.get("attachment_ids", "[]"))
            except Exception:
                att_ids = []
            try:
                att_fnames = _json.loads(meta.get("attachment_filenames", "{}"))
            except Exception:
                att_fnames = {}
            all_sources.append({
                "issue_id": meta.get("issue_id", "N/A"),
                "subject": meta.get("subject", "N/A"),
                "distance": float(dist),
                "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                "url": f"{self.redmine_url}/issues/{meta.get('issue_id', '')}" if meta.get("issue_id") else None,
                "attachment_ids": att_ids,
                "attachment_filenames": att_fnames,
            })

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
        """문서 검색 (직접 조회 또는 벡터 검색 + BM25 하이브리드)"""
        documents, metadatas, distances = [], [], []

        lf = getattr(self, 'langfuse', None)
        retrieval_span = None
        if lf and getattr(self, '_current_trace', None):
            try:
                retrieval_span = lf.start_span(
                    name="retrieval",
                    input={"question": question[:200]}
                )
            except Exception:
                retrieval_span = None

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

            # Vector DB 검색 (Dense)
            where_filter = {"hospital": hospital_code} if hospital_code else None
            results = self.collection.query(
                query_embeddings=[query_embedding],
                where=where_filter,
                n_results=top_k
            )

            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            dense_ids = results.get('ids', [[]])[0]

            # BM25 하이브리드 검색 (Redmine 전용)
            if (self.use_case == "redmine" and
                getattr(self, 'bm25_index', None) is not None and
                C.BM25_CONFIG.get("enabled", False)):
                documents, metadatas, distances = self._merge_with_bm25(
                    question, documents, metadatas, distances, dense_ids
                )

        logger.info(f"  ✅ 검색된 문서: {len(documents)}개")
        if retrieval_span:
            try:
                retrieval_span.update(output={"doc_count": len(documents)})
                retrieval_span.end()
            except Exception:
                pass
        return documents, metadatas, distances

    def _merge_with_bm25(self, question: str, dense_docs: list, dense_metas: list,
                          dense_dists: list, dense_ids: list) -> tuple:
        """BM25 결과와 Dense 결과를 RRF(Reciprocal Rank Fusion)로 병합"""
        import numpy as np

        rrf_k = C.BM25_CONFIG.get("rrf_k", 60)
        dense_weight = C.BM25_CONFIG.get("dense_weight", 1.0)
        bm25_weight = C.BM25_CONFIG.get("bm25_weight", 1.0)
        bm25_candidate_k = C.BM25_CONFIG.get("bm25_candidate_k", 80)

        # BM25 검색
        query_tokens = self._tokenize_for_bm25(question)
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # BM25 상위 K개 인덱스
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_candidate_k]

        logger.info(f"  🔤 BM25 검색: query_tokens={query_tokens[:10]}...")
        if len(top_bm25_indices) > 0:
            logger.info(f"  🔤 BM25 top1 score={bm25_scores[top_bm25_indices[0]]:.3f}")

        # doc_id → 정보 매핑 (통합용)
        doc_info = {}

        # Dense 결과 등록
        for rank, (doc_id, doc, meta, dist) in enumerate(zip(dense_ids, dense_docs, dense_metas, dense_dists)):
            doc_info[doc_id] = {"doc": doc, "meta": meta, "dist": dist}
            doc_info[doc_id]["dense_rank"] = rank
            doc_info[doc_id]["rrf_score"] = dense_weight / (rrf_k + rank + 1)

        # BM25 결과 등록 및 RRF 스코어 합산
        bm25_added = 0
        for rank, idx in enumerate(top_bm25_indices):
            if bm25_scores[idx] <= 0:
                continue

            doc_id = self.bm25_doc_ids[idx]
            bm25_rrf = bm25_weight / (rrf_k + rank + 1)

            if doc_id in doc_info:
                # 이미 Dense에 있으면 RRF 스코어 합산
                doc_info[doc_id]["rrf_score"] += bm25_rrf
            else:
                # BM25에만 있는 새 문서 추가
                # dist=0.55: 컷오프(0.6) 아래로 설정하여 similarity cutoff에서 잘리지 않도록 함
                doc_info[doc_id] = {
                    "doc": self.bm25_corpus[idx],
                    "meta": self.bm25_metadatas[idx],
                    "dist": 0.55,
                    "dense_rank": None,
                    "rrf_score": bm25_rrf,
                }
                bm25_added += 1

        if bm25_added > 0:
            logger.info(f"  🔤 BM25 추가 문서: {bm25_added}개 (Dense에 없던 문서)")

        # RRF 스코어 기준 정렬
        sorted_items = sorted(doc_info.values(), key=lambda x: x["rrf_score"], reverse=True)

        # 상위 문서 반환 (candidate_k 개수만큼)
        max_results = max(len(dense_docs), bm25_candidate_k)
        sorted_items = sorted_items[:max_results]

        merged_docs = [item["doc"] for item in sorted_items]
        merged_metas = [item["meta"] for item in sorted_items]
        merged_dists = [item["dist"] for item in sorted_items]

        logger.info(f"  🔀 RRF 병합 완료: Dense={len(dense_docs)}개 + BM25 추가={bm25_added}개 → 총 {len(merged_docs)}개")

        return merged_docs, merged_metas, merged_dists

    def _post_process_documents(self, documents: list, metadatas: list, distances: list,
                                question: str, recent_query: bool):
        """문서 후처리 (재정렬, 필터링 등)"""
        # 1. 키워드 보강 검색 (Redmine 전용)
        #    BM25 활성화 시 _augment는 BM25가 대체하므로 건너뜀, _filter는 유지
        if self.use_case == "redmine":
            keywords = self._extract_model_keywords(question)
            if keywords:
                bm25_active = (getattr(self, 'bm25_index', None) is not None and
                               C.BM25_CONFIG.get("enabled", False))
                if not bm25_active:
                    documents, metadatas, distances = self._augment_with_keyword_matches(
                        documents, metadatas, distances, keywords
                    )
                documents, metadatas, distances = self._filter_by_keywords(
                    documents, metadatas, distances, keywords
                )

        # 3. 최신순 재정렬
        if recent_query:
            logger.info("  📅 최신 데이터 요청 감지 → 날짜순 재정렬")
            documents, metadatas, distances = self._sort_by_recency(documents, metadatas, distances)
            # 작성자별 가장 최신 이슈를 앞으로 올리기
            documents, metadatas, distances = self._promote_latest_per_author(documents, metadatas, distances)

        # 4. 단체 쿼리: 사원별 최신 이슈 보장
        if self._is_group_query(question):
            logger.info("  👥 단체 쿼리 감지 → 사원별 최신 이슈 보장")
            documents, metadatas, distances = self._ensure_staff_latest_issues(
                documents, metadatas, distances, question
            )

        return documents, metadatas, distances

    def _generate_answer(self, question: str, documents: list, metadatas: list, distances: list,
                        chat_history: list, relevant_history: list) -> dict:
        """컨텍스트 구성 및 LLM 답변 생성"""
        # 유사도 컷오프에서 이미 관련 문서만 남겼으므로 추가 제한 없이 전체 사용
        context_limit = len(documents)

        # 컨텍스트 포맷팅
        context = self._format_context(documents, metadatas, context_limit)
        logger.info(f"  📄 컨텍스트에 사용된 문서: {context_limit}개")

        # 3. 대화 히스토리 포맷팅
        history_text = self._format_history_text(chat_history, relevant_history)

        # 4. 프롬프트 생성
        prompt = self._build_prompt(context, history_text, question)

        # 5. LLM 답변 생성
        logger.info("💬 답변 생성 중...")
        lf = getattr(self, 'langfuse', None)
        gen_span = None
        if lf and getattr(self, '_current_trace', None):
            try:
                gen_span = lf.start_span(
                    name="llm-generation",
                    input=prompt
                )
            except Exception:
                gen_span = None

        response = self.genai_client.models.generate_content(
            model=self.model_name_pro,
            contents=prompt
        )
        answer = response.text

        if gen_span:
            try:
                gen_span.update(output=answer, metadata={"model": self.model_name_pro})
                gen_span.end()
            except Exception:
                pass

        # 6. 출처 생성
        sources = self._generate_sources(documents, metadatas, distances, answer)

        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "document_count": len(documents)
        }

    # ========================================
    # paperbanana 도식화 재작성 메서드
    # ========================================

    def _rewrite_for_diagram(self, question: str, rag_answer: str, mode: str = "default") -> dict:
        """
        RAG 답변을 paperbanana 입력 포맷으로 LLM(Gemini Pro)을 통해 재작성.

        RAG 답변(한글 산문) → 영문 구조화 설명(source_context + communicative_intent)
        4가지 few-shot 예시 포함 프롬프트 사용 (prompts.py 'diagram_rewrite' 키)

        Args:
            question: 사용자 원본 질문
            rag_answer: /chat에서 생성된 RAG 텍스트 답변
            mode: "patent" → 특허 청구항 계층 구조로 강제 변환, 그 외 → 일반 도식화

        Returns:
            {"source_context": str, "communicative_intent": str}
            실패 시 빈 dict {} 반환
        """
        import json as _json
        import re as _re
        from prompts import PROMPT_TEMPLATES

        import time as _time
        _DIAGRAM_REWRITE_MODEL = "gemini-2.5-pro"
        prompt_key = "diagram_rewrite_patent" if mode == "patent" else "diagram_rewrite"
        logger.info(f"✏️ paperbanana 재작성 시작 ({_DIAGRAM_REWRITE_MODEL}, mode={mode})")

        prompt = PROMPT_TEMPLATES[prompt_key].format(
            question=question,
            rag_answer=rag_answer
        )

        raw = ""
        try:
            # 503 UNAVAILABLE 시 최대 3회 재시도 (5s, 10s, 15s 대기)
            response = None
            last_exc = None
            for _attempt in range(3):
                try:
                    response = self.genai_client.models.generate_content(
                        model=_DIAGRAM_REWRITE_MODEL,
                        contents=prompt
                    )
                    break
                except Exception as _e:
                    last_exc = _e
                    if "503" in str(_e) or "UNAVAILABLE" in str(_e):
                        wait = 5 * (_attempt + 1)
                        logger.warning(f"⚠️ {_DIAGRAM_REWRITE_MODEL} 503, {wait}s 후 재시도 ({_attempt+1}/3)")
                        _time.sleep(wait)
                    else:
                        raise
            if response is None:
                raise last_exc

            raw = response.text.strip()

            # LLM이 ```json ... ``` 또는 ``` ... ``` 블록으로 감쌀 경우 제거
            code_block = _re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
            if code_block:
                raw = code_block.group(1).strip()

            parsed = _json.loads(raw)

            source_context = parsed.get("source_context", "").strip()
            communicative_intent = parsed.get("communicative_intent", "").strip()

            if not source_context or not communicative_intent:
                logger.error(f"❌ diagram rewrite: 필수 키 누락 (source_context={bool(source_context)}, communicative_intent={bool(communicative_intent)})")
                return {}

            logger.info(f"  ✅ 재작성 완료: intent={communicative_intent[:80]}...")
            return {
                "source_context": source_context,
                "communicative_intent": communicative_intent
            }

        except _json.JSONDecodeError as e:
            logger.error(f"❌ diagram rewrite JSON 파싱 실패: {e}\n원본: {raw[:200]}")
            return {}
        except Exception as e:
            logger.error(f"❌ diagram rewrite 실패: {e}")
            return {}
