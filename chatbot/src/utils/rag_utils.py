"""
RAG Utils - 범용 유틸리티 함수 모음

[역할]
이 파일은 RAG 엔진 전반에서 사용되는 **재사용 가능한 저수준 유틸리티 함수**들을 제공합니다.
패턴 매칭, 데이터 추출, 변환, 검색 쿼리 구성 등 독립적인 헬퍼 함수들을 모아놓은 공구함입니다.

[주요 기능]
1. 임베딩 및 프롬프트
   - 텍스트 임베딩, 프롬프트 빌딩, 컨텍스트 포맷팅
2. 질문 분류 (패턴 기반)
   - 일반 대화/CRF/Redmine/통계/메타데이터/최신 질문 등 판별
3. ID 및 토큰 추출
   - 이슈 번호, CRF record ID, 병원 코드, 버전 토큰 추출
   - 병원명↔코드 변환
4. 검색 쿼리 구성
   - 대화 맥락을 반영한 검색 쿼리 생성
5. 키워드 추출 및 필터링
   - 모델 키워드 캐싱, 추출, 필터링, 키워드 보강 검색
6. 정렬
   - 타임스탬프 파싱, 최신순 정렬
7. 대화 관리
   - 대화 저장/검색/요약, 사용자 목록 조회/삭제

[vs rag_engine_helpers.py]
- 이 파일은 범용 유틸리티 함수들을 제공 (공구함)
- rag_engine_helpers.py는 이 함수들을 조합하여 query() 메서드의 실행 흐름을 구현 (조립 매뉴얼)
"""
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from google.genai import types
from prompts import PROMPT_TEMPLATES
from config import patterns as P
from config import constants as C

logger = logging.getLogger(__name__)


class RAGHelperMixin:
    """
    RAG 엔진을 위한 범용 헬퍼 메서드 모음

    이 클래스는 재사용 가능한 저수준 유틸리티 함수들을 제공합니다.
    패턴 매칭, 데이터 추출/변환, 검색 쿼리 구성, 대화 관리 등의 독립적인 기능들을 담당합니다.
    """

    # 상수 가져오기
    SESSION_ID_PREFIX = C.SESSION_ID_PREFIX
    HOSPITAL_MAPPING = C.HOSPITAL_MAPPING
    HOSPITAL_PRIORITY = C.HOSPITAL_PRIORITY

    # 컴파일된 패턴 캐시 (클래스 변수)
    _compiled_patterns = {}

    @classmethod
    def _get_compiled_patterns(cls, pattern_name: str):
        """패턴을 컴파일하여 캐싱"""
        if pattern_name not in cls._compiled_patterns:
            pattern_list = getattr(P, pattern_name, [])
            cls._compiled_patterns[pattern_name] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]
        return cls._compiled_patterns[pattern_name]

    def _embed(self, text: str, task_type: str):
        lf = getattr(self, 'langfuse', None)
        lf_span = None
        if lf and getattr(self, '_current_trace', None):
            try:
                lf_span = lf.start_span(
                    name="embedding",
                    input={"text": text[:200], "task_type": task_type}
                )
            except Exception:
                lf_span = None

        result = self.genai_client.models.embed_content(
            model=self.embedding_model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )

        if lf_span:
            try:
                lf_span.end()
            except Exception:
                pass
        return result.embeddings[0].values

    def _build_prompt(self, context: str, history_text: str, question: str) -> str:
        template_key = {"redmine": "redmine", "crf": "crf"}.get(self.use_case, "document")
        return PROMPT_TEMPLATES[template_key].format(
            context=context,
            history_text=history_text,
            question=question
        )

    def _build_general_conversation_prompt(self, question: str) -> str:
        """일반 대화용 프롬프트"""
        return PROMPT_TEMPLATES["general"].format(question=question)

    _METRIC_PATTERN = re.compile(
        r'(?:AUC|F1(?:-score)?|Accuracy|Dice(?:\s*score)?|mAP\d*|Precision|Recall|Sensitivity|Specificity|AUROC|AP|IoU|Score|Loss)'
        r'(?:\s*[-:=|]\s*|\s+)'   # 구분자: : = - | 또는 공백
        r'(?:\d+\.?\d*\s*%?'       # 숫자 + 선택적 %
        r'|\|\s*\d+\.?\d*)',       # 마크다운 표: | 0.91
        re.IGNORECASE
    )

    def _format_context(self, documents: list, metadatas: list, limit: int) -> str:
        if self.use_case == "redmine":
            parts = []
            for m, doc in zip(metadatas[:limit], documents[:limit]):
                try:
                    att_ids = json.loads(m.get("attachment_ids", "[]"))
                except Exception:
                    att_ids = []
                has_image = len(att_ids) > 0
                has_metrics = bool(self._METRIC_PATTERN.search(doc))
                header = f"[이슈 #{m.get('issue_id')} - {m.get('subject')}]"
                if has_image:
                    header += (
                        f"\n[HAS_IMAGE_ATTACHMENTS=true]"
                        f"\n[IMAGE_ATTACHMENT_COUNT={len(att_ids)}]"
                        f"\n[TEXT_METRICS_PRESENT={'true' if has_metrics else 'false'}]"
                    )
                parts.append(f"{header}\n{doc}")
            return "\n\n".join(parts)
        if self.use_case == "crf":
            return "\n\n".join(
                "[CRF {record_id} | 병원 {hospital} | 시트 {sheet}]\n{doc}".format(
                    record_id=m.get("record_id", "N/A"),
                    hospital=m.get("hospital", "N/A"),
                    sheet=m.get("sheet", "N/A"),
                    doc=doc,
                )
                for m, doc in zip(metadatas[:limit], documents[:limit])
            )
        return "\n\n".join(
            f"[문서: {m.get('filename', 'Unknown')} (청크 {m.get('chunk_index', 0)+1}/{m.get('total_chunks', 1)})]\n{doc}"
            for m, doc in zip(metadatas[:limit], documents[:limit])
        )

    # ========================================
    # 질문 분류 메서드 (패턴 기반)
    # ========================================

    def _is_general_conversation(self, question: str) -> bool:
        """일반 대화인지 판별"""
        patterns = self._get_compiled_patterns('GENERAL_CONVERSATION_PATTERNS')
        return any(p.search(question.strip().lower()) for p in patterns)

    def is_crf_data_query(self, question: str) -> bool:
        """CRF/임상 데이터 관련 질문인지 판별"""
        # 병원명 패턴 동적 생성
        hospital_patterns = [re.escape(name) for name in self.HOSPITAL_MAPPING.keys()]

        # 모든 CRF 관련 패턴 합치기
        all_patterns = (
            P.CRF_BASE_PATTERNS +
            hospital_patterns +
            P.CRF_HOSPITAL_CODE_PATTERNS +
            P.CRF_MEDICAL_PATTERNS +
            P.CRF_FIELD_PATTERNS
        )
        return any(re.search(p, question, re.IGNORECASE) for p in all_patterns)

    def is_redmine_data_query(self, question: str) -> bool:
        """Redmine 이슈 관련 질문인지 판별"""
        patterns = self._get_compiled_patterns('REDMINE_QUERY_PATTERNS')
        return any(p.search(question) for p in patterns)

    def _is_conversation_history_query(self, question: str) -> bool:
        """대화 이력 조회 질문인지 판별"""
        patterns = self._get_compiled_patterns('CONVERSATION_HISTORY_PATTERNS')
        return any(p.search(question) for p in patterns)

    def _is_version_or_comparison_query(self, question: str) -> bool:
        """버전/비교 질문인지 판별"""
        patterns = self._get_compiled_patterns('VERSION_COMPARISON_PATTERNS')
        return any(p.search(question) for p in patterns)

    def _is_specific_technical_query(self, question: str) -> bool:
        """기술적 질문인지 판별"""
        patterns = self._get_compiled_patterns('TECHNICAL_QUERY_PATTERNS')
        return any(p.search(question) for p in patterns)

    def _is_metadata_query(self, question: str) -> bool:
        """메타데이터 질문 감지"""
        patterns = self._get_compiled_patterns('METADATA_QUERY_PATTERNS')
        return any(p.search(question) for p in patterns)

    def _is_sample_query(self, question: str) -> bool:
        """
        소수 사례(샘플)만 요청하는 질문인지 판별
        - "사례/케이스", "보여줘/알려줘", "n개만/건만" 등의 표현이 있는 경우 통계로 분류하지 않음
        """
        patterns = self._get_compiled_patterns('SAMPLE_QUERY_PATTERNS')
        return any(p.search(question) for p in patterns)

    def _is_statistics_query(self, question: str) -> bool:
        """통계/요약 질문 감지"""
        if self._is_metadata_query(question):
            return False
        # 사례 샘플 요청(소량 보여줘)은 통계가 아닌 검색 경로로 보낸다
        if self._is_sample_query(question):
            return False
        patterns = self._get_compiled_patterns('STATISTICS_QUERY_PATTERNS')
        return any(p.search(question) for p in patterns)

    def _is_recent_query(self, question: str) -> bool:
        """최신 질문인지 판별"""
        return bool(P.RECENT_QUERY_PATTERN.search(question))

    # ========================================
    # 추출 메서드 (ID, 토큰 등)
    # ========================================

    def _extract_issue_ids(self, question: str) -> list:
        """이슈 번호 추출"""
        matches = P.ISSUE_ID_PATTERN.findall(question)
        return list({str(int(g)) for m in matches for g in m if g})

    def _extract_crf_record_ids(self, question: str) -> list:
        """CRF record ID 추출"""
        matches = P.CRF_RECORD_PATTERN.findall(question)
        return list({m.upper() for m in matches})

    def _extract_hospital_code_from_question(self, question: str) -> str:
        """질문에서 병원 코드 추출"""
        normalized_question = question.replace(" ", "")

        for hospital_name in self.HOSPITAL_PRIORITY:
            normalized_name = hospital_name.replace(" ", "")
            if normalized_name in normalized_question:
                code = self.HOSPITAL_MAPPING.get(hospital_name)
                if code:
                    logger.info(f"  🏥 병원 감지: {hospital_name} (코드: {code})")
                    return code
        return None


    def _convert_hospital_names_to_codes(self, text: str) -> str:
        """병원명을 코드로 변환"""
        converted_text = text
        for hospital_name in self.HOSPITAL_PRIORITY:
            if hospital_name in converted_text:
                code = self.HOSPITAL_MAPPING.get(hospital_name)
                if code:
                    converted_text = converted_text.replace(
                        hospital_name,
                        f"{hospital_name} {code}"
                    )
        return converted_text

    def _chunk_statistics_text(self, text: str, max_chars: int = 80000) -> list:
        """통계/메타데이터 텍스트를 고정 길이로 강제 청크 분할"""
        if len(text) <= max_chars:
            return [text]
        chunks = [
            text[i:i + max_chars]
            for i in range(0, len(text), max_chars)
        ]
        if C.DEBUG_CONFIG.get("crf_chunk_logging"):
            logger.info(f"  📦 청크 분할: 총 {len(chunks)}개 (입력 길이: {len(text):,}, 청크 최대: {max_chars})")
            if chunks:
                logger.info(f"    - 첫 청크 길이: {len(chunks[0])}, 마지막 청크 길이: {len(chunks[-1])}")
        return chunks

    # ========================================
    # 검색 및 필터링 메서드
    # ========================================

    def _build_search_query(self, question: str, chat_history: list) -> str:
        """검색 쿼리 구성 (대화 맥락 포함)"""
        if self.use_case == "crf":
            question = self._convert_hospital_names_to_codes(question)

        if not chat_history:
            return question

        recent_turns = chat_history[-C.SEARCH_QUERY_CONFIG["recent_turns_to_include"]:]
        parts = [question]

        for turn in recent_turns:
            if turn.get("question"):
                hist_question = turn["question"]

                # use_case별 필터링
                if self.use_case == "crf":
                    if self.is_crf_data_query(hist_question):
                        hist_question = self._convert_hospital_names_to_codes(hist_question)
                        parts.append(hist_question)
                elif self.use_case == "redmine":
                    if not self.is_crf_data_query(hist_question):
                        parts.append(hist_question)
                else:
                    if not self.is_crf_data_query(hist_question):
                        parts.append(hist_question)

            if turn.get("answer"):
                answer = turn["answer"]
                if self.use_case == "crf":
                    if any(kw in answer for kw in C.SEARCH_QUERY_CONFIG["crf_answer_keywords"]):
                        parts.append(answer[:200])
                elif self.use_case == "redmine":
                    if any(kw in answer for kw in C.SEARCH_QUERY_CONFIG["redmine_answer_keywords"]) and not any(
                        kw in answer for kw in C.SEARCH_QUERY_CONFIG["redmine_answer_exclude_keywords"]
                    ):
                        parts.append(answer[:200])

        search_query = " ".join(parts)
        logger.info(f"  🔄 맥락 기반 검색 (use_case={self.use_case}): {search_query[:300]}")
        return search_query

    def _get_model_keyword_cache(self):
        """모델 키워드 캐시 생성 및 관리"""
        cache = getattr(self, "_model_keyword_cache", None)
        cache_count = getattr(self, "_model_keyword_cache_count", None)

        try:
            current_count = self.collection.count()
        except Exception:
            current_count = None

        if cache is not None and (current_count is None or cache_count == current_count):
            return cache

        keywords = set()
        try:
            results = self.collection.get(include=["metadatas"])
            for meta in results.get("metadatas", []):
                subject = str(meta.get("subject", ""))
                for token in P.MODEL_KEYWORD_PATTERN.findall(subject):
                    if P.VERSION_FILTER_PATTERN.fullmatch(token):
                        continue
                    keywords.add(token.lower())
                    for part in re.split(r"[-_]", token):
                        if len(part) > 2 and not P.VERSION_FILTER_PATTERN.fullmatch(part):
                            keywords.add(part.lower())
            self._model_keyword_cache = keywords
            self._model_keyword_cache_count = current_count
        except Exception as e:
            logger.warning(f"  ⚠️ 모델 키워드 캐시 생성 실패: {e}")
            self._model_keyword_cache = set()

        return self._model_keyword_cache

    def _extract_model_keywords(self, question: str) -> list:
        """모델 키워드 추출"""
        cache = self._get_model_keyword_cache()
        if not cache:
            return []

        found = set()
        for token in P.MODEL_KEYWORD_PATTERN.findall(question):
            if P.VERSION_FILTER_PATTERN.fullmatch(token):
                continue
            key = token.lower()
            if key in cache:
                found.add(key)
            for part in re.split(r"[-_]", token):
                part_key = part.lower()
                if len(part) > 2 and part_key in cache:
                    found.add(part_key)

        return list(found)

    def _contains_keywords(self, documents: list, metadatas: list, keywords: list) -> bool:
        """문서에 키워드 포함 여부 확인"""
        if not keywords:
            return False

        keywords_lower = [k.lower() for k in keywords]
        for doc, meta in zip(documents, metadatas):
            subject = str(meta.get("subject", "")).lower() if meta else ""
            doc_lower = str(doc).lower()
            for keyword in keywords_lower:
                if keyword in subject or keyword in doc_lower:
                    return True
        return False

    def _filter_by_keywords(self, documents: list, metadatas: list, distances: list, keywords: list):
        """키워드로 문서 필터링"""
        if not keywords:
            return documents, metadatas, distances

        matched_docs, matched_metas, matched_dists = [], [], []
        keywords_lower = [k.lower() for k in keywords]

        for doc, meta, dist in zip(documents, metadatas, distances):
            subject = str(meta.get("subject", "")).lower() if meta else ""
            doc_lower = str(doc).lower()
            if any(keyword in subject or keyword in doc_lower for keyword in keywords_lower):
                matched_docs.append(doc)
                matched_metas.append(meta)
                matched_dists.append(dist)

        if matched_docs:
            logger.info(f"  ✅ 키워드 일치 문서 필터링: {len(matched_docs)}개")
            return matched_docs, matched_metas, matched_dists

        return documents, metadatas, distances

    def _augment_with_keyword_matches(self, documents: list, metadatas: list, distances: list, keywords: list,
                                      limit_per_keyword: int = None):
        """키워드 보강 검색"""
        if limit_per_keyword is None:
            limit_per_keyword = C.KEYWORD_SEARCH_CONFIG['limit_per_keyword']

        if not keywords or self._contains_keywords(documents, metadatas, keywords):
            return documents, metadatas, distances

        existing_issue_ids = set()
        for meta in metadatas:
            if meta and meta.get("issue_id") is not None:
                existing_issue_ids.add(str(meta.get("issue_id")))

        extra_docs, extra_metas, extra_dists = [], [], []

        for keyword in keywords:
            try:
                result = self.collection.get(
                    where_document={"$contains": keyword},
                    include=["documents", "metadatas"],
                    limit=limit_per_keyword
                )
            except Exception as e:
                logger.warning(f"  ⚠️ 키워드 보강 검색 실패: {keyword} ({e})")
                continue

            docs = result.get("documents", [])
            metas = result.get("metadatas", [])
            for doc, meta in zip(docs, metas):
                issue_id = str(meta.get("issue_id")) if meta and meta.get("issue_id") is not None else None
                if issue_id and issue_id in existing_issue_ids:
                    continue
                if issue_id:
                    existing_issue_ids.add(issue_id)
                extra_docs.append(doc)
                extra_metas.append(meta)
                extra_dists.append(1.0)

        if extra_docs:
            logger.info(f"  🔎 키워드 보강 검색 추가: {len(extra_docs)}개")
            documents = list(documents) + extra_docs
            metadatas = list(metadatas) + extra_metas
            distances = list(distances) + extra_dists

        return documents, metadatas, distances

    # ========================================
    # 정렬 메서드
    # ========================================

    def _parse_timestamp(self, value: str):
        """타임스탬프 파싱"""
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00'))
        except Exception:
            return None

    def _ensure_staff_latest_issues(self, documents: list, metadatas: list, distances: list, question: str = ""):
        """단체 쿼리 시 지정 사원별 최신 이슈를 DB에서 직접 조회해 앞에 보장

        벡터 유사도와 무관하게 JUNIOR_STAFF 각 사원의 가장 최신 이슈를
        현재 문서 목록 앞에 추가(이미 있으면 중복 제거).
        질문에 '기획' 키워드가 있으면 PLANNING_STAFF도 포함.
        """
        existing_ids = set()
        for meta in metadatas:
            if meta:
                issue_id = str(meta.get('issue_id', ''))
                if issue_id:
                    existing_ids.add(issue_id)

        guaranteed_docs, guaranteed_metas, guaranteed_dists = [], [], []

        staff_list = list(C.JUNIOR_STAFF)
        if "기획" in question:
            staff_list += list(C.PLANNING_STAFF)

        for staff_name in staff_list:
            try:
                results = self.collection.get(
                    where={"author_name": {"$eq": staff_name}},
                    include=["metadatas", "documents"]
                )
            except Exception:
                continue

            if not results['ids']:
                continue

            items = list(zip(results['ids'], results['metadatas'], results['documents']))
            # 제목에 주간/보고/회의 키워드 포함된 이슈만
            report_items = [
                item for item in items
                if any(kw in (item[1] or {}).get('subject', '') for kw in C.STAFF_REPORT_SUBJECT_KEYWORDS)
            ]
            if not report_items:
                continue

            report_items.sort(key=lambda x: x[1].get('updated_on', '') if x[1] else '', reverse=True)

            latest_id, latest_meta, latest_doc = report_items[0]

            if str(latest_id) not in existing_ids:
                guaranteed_docs.append(latest_doc)
                guaranteed_metas.append(latest_meta)
                guaranteed_dists.append(0.0)
                existing_ids.add(str(latest_id))
                logger.info(f"    ➕ {staff_name} 최신 이슈 보장: #{latest_id} ({latest_meta.get('updated_on','')[:10]})")

        if guaranteed_docs:
            return guaranteed_docs + documents, guaranteed_metas + metadatas, guaranteed_dists + distances

        return documents, metadatas, distances

    def _sort_by_recency(self, documents: list, metadatas: list, distances: list):
        """최신순 정렬"""
        if not metadatas:
            return documents, metadatas, distances

        scored = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            timestamp = None
            if meta:
                timestamp = meta.get('updated_on') or meta.get('created_on')
            parsed = self._parse_timestamp(timestamp) if timestamp else None
            scored.append((parsed, doc, meta, dist))

        scored.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
        return (
            [s[1] for s in scored],
            [s[2] for s in scored],
            [s[3] for s in scored],
        )

    def _promote_latest_per_author(self, documents: list, metadatas: list, distances: list):
        """작성자별 가장 최신 이슈를 앞으로 올림 (최신 쿼리 전용)

        날짜순 정렬 후에도 특정 작성자의 최신 이슈가 뒤로 밀릴 수 있으므로,
        각 작성자의 가장 최신 이슈를 먼저 모은 뒤 나머지를 붙인다.
        """
        if not metadatas:
            return documents, metadatas, distances

        seen_authors = {}
        promoted = []   # (timestamp, doc, meta, dist) — 작성자별 최신 1개
        rest = []       # 나머지

        for doc, meta, dist in zip(documents, metadatas, distances):
            author = (meta or {}).get('author_name', '') if meta else ''
            timestamp = None
            if meta:
                timestamp = meta.get('updated_on') or meta.get('created_on')
            parsed = self._parse_timestamp(timestamp) if timestamp else None

            if author and author not in seen_authors:
                seen_authors[author] = True
                promoted.append((parsed, doc, meta, dist))
            else:
                rest.append((parsed, doc, meta, dist))

        # promoted도 최신순 정렬
        promoted.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)

        merged = promoted + rest
        return (
            [s[1] for s in merged],
            [s[2] for s in merged],
            [s[3] for s in merged],
        )

    # ========================================
    # 대화 관리 메서드
    # ========================================

    def save_conversation(self, session_id: str, turn_index: int, question: str, answer: str):
        """대화를 Vector DB에 저장"""
        if not self.conversation_collection:
            return

        try:
            conversation_text = f"Q: {question}\nA: {answer}"
            embedding = self._embed(conversation_text, "RETRIEVAL_DOCUMENT")

            ttl_expire = (datetime.now() + timedelta(days=C.CHAT_HISTORY_CONFIG['ttl_days'])).isoformat()
            timestamp_id = int(time.time() * 1000000)
            doc_id = f"{session_id}_{timestamp_id}"

            self.conversation_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[{
                    "session_id": session_id,
                    "turn_index": turn_index,
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "answer": answer,
                    "ttl_expire": ttl_expire
                }]
            )

            logger.info(f"  💾 대화 저장: {doc_id}")

        except Exception as e:
            logger.error(f"❌ 대화 저장 실패: {str(e)}")

    def search_conversation_history(self, session_id: str, current_question: str, top_k: int = None) -> list:
        """세션의 과거 대화에서 현재 질문과 관련된 내용 검색"""
        if top_k is None:
            top_k = C.CHAT_HISTORY_CONFIG['max_relevant_history']

        if not self.conversation_collection:
            return []

        try:
            if self.conversation_collection.count() == 0:
                logger.info("  🔍 과거 대화 없음 (컬렉션 비어 있음)")
                return []

            query_embedding = self._embed(current_question, "RETRIEVAL_QUERY")

            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                where={"session_id": session_id},
                n_results=top_k
            )

            if not results['metadatas'] or not results['metadatas'][0]:
                return []

            history = []
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                history.append({
                    "question": metadata.get("question", ""),
                    "answer": metadata.get("answer", ""),
                    "turn_index": metadata.get("turn_index", 0),
                    "timestamp": metadata.get("timestamp", ""),
                    "relevance_score": 1 - distance
                })

            logger.info(f"  🔍 과거 대화 검색: {len(history)}개 발견 (세션: {session_id})")
            return history

        except Exception as e:
            logger.error(f"❌ 대화 검색 실패: {str(e)}")
            return []

    def get_conversation_history_summary(self, session_id: str = None) -> dict:
        """대화 이력 요약 정보 반환"""
        if not self.conversation_collection:
            return {
                "total_conversations": 0,
                "sessions": [],
                "message": "대화 이력 컬렉션이 없습니다."
            }

        try:
            if session_id:
                results = self.conversation_collection.get(
                    where={"session_id": session_id},
                    include=["metadatas"]
                )
            else:
                results = self.conversation_collection.get(include=["metadatas"])

            if not results['metadatas']:
                return {
                    "total_conversations": 0,
                    "sessions": [],
                    "message": "저장된 대화 이력이 없습니다."
                }

            sessions_data = {}
            for metadata in results['metadatas']:
                sid = metadata.get('session_id', 'Unknown')
                if sid not in sessions_data:
                    sessions_data[sid] = {
                        'session_id': sid,
                        'conversation_count': 0,
                        'conversations': [],
                        'first_timestamp': None,
                        'last_timestamp': None
                    }

                sessions_data[sid]['conversation_count'] += 1
                sessions_data[sid]['conversations'].append({
                    'turn_index': metadata.get('turn_index', 0),
                    'timestamp': metadata.get('timestamp', ''),
                    'question': metadata.get('question', ''),
                    'answer': metadata.get('answer', '')[:100] + '...' if len(metadata.get('answer', '')) > 100 else metadata.get('answer', '')
                })

                timestamp = metadata.get('timestamp', '')
                if not sessions_data[sid]['first_timestamp'] or timestamp < sessions_data[sid]['first_timestamp']:
                    sessions_data[sid]['first_timestamp'] = timestamp
                if not sessions_data[sid]['last_timestamp'] or timestamp > sessions_data[sid]['last_timestamp']:
                    sessions_data[sid]['last_timestamp'] = timestamp

            for sid in sessions_data:
                sessions_data[sid]['conversations'].sort(key=lambda x: x['turn_index'])

            sessions_list = sorted(sessions_data.values(), key=lambda x: x['last_timestamp'], reverse=True)

            return {
                "total_conversations": len(results['metadatas']),
                "total_sessions": len(sessions_data),
                "sessions": sessions_list
            }

        except Exception as e:
            logger.error(f"대화 이력 조회 실패: {str(e)}")
            return {
                "total_conversations": 0,
                "sessions": [],
                "error": str(e)
            }

    def get_user_list(self) -> list:
        """대화 로그에서 사용자 목록 추출"""
        if not self.conversation_collection:
            return []

        try:
            if self.conversation_collection.count() == 0:
                logger.info("  📋 사용자 없음 (대화 로그 비어 있음)")
                return []

            results = self.conversation_collection.get()

            if not results or not results.get('metadatas'):
                return []

            user_stats = defaultdict(lambda: {
                'user_name': '',
                'total_conversations': 0,
                'first_seen': None,
                'last_seen': None
            })

            for metadata in results['metadatas']:
                session_id = metadata.get('session_id', '')
                timestamp = metadata.get('timestamp', '')

                if session_id.startswith(self.SESSION_ID_PREFIX):
                    user_name = session_id.replace(self.SESSION_ID_PREFIX, '')

                    user_stats[user_name]['user_name'] = user_name
                    user_stats[user_name]['total_conversations'] += 1

                    try:
                        ts = datetime.fromisoformat(timestamp)
                        if not user_stats[user_name]['first_seen'] or ts < user_stats[user_name]['first_seen']:
                            user_stats[user_name]['first_seen'] = ts
                        if not user_stats[user_name]['last_seen'] or ts > user_stats[user_name]['last_seen']:
                            user_stats[user_name]['last_seen'] = ts
                    except:
                        pass

            kst = timezone(timedelta(hours=9))
            user_list = []
            for user_name, stats in user_stats.items():
                first_seen_kst = stats['first_seen'].astimezone(kst) if stats['first_seen'] else None
                last_seen_kst = stats['last_seen'].astimezone(kst) if stats['last_seen'] else None

                user_list.append({
                    'user_name': stats['user_name'],
                    'total_conversations': stats['total_conversations'],
                    'first_seen': first_seen_kst.isoformat() if first_seen_kst else None,
                    'last_seen': last_seen_kst.isoformat() if last_seen_kst else None,
                })

            user_list.sort(key=lambda x: x['last_seen'] or '', reverse=True)

            logger.info(f"  📋 사용자 목록 조회: {len(user_list)}명")
            return user_list

        except Exception as e:
            logger.error(f"❌ 사용자 목록 조회 실패: {str(e)}")
            return []

    def delete_user(self, user_name: str) -> bool:
        """특정 사용자의 모든 대화 로그 삭제"""
        if not self.conversation_collection:
            return False

        try:
            session_id = f"{self.SESSION_ID_PREFIX}{user_name}"

            results = self.conversation_collection.get(
                where={"session_id": session_id}
            )

            if not results or not results.get('ids'):
                logger.info(f"  📋 삭제할 데이터 없음: {user_name}")
                return True

            ids_to_delete = results['ids']
            self.conversation_collection.delete(ids=ids_to_delete)

            logger.info(f"  ✅ 사용자 삭제 완료: {user_name} ({len(ids_to_delete)}개 대화)")
            return True

        except Exception as e:
            logger.error(f"❌ 사용자 삭제 실패: {str(e)}")
            return False

    def get_document_count(self) -> int:
        """저장된 문서 개수 반환"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"문서 개수 조회 실패: {str(e)}")
            return 0
