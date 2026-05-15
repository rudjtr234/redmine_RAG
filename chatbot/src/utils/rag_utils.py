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
        import time as _time
        t0 = _time.time()

        # OpenAI 임베딩
        if getattr(self, 'embedding_mode', 'gemini') == 'openai':
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            elapsed = _time.time() - t0
            usage = response.usage
            logger.debug(f"  [embed/openai] tokens={usage.total_tokens} elapsed={elapsed:.3f}s")
            # Langfuse span 기록
            lf = getattr(self, 'langfuse', None)
            if lf and getattr(self, '_current_trace', None):
                try:
                    sp = lf.start_span(name="embedding", input={"text": text[:200]})
                    sp.end(metadata={
                        "provider": "openai",
                        "model": self.embedding_model_name,
                        "total_tokens": usage.total_tokens,
                        "elapsed_s": round(elapsed, 3),
                    })
                except Exception:
                    pass
            return response.data[0].embedding

        # Gemini 임베딩 (기존)
        result = self.genai_client.models.embed_content(
            model=self.embedding_model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type)
        )
        elapsed = _time.time() - t0
        logger.debug(f"  [embed/gemini] elapsed={elapsed:.3f}s")
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

    _MD_IMG_RE = re.compile(r'(!\[[^\]]*\]\()([^)]+)(\))')
    _HTML_IMG_RE = re.compile(r'<img[^>]*>', re.IGNORECASE)
    _IMG_EXTS = ('.png', '.jpg', '.jpeg', '.gif', '.webp')
    _MAX_IMG_PER_ISSUE = 3
    _MAX_IMG_TOTAL = 5

    @staticmethod
    def _normalize_img_fname(ref: str) -> str:
        """URL decode + 파일명만 추출 + strip + lower (양쪽 동일 규칙)"""
        from urllib.parse import unquote
        return unquote(ref).split('/')[-1].strip().lower()

    def _format_context(self, documents: list, metadatas: list, limit: int) -> str:
        if self.use_case == "redmine":
            parts = []
            total_img_count = 0

            for m, doc in zip(metadatas[:limit], documents[:limit]):
                try:
                    att_ids = json.loads(m.get("attachment_ids", "[]"))
                except Exception:
                    att_ids = []
                try:
                    att_ctypes = json.loads(m.get("attachment_content_types", "{}"))
                except Exception:
                    att_ctypes = {}
                try:
                    att_fnames = json.loads(m.get("attachment_filenames", "{}"))
                except Exception:
                    att_fnames = {}

                has_metrics = bool(self._METRIC_PATTERN.search(doc))
                issue_id = m.get("issue_id", "")
                tracker  = m.get("tracker_name", "")
                status   = m.get("status_name", "")
                assigned = m.get("assigned_to_name", "")
                author   = m.get("author_name", "")
                project  = m.get("project_name", "")
                date_val = (m.get("updated_on") or m.get("created_on") or "")[:10]
                header_parts = [f"[이슈 #{issue_id} - {m.get('subject', '제목 없음')}]"]
                meta_parts = " | ".join(filter(None, [
                    f"유형:{tracker}"      if tracker  else "",
                    f"프로젝트:{project}"  if project  else "",
                    f"상태:{status}"       if status   else "",
                    f"작성:{author}"       if author   else "",
                    f"담당:{assigned}"     if assigned else "",
                    f"메타날짜:{date_val}" if date_val else "",
                ]))
                if meta_parts:
                    header_parts.append(f"[{meta_parts}]")
                header = "\n".join(header_parts)

                # 이미지 attachment 선별: MIME → 확장자 fallback
                image_att_ids = [
                    aid for aid in att_ids
                    if att_ctypes.get(str(aid), "").startswith("image/")
                ]
                if not image_att_ids:
                    image_att_ids = [
                        aid for aid in att_ids
                        if att_fnames.get(str(aid), "").lower().endswith(self._IMG_EXTS)
                    ]

                # fname → id 역매핑 (양쪽 동일 normalize 규칙)
                # att_fnames key는 문자열, image_att_ids는 정수 → str 변환 필요
                from urllib.parse import unquote as _unquote
                image_att_id_strs = {str(aid) for aid in image_att_ids}
                fname_to_id = {
                    _unquote(v).strip().lower(): k
                    for k, v in att_fnames.items() if k in image_att_id_strs
                }

                if image_att_ids:
                    header += (
                        f"\n[HAS_IMAGE_ATTACHMENTS=true]"
                        f"\n[TEXT_METRICS_PRESENT={'true' if has_metrics else 'false'}]"
                    )

                # === 1순위: 본문 이미지 URL 정규화 ===
                # 클로저 대신 리스트로 카운터 관리 (nonlocal 스코프 문제 방지)
                counters = [0]  # [issue_img_count]
                seen_aids = set()

                def _md_replacer(match, _ftoi=fname_to_id, _ctrs=counters, _seen=seen_aids):
                    ref_str = match.group(2)
                    if _ctrs[0] >= self._MAX_IMG_PER_ISSUE or total_img_count >= self._MAX_IMG_TOTAL:
                        return ""
                    if ref_str.startswith('http'):
                        return ""
                    fname = self._normalize_img_fname(ref_str)
                    aid = _ftoi.get(fname)
                    if aid and aid not in _seen:
                        _seen.add(aid)
                        _ctrs[0] += 1
                        return f"{match.group(1)}/redmine-image/{aid}{match.group(3)}"
                    return ""

                new_doc = self._MD_IMG_RE.sub(_md_replacer, doc)
                new_doc = self._HTML_IMG_RE.sub("", new_doc)  # raw <img> 차단
                has_inline_img = counters[0] > 0
                total_img_count += counters[0]

                block = f"{header}\n{new_doc}"
                # 본문에 명시적으로 참조된 이미지만 context에 포함
                # attachment만 있고 본문 마크다운 참조 없는 이미지는 제외 (실험결과 스크린샷 등 난잡함 방지)

                parts.append(block)
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
        q_lower = question.lower()

        # Redmine 프로젝트 컨텍스트(AI 모델 개발)가 감지되면 CRF 아님
        if any(re.search(p, question, re.IGNORECASE) for p in P.CRF_EXCLUSION_PATTERNS):
            return False

        # 암종+HRD/모델명이 함께 있으면 Redmine AI 프로젝트 질문으로 처리
        # (임상 데이터 질문이 아닌 모델 개발 프로젝트 질문)
        cancer_model_pairs = [
            (["난소암", "ovarian"], ["hrd"]),
            (["유방암", "breast"], ["lnmp", "림프절 전이"]),
        ]
        for cancer_kws, model_kws in cancer_model_pairs:
            has_cancer = any(kw in q_lower for kw in cancer_kws)
            has_model_kw = any(kw in q_lower for kw in model_kws)
            if has_cancer and has_model_kw:
                # 임상 데이터 특화 키워드가 없으면 Redmine으로 처리
                crf_specific = ["환자", "record id", "병리번호", "병원코드", "수술",
                                "진단명", "임상 데이터", "crf", "her2", "ihc"]
                has_crf_specific = any(kw in q_lower for kw in crf_specific)
                if not has_crf_specific:
                    return False

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

    def _classify_recent_intent(self, question: str) -> str:
        """최신 질문의 의도를 분류
        Returns:
            'experiment' — 최신 모델/실험 결과 (날짜 정렬만, 작성자 승격 없음)
            'report'     — 최신 보고/회의 (날짜 정렬 + 작성자 승격)
            'none'       — 최신 질문 아님
        """
        if not self._is_recent_query(question):
            return 'none'
        if P.RECENT_EXPERIMENT_PATTERN.search(question):
            return 'experiment'
        if P.RECENT_REPORT_PATTERN.search(question):
            return 'report'
        # 패턴 미분류: 기본적으로 report 방식 적용 (기존 동작 유지)
        return 'report'

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

    # 본문 텍스트에서 병원코드 추출 패턴
    # 위암: "병원코드.1: 02"  →  병원코드\.1:\s*(\d{2})
    # 대장암: "Unnamed: 0: 01: 신촌"  →  Unnamed:\s*0[^\d]*0*(\d{1,2})
    # 공통: "병원 번호: 02"
    _HOSPITAL_CODE_FROM_TEXT_PATTERNS = [
        re.compile(r'병원코드\.1:\s*(\d{2})', re.IGNORECASE),           # 위암 전용
        re.compile(r'Unnamed:\s*0[^\d]*0*(\d{1,2})', re.IGNORECASE),   # 대장암
        re.compile(r'병원\s*번호[^\d]*(\d{2})', re.IGNORECASE),
    ]
    # HRD: "Pt no. : CMC-02" — 기관코드-숫자 형식에서 숫자 직접 사용
    # 숫자가 없는 경우만 기관코드 매핑으로 fallback
    _HRD_INSTITUTION_MAP = {
        'CMC': '02', 'SS': '01', 'YS': '01', 'KU': '06',
        'SCH': '06', 'IU': '07', 'GS': '04',
    }

    def _extract_hospital_code_from_text(self, text: str) -> str:
        """본문 텍스트에서 병원코드 추출 (metadata.hospital 없는 암종용 fallback)"""
        if not text:
            return None
        # 위암/대장암 패턴
        for pat in self._HOSPITAL_CODE_FROM_TEXT_PATTERNS:
            m = pat.search(text)
            if m:
                code = m.group(1).zfill(2)
                return code
        # HRD: "Pt no. : CMC-02" → 숫자 직접 추출 우선, 없으면 기관코드 매핑
        m = re.search(r'Pt\s*no\.?\s*[:\-]\s*([A-Z]{2,4})-(\d{2})', text, re.IGNORECASE)
        if m:
            return m.group(2).zfill(2)
        m = re.search(r'Pt\s*no\.?\s*[:\-]\s*([A-Z]{2,4})', text, re.IGNORECASE)
        if m:
            inst = m.group(1).upper()
            return self._HRD_INSTITUTION_MAP.get(inst)
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

    def _filter_by_hospital_with_fallback(self, documents: list, metadatas: list, distances: list,
                                          hospital_code: str) -> tuple:
        """병원 필터 — metadata.hospital 없는 암종은 본문 텍스트에서 파싱하여 필터
        stomach/colorectal/hrd처럼 metadata.hospital이 비어 있는 경우 fallback 적용.
        매칭 row가 하나도 없으면 필터 없이 전체 반환 (과도한 필터링 방지).
        """
        if not hospital_code or not documents:
            return documents, metadatas, distances

        filtered = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            meta_hospital = (meta or {}).get('hospital', '')
            if meta_hospital:
                # metadata에 hospital 있으면 그대로 비교
                if meta_hospital == hospital_code:
                    filtered.append((doc, meta, dist))
            else:
                # metadata 없으면 본문 텍스트 파싱 fallback
                parsed = self._extract_hospital_code_from_text(doc or '')
                if parsed == hospital_code:
                    filtered.append((doc, meta, dist))

        if not filtered:
            # 하나도 안 걸리면 필터 미적용 (데이터 정규화 전 암종 보호)
            logger.info(f"  ⚠️ 병원 fallback 필터 결과 0건 → 필터 해제 (코드: {hospital_code})")
            return documents, metadatas, distances

        logger.info(f"  🏥 병원 fallback 필터 적용: {len(documents)}건 → {len(filtered)}건 (코드: {hospital_code})")
        return (
            [r[0] for r in filtered],
            [r[1] for r in filtered],
            [r[2] for r in filtered],
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

    def save_conversation(self, session_id: str, turn_index: int, question: str, answer: str,
                          conversation_id: str = None, sources_summary: list = None,
                          answer_kind: str = "rag"):
        """대화를 Vector DB에 저장"""
        if not self.conversation_collection:
            return

        try:
            import json as _json_mod
            conversation_text = f"Q: {question}\nA: {answer}"
            embedding = self._embed(conversation_text, "RETRIEVAL_DOCUMENT")

            ttl_expire = (datetime.now() + timedelta(days=C.CHAT_HISTORY_CONFIG['ttl_days'])).isoformat()
            timestamp_id = int(time.time() * 1000000)
            doc_id = f"{session_id}_{timestamp_id}"

            metadata = {
                "session_id": session_id,
                "turn_index": turn_index,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "ttl_expire": ttl_expire,
                "answer_kind": answer_kind,
            }
            if conversation_id:
                metadata["conversation_id"] = conversation_id
            if sources_summary:
                metadata["sources_summary"] = _json_mod.dumps(sources_summary, ensure_ascii=False)

            self.conversation_collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[conversation_text],
                metadatas=[metadata]
            )

            logger.info(f"  💾 대화 저장: {doc_id} (conv: {conversation_id}, kind: {answer_kind})")

        except Exception as e:
            logger.error(f"❌ 대화 저장 실패: {str(e)}")

    def save_diagram_to_turn(self, session_id: str, conversation_id: str, turn_index: int,
                             image_base64: str, task_id: str = '', mode: str = 'default') -> bool:
        """완료된 도식화 이미지를 파일시스템에 저장하고 경로를 메타데이터에 기록"""
        if not self.conversation_collection:
            return False
        try:
            import json as _json_mod
            import os as _os
            import base64 as _b64

            # turn_index를 int로 강제 변환
            try:
                turn_index_int = int(turn_index)
            except (TypeError, ValueError):
                logger.warning(f"  ⚠️ turn_index 변환 실패: {turn_index}")
                return False

            where_filter = {"$and": [
                {"session_id": session_id},
                {"conversation_id": conversation_id},
                {"turn_index": turn_index_int}
            ]}
            results = self.conversation_collection.get(where=where_filter, include=["metadatas"])
            if not results['ids']:
                logger.warning(f"  ⚠️ 도식화 저장 대상 턴 없음: conv={conversation_id} turn={turn_index_int}")
                return False
            doc_id = results['ids'][0]
            meta = results['metadatas'][0]

            # base64 이미지를 파일시스템에 저장 (ChromaDB 메타데이터 크기 제한 우회)
            diagrams_dir = _os.environ.get("DIAGRAMS_DIR", "/vectordb/diagrams")
            _os.makedirs(diagrams_dir, exist_ok=True)
            safe_task = (task_id or "unknown").replace("/", "_")
            filename = f"{conversation_id}_{turn_index_int}_{safe_task}.png"
            filepath = _os.path.join(diagrams_dir, filename)

            # data:image/png;base64,... 형식에서 순수 base64 추출
            raw_b64 = image_base64
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                img_bytes = _b64.b64decode(raw_b64)
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
            except Exception as e:
                logger.error(f"  ❌ 도식화 파일 저장 실패: {e}")
                return False

            # 메타데이터에는 파일 경로만 저장
            diagrams_raw = meta.get('diagrams')
            try:
                diagrams = _json_mod.loads(diagrams_raw) if diagrams_raw else []
            except Exception:
                diagrams = []
            diagrams.append({"task_id": task_id, "mode": mode, "file_path": filepath})
            meta['diagrams'] = _json_mod.dumps(diagrams, ensure_ascii=False)
            self.conversation_collection.update(
                ids=[doc_id],
                metadatas=[meta]
            )
            logger.info(f"  🖼 도식화 저장 완료: {filepath} (mode={mode}, task={task_id})")
            return True
        except Exception as e:
            logger.error(f"❌ 도식화 저장 실패: {str(e)}")
            return False

    def search_conversation_history(self, session_id: str, current_question: str, top_k: int = None,
                                     conversation_id: str = None) -> list:
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

            if conversation_id:
                where_filter = {"$and": [{"session_id": session_id}, {"conversation_id": conversation_id}]}
            else:
                where_filter = {"session_id": session_id}

            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                where=where_filter,
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

    def get_conversations_list(self, session_id: str) -> list:
        """사용자의 대화 스레드 목록 반환 (conversation_id 기반)"""
        if not self.conversation_collection:
            return []

        try:
            import json as _json_mod
            results = self.conversation_collection.get(
                where={"session_id": session_id},
                include=["metadatas"]
            )
            if not results['metadatas']:
                return []

            conv_map = {}
            legacy_turns = []

            for meta in results['metadatas']:
                conv_id = meta.get("conversation_id")
                ts = meta.get("timestamp", "")
                question = meta.get("question", "")

                # soft delete된 대화는 목록에서 제외
                if meta.get("deleted") == 1:
                    continue

                if not conv_id:
                    # conversation_id 없는 기존 데이터 → legacy 그룹
                    legacy_turns.append({"timestamp": ts, "question": question})
                    continue

                if conv_id not in conv_map:
                    conv_map[conv_id] = {"first_ts": ts, "last_ts": ts, "first_question": question}
                else:
                    if ts < conv_map[conv_id]["first_ts"]:
                        conv_map[conv_id]["first_ts"] = ts
                        conv_map[conv_id]["first_question"] = question
                    if ts > conv_map[conv_id]["last_ts"]:
                        conv_map[conv_id]["last_ts"] = ts

            result = []
            for conv_id, info in conv_map.items():
                title = info["first_question"][:30] if info["first_question"] else "새 대화"
                result.append({
                    "conversation_id": conv_id,
                    "title": title,
                    "last_timestamp": info["last_ts"]
                })

            # legacy 항목: 기존 데이터가 있으면 목록에 추가 후 함께 최신순 정렬
            if legacy_turns:
                last_ts = max(t["timestamp"] for t in legacy_turns)
                result.append({
                    "conversation_id": "legacy",
                    "title": "이전 대화",
                    "last_timestamp": last_ts
                })

            result.sort(key=lambda x: x["last_timestamp"] or "", reverse=True)
            return result

        except Exception as e:
            logger.error(f"❌ 대화 목록 조회 실패: {str(e)}")
            return []

    def get_conversation_by_id(self, session_id: str, conversation_id: str) -> dict:
        """특정 대화 스레드의 전체 메시지 반환"""
        if not self.conversation_collection:
            return {"messages": [], "title": "새 대화", "last_timestamp": None}

        try:
            import json as _json_mod

            if conversation_id == "legacy":
                where_filter = {"session_id": session_id}
            else:
                where_filter = {"$and": [{"session_id": session_id}, {"conversation_id": conversation_id}]}

            results = self.conversation_collection.get(
                where=where_filter,
                include=["metadatas"]
            )

            if not results['metadatas']:
                return {"messages": [], "title": "새 대화", "last_timestamp": None}

            messages = []
            for meta in results['metadatas']:
                # legacy 조회 시 conversation_id 있는 항목은 제외
                if conversation_id == "legacy" and meta.get("conversation_id"):
                    continue

                sources_raw = meta.get("sources_summary")
                sources = []
                if sources_raw:
                    try:
                        sources = _json_mod.loads(sources_raw)
                    except Exception:
                        sources = []

                diagrams_raw = meta.get('diagrams')
                try:
                    diagrams = _json_mod.loads(diagrams_raw) if diagrams_raw else []
                except Exception:
                    diagrams = []

                exports_raw = meta.get('exports')
                try:
                    exports = _json_mod.loads(exports_raw) if exports_raw else []
                except Exception:
                    exports = []

                messages.append({
                    "turn_index": meta.get("turn_index", 0),
                    "timestamp": meta.get("timestamp", ""),
                    "question": meta.get("question", ""),
                    "answer": meta.get("answer", ""),
                    "sources_summary": sources,
                    "diagrams": diagrams,
                    "exports": exports,
                })

            if not messages:
                return {"messages": [], "title": "새 대화", "last_timestamp": None}

            messages.sort(key=lambda x: int(x["turn_index"]))

            title = messages[0]["question"][:30] if messages[0]["question"] else "새 대화"
            if conversation_id == "legacy":
                title = "이전 대화"
            last_timestamp = max(m["timestamp"] for m in messages)

            return {
                "messages": messages,
                "title": title,
                "last_timestamp": last_timestamp
            }

        except Exception as e:
            logger.error(f"❌ 대화 조회 실패: {str(e)}")
            return {"messages": [], "title": "새 대화", "last_timestamp": None}

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
                'total_turns': 0,
                'first_seen': None,
                'last_seen': None
            })

            for metadata in results['metadatas']:
                session_id = metadata.get('session_id', '')
                timestamp = metadata.get('timestamp', '')

                if session_id.startswith(self.SESSION_ID_PREFIX):
                    user_name = session_id.replace(self.SESSION_ID_PREFIX, '')

                    user_stats[user_name]['user_name'] = user_name
                    user_stats[user_name]['total_turns'] += 1

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
                    'total_turns': stats['total_turns'],  # 저장된 QA 턴 수 (스레드 수 아님)
                    'first_seen': first_seen_kst.isoformat() if first_seen_kst else None,
                    'last_seen': last_seen_kst.isoformat() if last_seen_kst else None,
                })

            user_list.sort(key=lambda x: x['last_seen'] or '', reverse=True)

            logger.info(f"  📋 사용자 목록 조회: {len(user_list)}명")
            return user_list

        except Exception as e:
            logger.error(f"❌ 사용자 목록 조회 실패: {str(e)}")
            return []

    def delete_conversation(self, session_id: str, conversation_id: str) -> bool:
        """특정 conversation_id를 soft delete (데이터 보존, 목록에서만 숨김)"""
        if not self.conversation_collection:
            return False
        try:
            results = self.conversation_collection.get(
                where={"$and": [{"session_id": session_id}, {"conversation_id": conversation_id}]},
                include=["metadatas", "documents", "embeddings"]
            )
            if not results or not results.get('ids'):
                logger.info(f"  📋 대화 없음 (soft delete 스킵): {conversation_id}")
                return True
            # 각 turn의 메타에 deleted=1 추가 후 upsert (데이터/임베딩 보존)
            updated_metas = []
            for meta in results['metadatas']:
                m = dict(meta)
                m['deleted'] = 1
                updated_metas.append(m)
            self.conversation_collection.upsert(
                ids=results['ids'],
                embeddings=results['embeddings'],
                documents=results['documents'],
                metadatas=updated_metas
            )
            logger.info(f"  ✅ 대화 soft delete: {conversation_id} ({len(results['ids'])}개 턴)")
            return True
        except Exception as e:
            logger.error(f"❌ 대화 soft delete 실패: {str(e)}")
            return False

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

    def get_latest_answers(self, session_id: str, conversation_id: str,
                           count=1, answer_kind: str = "rag") -> list:
        """최신 N개(count=None이면 전체) non-deleted RAG 답변 반환 (오래된 순)"""
        if not self.conversation_collection:
            return []
        try:
            where_filter = {"$and": [
                {"session_id": session_id},
                {"conversation_id": conversation_id},
            ]}
            results = self.conversation_collection.get(
                where=where_filter,
                include=["metadatas"]
            )
            if not results or not results["metadatas"]:
                return []

            not_deleted = [
                m for m in results["metadatas"]
                if int(m.get("deleted", 0)) != 1
            ]
            candidates = [
                m for m in not_deleted
                if m.get("answer_kind", "rag") == answer_kind
            ]
            if not candidates and answer_kind == "rag":
                candidates = [m for m in not_deleted if "answer_kind" not in m]

            candidates.sort(key=lambda m: int(m.get("turn_index", 0)))
            if count is not None:
                candidates = candidates[-count:]  # 최신 N개

            import json as _jm
            result = []
            for m in candidates:
                diagrams_raw = m.get("diagrams")
                try:
                    diagrams = _jm.loads(diagrams_raw) if diagrams_raw else []
                except Exception:
                    diagrams = []
                result.append({
                    "question": m.get("question", ""),
                    "answer": m.get("answer", ""),
                    "turn_index": int(m.get("turn_index", 0)),
                    "diagrams": diagrams,
                })
            return result
        except Exception as e:
            logger.error(f"❌ latest answers 조회 실패: {str(e)}")
            return []

    def get_latest_answer(self, session_id: str, conversation_id: str,
                          answer_kind: str = "rag") -> dict:
        """conversation의 최신 non-deleted turn 중 answer_kind 일치하는 것 반환"""
        if not self.conversation_collection:
            return {}
        try:
            where_filter = {"$and": [
                {"session_id": session_id},
                {"conversation_id": conversation_id},
            ]}
            results = self.conversation_collection.get(
                where=where_filter,
                include=["metadatas"]
            )
            if not results or not results["metadatas"]:
                return {}

            # deleted 필드 없는 기존 turn도 포함 (필드 있으면 1이 아닌 것만)
            not_deleted = [
                m for m in results["metadatas"]
                if int(m.get("deleted", 0)) != 1
            ]
            candidates = [
                m for m in not_deleted
                if m.get("answer_kind", "rag") == answer_kind
            ]
            if not candidates:
                # answer_kind 필드가 없는 구버전 turn도 rag로 간주
                if answer_kind == "rag":
                    candidates = [
                        m for m in results["metadatas"]
                        if "answer_kind" not in m
                    ]
            if not candidates:
                return {}

            latest = max(candidates, key=lambda m: int(m.get("turn_index", 0)))
            return {
                "question": latest.get("question", ""),
                "answer": latest.get("answer", ""),
                "turn_index": int(latest.get("turn_index", 0)),
            }
        except Exception as e:
            logger.error(f"❌ latest answer 조회 실패: {str(e)}")
            return {}

    def save_export_to_turn(self, session_id: str, conversation_id: str,
                             turn_index: int, export_id: str, file_path: str) -> bool:
        """DOCX export 정보를 해당 turn metadata에 append (save_diagram_to_turn 패턴 동일)"""
        if not self.conversation_collection:
            return False
        try:
            import json as _json_mod
            from datetime import timezone as _tz

            turn_index_int = int(turn_index)
            where_filter = {"$and": [
                {"session_id": session_id},
                {"conversation_id": conversation_id},
                {"turn_index": turn_index_int},
            ]}
            results = self.conversation_collection.get(where=where_filter, include=["metadatas"])
            if not results["ids"]:
                logger.warning(f"  ⚠️ export 저장 대상 턴 없음: conv={conversation_id} turn={turn_index_int}")
                return False

            doc_id = results["ids"][0]
            meta = results["metadatas"][0]

            exports_raw = meta.get("exports")
            try:
                exports = _json_mod.loads(exports_raw) if exports_raw else []
            except Exception:
                exports = []
            exports.append({
                "export_id": export_id,
                "file_path": file_path,
                "created_at": datetime.now(_tz.utc).isoformat(),
            })
            meta["exports"] = _json_mod.dumps(exports, ensure_ascii=False)
            self.conversation_collection.update(ids=[doc_id], metadatas=[meta])
            logger.info(f"  📄 export 저장 완료: {file_path} (export_id={export_id})")
            return True
        except Exception as e:
            logger.error(f"❌ export 저장 실패: {str(e)}")
            raise

    def is_conversation_deleted(self, session_id: str, conversation_id: str) -> bool:
        """conversation의 모든 turn이 soft-delete 됐는지 확인"""
        if not self.conversation_collection:
            return False
        try:
            where_filter = {"$and": [
                {"session_id": session_id},
                {"conversation_id": conversation_id},
            ]}
            results = self.conversation_collection.get(
                where=where_filter,
                include=["metadatas"]
            )
            if not results or not results["metadatas"]:
                return True  # 존재하지 않으면 삭제된 것으로 처리
            # 하나라도 deleted=1이면 삭제된 것으로 처리
            return all(int(m.get("deleted", 0)) == 1 for m in results["metadatas"])
        except Exception as e:
            logger.error(f"❌ conversation 삭제 여부 확인 실패: {str(e)}")
            return False
