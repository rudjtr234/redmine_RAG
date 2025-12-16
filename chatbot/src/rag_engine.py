"""Redmine RAG Engine - ChromaDB + Gemini"""
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging
import re

logger = logging.getLogger(__name__)

class RedmineRAG:
    def __init__(self, vectordb_path: str, collection_name: str, gemini_api_key: str,
                 embedding_model: str = "sentence-transformers", redmine_url: str = None,
                 use_case: str = "redmine"):
        logger.info(f"🔧 RAG 엔진 초기화: {collection_name} ({embedding_model})")

        self.redmine_url = redmine_url or "https://redmine.192.168.20.150.nip.io:30443"
        self.embedding_type = embedding_model
        self.use_case = use_case

        self.client = chromadb.PersistentClient(path=vectordb_path)
        self.collection = self.client.get_collection(name=collection_name)

        try:
            self.conversation_collection = self.client.get_or_create_collection(
                name="conversation_history",
                metadata={"description": "Multi-turn conversation history"}
            )
            logger.info(f"  - 대화 이력: {self.conversation_collection.count()}개")
        except Exception as e:
            logger.warning(f"  ⚠️ 대화 이력 초기화 실패: {e}")
            self.conversation_collection = None

        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-pro')

        if embedding_model == "gemini":
            self.embedding_model_name = "models/text-embedding-004"
        else:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

        logger.info("✅ RAG 엔진 준비 완료!")

    def _embed(self, text: str, task_type: str):
        if self.embedding_type == "gemini":
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type=task_type
            )
            return result["embedding"]
        return self.embedding_model.encode(text).tolist()

    def _build_prompt(self, context: str, history_text: str, question: str) -> str:
        builder = self._build_redmine_prompt if self.use_case == "redmine" else self._build_document_prompt
        return builder(context, history_text, question)

    def _format_context(self, documents: list, metadatas: list, limit: int) -> str:
        if self.use_case == "redmine":
            return "\n\n".join(
                f"[이슈 #{m.get('issue_id')} - {m.get('subject')}]\n{doc}"
                for m, doc in zip(metadatas[:limit], documents[:limit])
            )
        return "\n\n".join(
            f"[문서: {m.get('filename', 'Unknown')} (청크 {m.get('chunk_index', 0)+1}/{m.get('total_chunks', 1)})]\n{doc}"
            for m, doc in zip(metadatas[:limit], documents[:limit])
        )

    def _build_redmine_prompt(self, context: str, history_text: str, question: str) -> str:
        """Redmine 이슈 검색용 프롬프트 생성"""
        return f"""당신은 MTS BIO-DT팀의 실험 데이터 검색 어시스턴트입니다.

<검색된_문서>
{context}
</검색된_문서>
{history_text}
<사용자_질문>
{question}
</사용자_질문>

답변 작성 지침:
1. **대화 맥락 활용**: 대화 히스토리가 있으면 맥락을 고려하여 답변 (예: "그것", "그 모델" 등의 지시어 해석)
2. **핵심만 간결하게**: 불필요한 인사말이나 부연설명 없이 질문에 대한 답만 제공
3. **구조화된 형식**:
   - 단일 결과: "모델명 v0.0.0의 Dice Score는 0.0000입니다. (Issue #000)"
   - 다중 결과: 마크다운 표 사용
4. **필수 포함 정보**: 모델명/버전, 성능지표, 이슈 번호, 날짜 (문서에 있는 경우)
5. **근거 명시**: 각 정보 뒤에 "(Issue #번호)" 형식으로 출처 표기
6. **검색 문서 내 정보만 사용**: 추측이나 일반 지식 사용 금지
7. **답변 가능한 질문 유형**:
   - 모델/실험 설명 및 개요 (예: "TMR 모델이란?", "TMR 모델에 대한 설명")
   - 성능 지표 조회 (예: "Dice Score는?")
   - 실험 환경 및 설정 (예: "하이퍼파라미터는?")
   - 문제 해결 사례 (예: "GPU 메모리 오류 해결 방법")
   - 모델 개선 이력 (예: "최근 변경사항")
8. **답변 불가능한 질문**: 검색된 문서에 없는 내용이거나 실험/모델과 무관한 질문 (예: "날씨", "맛집 추천")은 "검색된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변

예시 답변 형식:
- 단일 질문: "Aialpa-TSR-brst V0.4.0의 Validation Dice Score는 0.3018입니다. (Issue #496)"
- 비교 질문 (날짜 포함 가능):
| 버전 | Dice Score | 날짜 | Issue |
|------|-----------|------|-------|
| v0.3.0 | 0.2850 | 2024-03-15 | #450 |
| v0.4.0 | 0.3018 | 2024-04-20 | #496 |
"""

    def _build_document_prompt(self, context: str, history_text: str, question: str) -> str:
        """범용 문서 검색용 프롬프트 생성"""
        return f"""당신은 문서 검색 및 질의응답 어시스턴트입니다.

<검색된_문서>
{context}
</검색된_문서>
{history_text}
<사용자_질문>
{question}
</사용자_질문>

답변 작성 지침:
1. **대화 맥락 활용**: 대화 히스토리가 있으면 맥락을 고려하여 답변
2. **핵심만 간결하게**: 불필요한 인사말이나 부연설명 없이 질문에 대한 답만 제공
3. **검색 문서 내 정보만 사용**: 추측이나 일반 지식 사용 금지. 문서에 없는 내용은 "검색된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변
4. **근거 명시**: 정보의 출처를 명확히 표기 (예: "문서 ABC.pdf에 따르면...")
5. **구조화된 형식**: 여러 정보를 나열할 때는 마크다운 목록이나 표 사용
6. **한국어로 답변**: 모든 답변은 한국어로 작성

답변 예시:
- "딥러닝서버(DG5W)의 주요 요구사항은 다음과 같습니다:
  - CPU와 GPU를 단일 수냉시스템으로 냉각
  - 시스템 사용율과 온도를 자동 체크
  - 수냉시스템을 적응적으로 컨트롤
  (출처: 딥러닝(DG5W)시스템의 특징.docx)"
"""

    def _is_version_or_comparison_query(self, question: str) -> bool:
        return any(re.search(p, question, re.IGNORECASE) for p in [
            r"\b0\.\d", r"\bv\d", r"ver", r"version", r"버전",
            r"전체", r"목록", r"비교", r"차이", r"모든", r"최신", r"이전", r"변경", r"업데이트"
        ])

    def _is_specific_technical_query(self, question: str) -> bool:
        return any(re.search(p, question, re.IGNORECASE) for p in [
            r"pytorch", r"tensorflow", r"cuda", r"framework", r"환경", r"설정", r"config",
            r"파라미터", r"하이퍼파라미터", r"gpu", r"cpu", r"메모리", r"배치", r"epoch",
            r"optimizer", r"learning.?rate", r"loss", r"metric", r"데이터셋", r"dataset",
            r"모델 구조", r"architecture"
        ])

    def _build_search_query(self, question: str, chat_history: list) -> str:
        """
        최근 질문/답변까지 포함해 검색 쿼리를 구성해, 직전 턴의 버전·이슈 언급을 활용한다.
        """
        if not chat_history:
            return question

        recent_turns = chat_history[-2:]  # 마지막 2턴만 사용해 길이를 제한
        parts = [question]
        for turn in recent_turns:
            if turn.get("question"):
                parts.append(turn["question"])
            if turn.get("answer"):
                parts.append(turn["answer"])

        search_query = " ".join(parts)
        logger.info(f"  🔄 맥락 기반 검색: {search_query[:500]}")  # 로그 길이 제한
        return search_query

    def _extract_issue_ids(self, question: str) -> list:
        matches = re.findall(r'issue\s*#?(\d+)|이슈\s*#?(\d+)|#(\d+)', question, re.IGNORECASE)
        return list({str(int(g)) for m in matches for g in m if g})

    def _extract_version_tokens(self, text: str) -> list:
        """
        질문에서 버전 문자열(v0.7.0, 0.7.0 등)을 추출하여 검색 재정렬에 사용.
        """
        tokens = set()
        for match in re.findall(r'(v?\d+\.\d+(?:\.\d+)?)', text, re.IGNORECASE):
            tokens.add(match.lower())
            if match.lower().startswith('v'):
                tokens.add(match.lower()[1:])  # v0.7.0 → 0.7.0도 함께 저장
            else:
                tokens.add(f"v{match.lower()}")  # 0.7.0 → v0.7.0도 함께 저장
        return list(tokens)

    def _is_conversation_history_query(self, question: str) -> bool:
        return any(re.search(p, question, re.IGNORECASE) for p in [
            r"과거\s*(대화|질문|내용|이력|히스토리)", r"(대화|질문)\s*(목록|리스트|내역|이력)",
            r"전에\s*(물어본|질문한|했던)", r"이전\s*(대화|질문)", r"history|past\s*conversation",
            r"저장된\s*대화", r"대화\s*몇\s*개"
        ])

    def query(self, question: str, top_k: int = None, chat_history: list = None, session_id: str = None) -> dict:
        """
        질문에 대한 답변 생성 (Multi-turn 지원 + 과거 대화 검색)

        Args:
            question: 사용자 질문
            top_k: 검색할 문서 개수 (None이면 자동 설정)
            chat_history: 이전 대화 히스토리 (선택)
            session_id: 세션 ID (과거 대화 검색용)

        Returns:
            답변 및 관련 문서 정보
        """
        try:
            if chat_history is None:
                chat_history = []

            # 0-1. 대화 이력 조회 질문인지 확인
            if self._is_conversation_history_query(question):
                logger.info("💬 대화 이력 조회 질문 감지")
                if not session_id:
                    return {
                        "answer": "세션 ID가 없습니다. 다시 접속 후 질문해주세요.",
                        "sources": [],
                        "question": question
                    }

                # 현재 세션의 대화만 조회
                history_summary = self.get_conversation_history_summary(session_id=session_id)

                if history_summary.get('total_conversations', 0) == 0:
                    return {
                        "answer": "이 세션에서 저장된 대화 이력이 없습니다.",
                        "sources": [],
                        "question": question
                    }

                # 응답 포맷팅 - 현재 세션만 표시
                answer_lines = []
                answer_lines.append(f"**세션 대화 이력 ({session_id})**")
                answer_lines.append(f"- 대화 수: {history_summary['total_conversations']}개")
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

            # 0. 이슈 번호 직접 조회 우선 처리
            issue_ids = self._extract_issue_ids(question)
            direct_issue_results = None
            if issue_ids:
                logger.info(f"🔎 이슈 번호 감지 → {issue_ids}")
                # 문자열/정수 타입 혼용을 모두 커버
                issue_ids_str = [str(i) for i in issue_ids]
                issue_ids_int = []
                for i in issue_ids_str:
                    try:
                        issue_ids_int.append(int(i))
                    except ValueError:
                        pass

                # 1) 문자열 매칭
                direct_issue_results = self.collection.get(
                    where={"issue_id": {"$in": issue_ids_str}},
                    include=["metadatas", "documents", "embeddings"]
                )

                # 2) 없으면 정수 매칭 추가 시도
                if not direct_issue_results.get("documents") and issue_ids_int:
                    logger.info("  ➡️ 문자열 매칭 실패 → 정수형 매칭 재시도")
                    direct_issue_results = self.collection.get(
                        where={"issue_id": {"$in": issue_ids_int}},
                        include=["metadatas", "documents", "embeddings"]
                    )

                found = len(direct_issue_results.get("documents", []))
                if found:
                    logger.info(f"  ✅ 이슈 번호 매칭 성공: {found}건")
                else:
                    logger.info("  ⚠️ 해당 이슈를 찾지 못했습니다. 일반 검색으로 폴백합니다.")

            # 과거 대화 검색 (session_id가 있고, 대화 이력이 2턴 이상일 때)
            relevant_history = []
            if session_id and len(chat_history) >= 2:
                relevant_history = self.search_conversation_history(session_id, question, top_k=10)
                if relevant_history:
                    logger.info(f"  📚 관련 과거 대화: {len(relevant_history)}개 발견")
            # 1. Top-K 자동 설정 (적응형)
            if top_k is None:
                if self._is_version_or_comparison_query(question):
                    top_k = 50  # 버전 비교/목록 조회
                    logger.info(f"📊 버전/비교 질문 감지 → top_k={top_k}")
                elif self._is_specific_technical_query(question):
                    top_k = 30  # 기술적 특정 정보 검색 (환경, 버전, 설정 등)
                    logger.info(f"🔧 기술 검색 질문 감지 → top_k={top_k}")
                else:
                    top_k = 15  # 일반 질문 (기존 10 → 15로 증가)
                    logger.info(f"📝 일반 질문 → top_k={top_k}")

            # 2. 대화 맥락을 고려한 검색 쿼리 생성
            documents = []
            metadatas = []
            distances = []

            if direct_issue_results and direct_issue_results.get("documents"):
                # 이슈 번호 직접 조회 결과 사용
                documents = direct_issue_results.get("documents", [])
                metadatas = direct_issue_results.get("metadatas", [])
                distances = [0.0 for _ in documents]  # 직접 조회이므로 거리 0으로 설정
                logger.info(f"  ✅ 이슈 번호 매칭: {len(documents)}건")
            else:
                search_query = self._build_search_query(question, chat_history)

                # 3. 질문 임베딩
                logger.info(f"🔍 검색 중... (top_k={top_k})")
                query_embedding = self._embed(search_query, "RETRIEVAL_QUERY")

                # 4. Vector DB 검색
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )

                # 5. 검색 결과 정리
                documents = results.get('documents', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                distances = results.get('distances', [[]])[0]

            logger.info(f"  ✅ 검색된 문서: {len(documents)}개")

            if not documents:
                return {
                    "answer": "관련 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "question": question
                }

            # 5.1. 버전 토큰이 있으면 문서/메타데이터에 포함된 항목을 가중치로 재정렬
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

            # 5.5. "최신" 키워드 감지 시 날짜순 재정렬
            if re.search(r"최신|최근|가장\s*새로운|latest|recent", question, re.IGNORECASE):
                logger.info("  📅 최신 데이터 요청 감지 → 날짜순 재정렬")
                sorted_data = sorted(
                    zip(documents, metadatas, distances),
                    key=lambda x: x[1].get('updated_on', ''),
                    reverse=True  # 최신이 먼저
                )
                if sorted_data:
                    documents, metadatas, distances = zip(*sorted_data)
                    documents = list(documents)
                    metadatas = list(metadatas)
                    distances = list(distances)

            # 6. 프롬프트 구성 (상위 문서만 사용)
            context_limit = min(15, len(documents))  # 최대 15개 문서만 컨텍스트로 사용

            context = self._format_context(documents, metadatas, context_limit)

            logger.info(f"  📄 컨텍스트에 사용된 문서: {context_limit}개")

            # 대화 히스토리 포맷팅 (최근 대화 + 관련 과거 대화)
            history_text = ""
            if chat_history or relevant_history:
                history_text = "\n<대화_히스토리>\n"

                # 1. 최근 대화 (최근 3턴)
                if chat_history:
                    history_text += "[최근 대화]\n"
                    for i, turn in enumerate(chat_history[-3:], 1):
                        history_text += f"- 사용자: {turn['question']}\n"
                        history_text += f"  어시스턴트: {turn['answer']}\n"

                # 2. 관련 과거 대화 (검색된 것)
                if relevant_history:
                    history_text += "\n[관련 과거 대화]\n"
                    for hist in relevant_history[:10]:  # 최대 10개로 확장
                        history_text += f"- 사용자: {hist['question']}\n"
                        history_text += f"  어시스턴트: {hist['answer']}\n"

                history_text += "</대화_히스토리>\n"

            prompt = self._build_prompt(context, history_text, question)

            # 7. LLM 답변 생성
            logger.info("💬 답변 생성 중...")
            response = self.llm.generate_content(prompt)
            answer = response.text

            # 8. 출처 정보 생성 (use_case에 따라 다르게 처리)
            if self.use_case == "redmine":
                # Redmine: 답변에서 언급된 이슈 번호 추출
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

                # 답변에 언급된 이슈가 있으면 그것만, 없으면 상위 N개 반환
                if mentioned_issues:
                    filtered_sources = [
                        src for src in all_sources
                        if src["issue_id"] != "N/A" and int(src["issue_id"]) in mentioned_issues
                    ]
                    filtered_sources.sort(key=lambda x: int(x["issue_id"]))
                    logger.info(f"  📌 답변에 언급된 이슈: {len(filtered_sources)}개 (전체 검색: {len(documents)}개)")
                else:
                    # 언급된 이슈가 없으면 상위 5개 반환
                    top_n = min(5, len(all_sources))
                    filtered_sources = [src for src in all_sources[:top_n] if src["issue_id"] != "N/A"]
                    logger.info(f"  📌 참조 이슈 (언급 없음): {len(filtered_sources)}개 (전체 검색: {len(documents)}개)")
            else:
                # Document: 상위 N개의 관련 문서만 반환
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

            return {
                "answer": answer,
                "sources": filtered_sources,
                "question": question,
                "document_count": len(documents)
            }

        except Exception as e:
            logger.error(f"❌ 쿼리 처리 중 오류: {str(e)}")
            raise

    def save_conversation(self, session_id: str, turn_index: int, question: str, answer: str):
        """
        대화를 Vector DB에 저장

        Args:
            session_id: 세션 ID
            turn_index: 턴 번호
            question: 질문
            answer: 답변
        """
        if not self.conversation_collection:
            return

        try:
            from datetime import datetime, timedelta

            # 대화 문서 생성
            conversation_text = f"Q: {question}\nA: {answer}"

            embedding = self._embed(conversation_text, "RETRIEVAL_DOCUMENT")

            # TTL 설정 (2년 후 만료)
            ttl_expire = (datetime.now() + timedelta(days=730)).isoformat()

            # ID 생성
            doc_id = f"{session_id}_turn{turn_index}"

            # Vector DB에 저장
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

            logger.info(f"  💾 대화 저장: {session_id} 턴{turn_index}")

        except Exception as e:
            logger.error(f"❌ 대화 저장 실패: {str(e)}")

    def search_conversation_history(self, session_id: str, current_question: str, top_k: int = 5) -> list:
        """
        세션의 과거 대화에서 현재 질문과 관련된 내용 검색

        Args:
            session_id: 세션 ID
            current_question: 현재 질문
            top_k: 검색할 대화 수

        Returns:
            관련 대화 리스트
        """
        if not self.conversation_collection:
            return []

        try:
            # 대화가 하나도 없으면 Chroma가 빈 HNSW 세그먼트를 읽으려다 오류를 낼 수 있으니 바로 종료
            if self.conversation_collection.count() == 0:
                logger.info("  🔍 과거 대화 없음 (컬렉션 비어 있음)")
                return []

            # 현재 질문 임베딩
            query_embedding = self._embed(current_question, "RETRIEVAL_QUERY")

            # 같은 세션의 과거 대화 검색
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                where={"session_id": session_id},
                n_results=top_k
            )

            if not results['metadatas'] or not results['metadatas'][0]:
                return []

            # 결과 정리
            history = []
            for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                history.append({
                    "question": metadata.get("question", ""),
                    "answer": metadata.get("answer", ""),
                    "turn_index": metadata.get("turn_index", 0),
                    "timestamp": metadata.get("timestamp", ""),
                    "relevance_score": 1 - distance  # 거리를 유사도로 변환
                    
                })

            logger.info(f"  🔍 과거 대화 검색: {len(history)}개 발견 (세션: {session_id})")
            return history

        except Exception as e:
            logger.error(f"❌ 대화 검색 실패: {str(e)}")
            return []

    def get_document_count(self) -> int:
        """저장된 문서 개수 반환"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"문서 개수 조회 실패: {str(e)}")
            return 0

    def get_conversation_history_summary(self, session_id: str = None) -> dict:
        """
        대화 이력 요약 정보 반환

        Args:
            session_id: 특정 세션의 이력만 조회 (None이면 전체)

        Returns:
            대화 이력 요약 (총 개수, 세션 정보 등)
        """
        if not self.conversation_collection:
            return {
                "total_conversations": 0,
                "sessions": [],
                "message": "대화 이력 컬렉션이 없습니다."
            }

        try:
            # 전체 대화 이력 조회
            if session_id:
                results = self.conversation_collection.get(
                    where={"session_id": session_id},
                    include=["metadatas"]
                )
            else:
                results = self.conversation_collection.get(
                    include=["metadatas"]
                )

            if not results['metadatas']:
                return {
                    "total_conversations": 0,
                    "sessions": [],
                    "message": "저장된 대화 이력이 없습니다."
                }

            # 세션별로 그룹화
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

            # 각 세션의 대화를 턴 순서로 정렬
            for sid in sessions_data:
                sessions_data[sid]['conversations'].sort(key=lambda x: x['turn_index'])

            # 세션을 최신순으로 정렬
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
