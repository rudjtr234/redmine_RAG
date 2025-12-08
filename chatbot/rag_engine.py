"""
Redmine RAG Engine
- ChromaDB 기반 Vector Store
- Sentence Transformers 임베딩
- Google Gemini API 활용
"""
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging
import re

logger = logging.getLogger(__name__)


class RedmineRAG:
    """Redmine 이슈 검색을 위한 RAG 엔진"""

    def __init__(self, vectordb_path: str, collection_name: str, gemini_api_key: str, redmine_url: str = None):
        """
        Args:
            vectordb_path: ChromaDB 데이터 경로
            collection_name: 컬렉션 이름
            gemini_api_key: Google Gemini API 키
            redmine_url: Redmine 서버 URL (선택)
        """
        logger.info(f"🔧 RAG 엔진 초기화...")
        logger.info(f"  - VectorDB 경로: {vectordb_path}")
        logger.info(f"  - Collection: {collection_name}")

        self.redmine_url = redmine_url or "https://redmine.192.168.20.150.nip.io:30443"

        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(path=vectordb_path)
        self.collection = self.client.get_collection(name=collection_name)

        # 임베딩 모델 초기화 (Vector DB와 동일한 모델 사용 필수!)
        logger.info("  - 임베딩 모델 로딩...")
        self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

        # Gemini API 초기화
        logger.info("  - Gemini API 설정...")
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-pro')

        logger.info("✅ RAG 엔진 초기화 완료!")

    def _is_version_or_comparison_query(self, question: str) -> bool:
        """버전 비교 또는 목록 조회 질문인지 판단"""
        patterns = [
            r"\b0\.\d", r"\bv\d", r"ver", r"version", r"버전",
            r"전체", r"목록", r"비교", r"차이", r"모든",
            r"최신", r"이전", r"변경", r"업데이트"
        ]
        return any(re.search(p, question, re.IGNORECASE) for p in patterns)

    def query(self, question: str, top_k: int = None, chat_history: list = None) -> dict:
        """
        질문에 대한 답변 생성 (Multi-turn 지원)

        Args:
            question: 사용자 질문
            top_k: 검색할 문서 개수 (None이면 자동 설정)
            chat_history: 이전 대화 히스토리 (선택)

        Returns:
            답변 및 관련 문서 정보
        """
        try:
            if chat_history is None:
                chat_history = []
            # 1. Top-K 자동 설정
            if top_k is None:
                if self._is_version_or_comparison_query(question):
                    top_k = 50  # 버전 비교/목록 조회
                    logger.info(f"📊 버전/비교 질문 감지 → top_k={top_k}")
                else:
                    top_k = 10  # 일반 질문
                    logger.info(f"📝 일반 질문 → top_k={top_k}")

            # 2. 질문 임베딩
            logger.info(f"🔍 검색 중... (top_k={top_k})")
            query_embedding = self.embedding_model.encode(question).tolist()

            # 3. Vector DB 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # 4. 검색 결과 정리
            documents = results['documents'][0] if results['documents'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            distances = results['distances'][0] if results['distances'] else []

            logger.info(f"  ✅ 검색된 문서: {len(documents)}개")

            if not documents:
                return {
                    "answer": "관련 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "question": question
                }

            # 5. 프롬프트 구성 (상위 문서만 사용)
            context_limit = min(15, len(documents))  # 최대 15개 문서만 컨텍스트로 사용
            context = "\n\n".join([
                f"[이슈 #{metadatas[i].get('issue_id')} - {metadatas[i].get('subject')}]\n{doc}"
                for i, doc in enumerate(documents[:context_limit])
            ])

            logger.info(f"  📄 컨텍스트에 사용된 문서: {context_limit}개")

            # 대화 히스토리 포맷팅
            history_text = ""
            if chat_history:
                history_text = "\n<대화_히스토리>\n"
                for i, turn in enumerate(chat_history[-5:], 1):  # 최근 5턴만 사용
                    history_text += f"[{i}턴 전]\n"
                    history_text += f"사용자: {turn['question']}\n"
                    history_text += f"어시스턴트: {turn['answer']}\n\n"
                history_text += "</대화_히스토리>\n"

            prompt = f"""당신은 MTS BIO-DT팀의 실험 데이터 검색 어시스턴트입니다.

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
4. **필수 포함 정보**: 모델명/버전, 성능지표, 이슈 번호
5. **근거 명시**: 각 정보 뒤에 "(Issue #번호)" 형식으로 출처 표기
6. **검색 문서 내 정보만 사용**: 추측이나 일반 지식 사용 금지
7. **실험 외 질문 거부**: "실험 데이터 검색만 가능합니다"라고 답변

예시 답변 형식:
- 단일 질문: "Aialpa-TSR-brst V0.4.0의 Validation Dice Score는 0.3018입니다. (Issue #496)"
- 비교 질문:
| 버전 | Dice Score | Issue |
|------|-----------|-------|
| v0.3.0 | 0.2850 | #450 |
| v0.4.0 | 0.3018 | #496 |
"""

            # 6. LLM 답변 생성
            logger.info("💬 답변 생성 중...")
            response = self.llm.generate_content(prompt)
            answer = response.text

            # 7. 결과 반환 (모든 검색된 문서 포함)
            sources = [
                {
                    "issue_id": meta.get("issue_id", "N/A"),
                    "subject": meta.get("subject", "N/A"),
                    "distance": float(dist),
                    "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                    "url": f"{self.redmine_url}/issues/{meta.get('issue_id', '')}" if meta.get("issue_id") else None
                }
                for meta, dist, doc in zip(metadatas, distances, documents)
            ]

            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "document_count": len(documents)
            }

        except Exception as e:
            logger.error(f"❌ 쿼리 처리 중 오류: {str(e)}")
            raise

    def get_document_count(self) -> int:
        """저장된 문서 개수 반환"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"문서 개수 조회 실패: {str(e)}")
            return 0