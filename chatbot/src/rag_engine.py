"""Redmine RAG Engine - ChromaDB + Gemini (리팩토링 버전)"""
import logging
import os
import chromadb
from google import genai
from sentence_transformers import SentenceTransformer
from utils.rag_utils import RAGHelperMixin
from utils.crf_statistics import CRFStatisticsMixin
from utils.rag_engine_helpers import QueryHelperMixin
from config import constants as C


logger = logging.getLogger(__name__)

class RedmineRAG(RAGHelperMixin, CRFStatisticsMixin, QueryHelperMixin):
    def __init__(self, vectordb_path: str, collection_name: str, gemini_api_key: str,
                 embedding_model: str = "sentence-transformers", redmine_url: str = None,
                 use_case: str = "redmine", conversation_db_path: str = None,
                 crf_collection_name: str = None):
        logger.info(f"🔧 RAG 엔진 초기화: {collection_name} ({embedding_model})")

        self.redmine_url = redmine_url or "https://redmine.<INTERNAL-IP>.nip.io:30443"
        self.embedding_type = embedding_model
        self.use_case = use_case

        # 메인 DB 클라이언트
        self.client = chromadb.PersistentClient(path=vectordb_path)

        # use_case별 컬렉션 분리
        if use_case == "crf":
            crf_col = crf_collection_name or "crf_all_cancers_v0_3_1"
            self.collection = self.client.get_collection(name=crf_col)
            logger.info(f"  - CRF 컬렉션: {crf_col}")
        else:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"  - 메인 컬렉션: {collection_name}")

        # 대화 이력 DB (별도 경로)
        try:
            conversation_path = conversation_db_path or os.path.join(os.path.dirname(vectordb_path), "conversation_db")
            self.conversation_client = chromadb.PersistentClient(path=conversation_path)
            self.conversation_collection = self.conversation_client.get_or_create_collection(
                name="conversation_history",
                metadata={"description": "Multi-turn conversation history"}
            )
            logger.info(f"  - 대화 이력 DB: {conversation_path}")
            logger.info(f"  - 대화 이력: {self.conversation_collection.count()}개")
        except Exception as e:
            logger.warning(f"  ⚠️ 대화 이력 초기화 실패: {e}")
            self.conversation_collection = None

        # Gemini Client 생성
        self.genai_client = genai.Client(api_key=gemini_api_key)
        # Code Execution/차트용: 최신 Flash (빠르고 안정적)
        self.model_name = 'gemini-3-flash-preview'
        # Q&A용: 안정적인 2.5-pro
        self.model_name_pro = 'gemini-2.5-pro'

        logger.info(f"✅ Gemini Client 초기화 완료 (모델: {self.model_name_pro})")

        if embedding_model == "gemini":
            # gemini-embedding-001: 안정 버전 (권장)
            self.embedding_model_name = "models/gemini-embedding-001"
        else:
            self.embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

        logger.info("✅ RAG 엔진 준비 완료!")

    def query(self, question: str, top_k: int = None, chat_history: list = None, session_id: str = None) -> dict:
        """
        질문에 대한 답변 생성 (Multi-turn 지원 + 과거 대화 검색)
        리팩토링: 헬퍼 메서드 활용으로 간결화 (532줄 → 60줄)
        """
        try:
            if chat_history is None:
                chat_history = []

            recent_query = self._is_recent_query(question)

            # 1. 특수 질문 타입 처리 (early return)
            if self._is_general_conversation(question):
                return self._handle_general_conversation(question)

            if self._is_conversation_history_query(question):
                return self._handle_conversation_history_query(question, session_id)

            # 2. CRF 메타데이터 질문 (초기 체크)
            if self.use_case == "crf" and self._is_metadata_query(question):
                hospital_code = self._extract_hospital_code_from_question(question)
                return self._handle_crf_metadata_query(question, hospital_code, chat_history)

            # 3. CRF 통계/차트 질문은 바로 처리 (벡터 검색 생략)
            if self.use_case == "crf" and self._is_statistics_query(question):
                hospital_code = self._extract_hospital_code_from_question(question)
                return self._handle_crf_statistics_query(question, hospital_code)

            # 4. 직접 조회 시도 (이슈 번호 또는 CRF record_id)
            direct_results = self._perform_direct_lookup(question)

            # 5. 과거 대화 검색
            relevant_history = []
            if session_id and len(chat_history) >= C.CHAT_HISTORY_CONFIG['search_history_threshold']:
                relevant_history = self.search_conversation_history(
                    session_id, question, 
                    top_k=C.CHAT_HISTORY_CONFIG['max_relevant_history']
                )
                if relevant_history:
                    logger.info(f"  📚 관련 과거 대화: {len(relevant_history)}개 발견")

            # 6. Top-K 결정
            top_k = self._determine_top_k(question, top_k, recent_query)

            # 7. 문서 검색
            documents, metadatas, distances = self._search_documents(
                question, chat_history, direct_results, top_k
            )

            if not documents:
                return {
                    "answer": "관련 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "question": question
                }

            # 8. 문서 후처리 (버전 재정렬, 최신순 정렬, 키워드 보강)
            documents, metadatas, distances = self._post_process_documents(
                documents, metadatas, distances, question, recent_query
            )

            # 9. 컨텍스트 구성 및 답변 생성
            return self._generate_answer(
                question, documents, metadatas, distances, 
                chat_history, relevant_history
            )

        except Exception as e:
            logger.error(f"❌ 쿼리 처리 중 오류: {str(e)}")
            raise

    def compare_collection_similarity(self, question: str) -> dict:
        """컬렉션 유사도 비교 (라우팅용)"""
        try:
            query_embedding = self._embed(question, "RETRIEVAL_QUERY")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )
            distances = results.get('distances', [[]])[0]
            current_distance = distances[0] if distances else float('inf')
            return {
                'distance': current_distance,
                'collection_name': self.collection.name
            }
        except Exception as e:
            logger.error(f"❌ 유사도 비교 중 오류: {str(e)}")
            return {
                'distance': float('inf'),
                'collection_name': self.collection.name
            }
