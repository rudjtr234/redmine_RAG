"""Redmine RAG Engine - ChromaDB + Gemini/OpenAI 전환 가능"""
import logging
import os
import re
import chromadb
from google import genai
try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None
try:
    from langfuse import Langfuse as _Langfuse
except ImportError:
    _Langfuse = None
from rank_bm25 import BM25Okapi
from utils.rag_utils import RAGHelperMixin
from utils.crf_statistics import CRFStatisticsMixin
from utils.rag_engine_helpers import QueryHelperMixin
from config import constants as C


logger = logging.getLogger(__name__)

class RedmineRAG(RAGHelperMixin, CRFStatisticsMixin, QueryHelperMixin):
    def __init__(self, vectordb_path: str, collection_name: str, gemini_api_key: str,
                 redmine_url: str = None, use_case: str = "redmine",
                 conversation_db_path: str = None, crf_collection_name: str = None):
        embedding_mode = os.environ.get("EMBEDDING_MODEL", "gemini")
        logger.info(f"🔧 RAG 엔진 초기화: {collection_name} (embedding={embedding_mode})")

        self.redmine_url = redmine_url or "https://your-redmine.example.com"
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

        # Gemini Client (답변 생성 / CRF 통계 등 공통 사용)
        self.genai_client = genai.Client(api_key=gemini_api_key)

        # LLM 모델 선택: EMBEDDING_MODEL=openai 이면 gpt-5.5, 아니면 gemini
        embedding_mode = os.environ.get("EMBEDDING_MODEL", "gemini")
        if embedding_mode == "openai":
            openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            if _OpenAI and openai_api_key:
                self.openai_client = _OpenAI(api_key=openai_api_key)
                self.model_name     = "gpt-5.5"
                self.model_name_pro = "gpt-5.5"
                self.embedding_model_name = "text-embedding-3-large"  # 3072차원
                self.embedding_mode = "openai"
                logger.info(f"✅ OpenAI Client 초기화 완료 (LLM={self.model_name_pro}, embed={self.embedding_model_name})")
            else:
                raise RuntimeError("OPENAI_API_KEY 없음 또는 openai 패키지 미설치")
        else:
            self.openai_client    = None
            self.model_name       = "gemini-2.5-pro"
            self.model_name_pro   = "gemini-2.5-pro"
            self.embedding_model_name = "models/gemini-embedding-001"
            self.embedding_mode   = "gemini"
            logger.info(f"✅ Gemini Client 초기화 완료 (LLM={self.model_name_pro}, embed={self.embedding_model_name})")

        # BM25 인덱스 초기화 (Redmine 전용)
        self.bm25_index = None
        self.bm25_doc_ids = []
        self.bm25_corpus = []
        if use_case == "redmine" and C.BM25_CONFIG.get("enabled", False):
            self._build_bm25_index()

        # Langfuse 초기화 (환경변수 없으면 비활성화)
        self._current_trace = None
        lf_host = os.environ.get("LANGFUSE_HOST")
        lf_pub  = os.environ.get("LANGFUSE_PUBLIC_KEY")
        lf_sec  = os.environ.get("LANGFUSE_SECRET_KEY")
        if _Langfuse and lf_host and lf_pub and lf_sec:
            try:
                self.langfuse = _Langfuse(host=lf_host, public_key=lf_pub, secret_key=lf_sec)
                logger.info(f"✅ Langfuse 초기화 완료 ({lf_host})")
            except Exception as e:
                logger.warning(f"⚠️ Langfuse 초기화 실패: {e}")
                self.langfuse = None
        else:
            self.langfuse = None

        logger.info("✅ RAG 엔진 준비 완료!")

    def _lf_trace_end(self, output: str = None, status: str = "success", error_message: str = None):
        """Langfuse trace span 종료 (공통 헬퍼)"""
        span = getattr(self, '_lf_span', None)
        if not span:
            return
        try:
            meta = {"status": status}
            if error_message:
                meta["error_message"] = error_message
            kwargs = {"metadata": meta}
            if output is not None:
                kwargs["output"] = output
            span.update(**kwargs)
            span.end()
        except Exception:
            pass
        self._lf_span = None
        self._current_trace = None

    def query(self, question: str, chat_history: list = None, session_id: str = None,
              engine_name: str = None, route_reason: str = None, conversation_id: str = None) -> dict:
        """
        질문에 대한 답변 생성 (Multi-turn 지원 + 과거 대화 검색)

        변경: top_k 파라미터 제거 → candidate_k + 유사도 컷오프 방식으로 자동 결정
        """
        try:
            if chat_history is None:
                chat_history = []

            # Langfuse trace 시작 (start_span 직접 호출 방식)
            self._lf_span = None
            self._current_trace = None
            if self.langfuse:
                try:
                    from langfuse.types import TraceContext
                    self._lf_span = self.langfuse.start_span(
                        name="rag-query",
                        input=question,
                        metadata={
                            "use_case": self.use_case,
                            "engine_name": engine_name or self.use_case,
                            "route_reason": route_reason or "unknown",
                            "status": "in_progress",
                            "user_id": session_id,
                            "session_id": session_id,
                        },
                    )
                    self._current_trace = True
                except Exception as e:
                    logger.warning(f"⚠️ Langfuse span 시작 실패: {e}")
                    self._lf_span = None

            recent_intent = self._classify_recent_intent(question)  # 'experiment' | 'report' | 'none'
            recent_query = recent_intent != 'none'  # 하위 호환 (candidate_k 결정용)

            # 1. 특수 질문 타입 처리 (early return)
            if self._is_general_conversation(question):
                result = self._handle_general_conversation(question)
                self._lf_trace_end(output=result.get("answer", ""), status="success")
                return result

            if self._is_conversation_history_query(question):
                result = self._handle_conversation_history_query(question, session_id)
                self._lf_trace_end(output=result.get("answer", ""), status="success")
                return result

            # 2. CRF 메타데이터 질문 (초기 체크)
            if self.use_case == "crf" and self._is_metadata_query(question):
                hospital_code = self._extract_hospital_code_from_question(question)
                result = self._handle_crf_metadata_query(question, hospital_code, chat_history)
                self._lf_trace_end(output=result.get("answer", ""), status="success")
                return result

            # 3. CRF 통계/차트 질문은 바로 처리 (벡터 검색 생략)
            if self.use_case == "crf" and self._is_statistics_query(question):
                hospital_code = self._extract_hospital_code_from_question(question)
                result = self._handle_crf_statistics_query(question, hospital_code)
                self._lf_trace_end(output=result.get("answer", ""), status="success")
                return result

            # 4. 직접 조회 시도 (이슈 번호 또는 CRF record_id)
            direct_results = self._perform_direct_lookup(question)

            # 5. 과거 대화 검색
            relevant_history = []
            if session_id and len(chat_history) >= C.CHAT_HISTORY_CONFIG['search_history_threshold']:
                relevant_history = self.search_conversation_history(
                    session_id, question,
                    top_k=C.CHAT_HISTORY_CONFIG['max_relevant_history'],
                    conversation_id=conversation_id
                )
                if relevant_history:
                    logger.info(f"  📚 관련 과거 대화: {len(relevant_history)}개 발견")

            # 6. Candidate-K 결정 (많이 검색 후 컷오프)
            candidate_k = self._determine_candidate_k(question)

            # 7. 문서 검색 (candidate_k 개수만큼)
            documents, metadatas, distances = self._search_documents(
                question, chat_history, direct_results, candidate_k
            )

            if not documents:
                self._lf_trace_end(output="관련 정보를 찾을 수 없습니다.", status="not_found")
                return {
                    "answer": "관련 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "question": question
                }

            # 8. 유사도 기반 컷오프 적용 (관련 문서만 남김)
            documents, metadatas, distances = self._apply_similarity_cutoff(
                documents, metadatas, distances
            )

            # 9. 문서 후처리 (버전 재정렬, 최신순 정렬, 키워드 보강)
            documents, metadatas, distances = self._post_process_documents(
                documents, metadatas, distances, question, recent_intent
            )

            # 10. 컨텍스트 구성 및 답변 생성
            result = self._generate_answer(
                question, documents, metadatas, distances,
                chat_history, relevant_history
            )

            self._lf_trace_end(output=result.get("answer", ""), status="success")
            return result

        except Exception as e:
            logger.error(f"❌ 쿼리 처리 중 오류: {str(e)}")
            self._lf_trace_end(status="error", error_message=str(e))
            raise

    def _tokenize_for_bm25(self, text: str) -> list:
        """BM25용 토크나이징 (한글+영문+숫자+버전 토큰 보존)"""
        text_lower = text.lower()
        tokens = re.findall(r'[a-z0-9][a-z0-9._\-]*[a-z0-9]|[a-z0-9]+|[\uac00-\ud7af]+', text_lower)
        return tokens

    def _build_bm25_index(self):
        """ChromaDB 전체 문서를 로드하여 BM25 인덱스 구축"""
        try:
            logger.info("📚 BM25 인덱스 구축 시작...")
            all_data = self.collection.get(include=["documents", "metadatas"])
            docs = all_data.get("documents", [])
            ids = all_data.get("ids", [])
            metadatas = all_data.get("metadatas", [])

            if not docs:
                logger.warning("  ⚠️ BM25 인덱스 구축 실패: 문서 없음")
                return

            # 토크나이징
            tokenized_corpus = []
            for doc, meta in zip(docs, metadatas):
                subject = meta.get("subject", "") if meta else ""
                combined = f"{subject} {doc}"
                tokenized_corpus.append(self._tokenize_for_bm25(combined))

            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.bm25_doc_ids = list(ids)
            self.bm25_corpus = list(docs)
            self.bm25_metadatas = list(metadatas)

            logger.info(f"  ✅ BM25 인덱스 구축 완료: {len(docs)}개 문서")
        except Exception as e:
            logger.error(f"  ❌ BM25 인덱스 구축 실패: {e}")
            self.bm25_index = None

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
