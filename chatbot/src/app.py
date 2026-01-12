"""Redmine RAG 챗봇 웹 서비스"""


"""
- vector db  읽기
- gemini api 연동
- 웹 인터페이스 연동
- multi-turn 대화 지원
- 세션 별 로그인 추가

"""

from flask import Flask, render_template, request, jsonify, session
import os
import sys
import logging
import re
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RedmineRAG
from utils.rag_utils import RAGHelperMixin
from config import constants as C
from config import patterns as P

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.environ.get("SECRET_KEY", "redmine-rag-secret-key-change-in-production")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

logger.info("🚀 Redmine RAG 챗봇 초기화 중...")
conversation_db_path = os.environ.get("CONVERSATION_DB_PATH")
if not conversation_db_path:
    default_vectordb = os.environ.get("VECTORDB_PATH", "/vectordb/chroma_db_v0.1.2")
    conversation_db_path = os.path.join(os.path.dirname(default_vectordb), "conversation_db")

rag_engine = RedmineRAG(
    vectordb_path=os.environ.get("VECTORDB_PATH", "/vectordb/chroma_db_v0.1.2"),
    collection_name=os.environ.get("COLLECTION_NAME", "redmine_issues_raw_v2"),
    gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    embedding_model=os.environ.get("EMBEDDING_MODEL", "gemini"),
    redmine_url=os.environ.get("REDMINE_URL", "https://redmine.<INTERNAL-IP>.nip.io:30443"),
    use_case=os.environ.get("USE_CASE", "redmine"),
    conversation_db_path=conversation_db_path
)
crf_engine = None
crf_db_path = os.environ.get(
    "CRF_VECTORDB_PATH",
    "/data/member/jks/redmine_RAG/vectordb/crf_data/chroma_db_v0.3.0"
)
crf_collection_name = os.environ.get("CRF_COLLECTION_NAME", "crf_breast_v0.3.0")
try:
    if os.path.exists(crf_db_path):
        crf_engine = RedmineRAG(
            vectordb_path=crf_db_path,
            collection_name="redmine_issues_raw_v2",  # 사용되지 않음 (use_case=crf이므로)
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            embedding_model=os.environ.get("CRF_EMBEDDING_MODEL", "gemini"),
            redmine_url=os.environ.get("REDMINE_URL", "https://redmine.<INTERNAL-IP>.nip.io:30443"),
            use_case="crf",
            conversation_db_path=conversation_db_path,
            crf_collection_name=crf_collection_name  # CRF 컬렉션 명시
        )
        logger.info(f"✅ CRF 엔진 준비 완료! (컬렉션: {crf_collection_name})")
    else:
        logger.warning(f"⚠️ CRF DB 경로 없음: {crf_db_path}")
except Exception as e:
    logger.warning(f"⚠️ CRF 엔진 초기화 실패: {e}")
logger.info("✅ RAG 엔진 준비 완료!")

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question', '')
        user_name = data.get('user_name', '')

        if not question:
            return jsonify({"error": "질문이 없습니다"}), 400
        if not user_name:
            return jsonify({"error": "사용자 이름이 필요합니다"}), 400

        # RAGHelperMixin의 SESSION_ID_PREFIX 사용 (중복 제거)
        session_id = f"{RAGHelperMixin.SESSION_ID_PREFIX}{user_name}"

        if session.get('user_name') != user_name:
            session.update({
                'session_id': session_id,
                'user_name': user_name,
                'chat_history': [],
                'turn_index': 0,
                'last_engine': None  # 이전 턴에서 사용한 엔진 추적
            })
            logger.info(f"🆕 새 세션: {session_id} (사용자: {user_name})")

        chat_history = session.get('chat_history', [])
        turn_index = session.get('turn_index', 0)
        last_engine = session.get('last_engine', None)  # 이전 엔진 가져오기

        logger.info(f"📝 질문: {question} (히스토리: {len(chat_history)}턴, 이전 엔진: {last_engine})")

        # 1단계: 명시적 키워드 체크 (빠른 경로)
        is_crf_query = crf_engine is not None and crf_engine.is_crf_data_query(question)
        is_redmine_query = bool(rag_engine._extract_issue_ids(question)) or rag_engine.is_redmine_data_query(question)

        # 후속 질문 패턴 감지 (애매한 질문)
        is_followup = any(re.search(p, question, re.IGNORECASE) for p in P.FOLLOWUP_PATTERNS)

        # 2단계: 라우팅 결정
        if is_redmine_query:
            # Redmine 이슈 번호나 명시적 키워드가 있으면 Redmine 확정
            engine = rag_engine
            engine_name = 'redmine'
            logger.info("🧭 라우팅: Redmine DB (명시적 키워드)")
        elif is_crf_query and not is_redmine_query:
            # CRF 키워드만 있고 Redmine 키워드 없으면 CRF 확정
            engine = crf_engine
            engine_name = 'crf'
            logger.info("🧭 라우팅: CRF DB (명시적 키워드)")
        elif last_engine and is_followup and crf_engine is not None:
            # 이전 엔진이 있고 후속 질문이면 이전 엔진 우선 사용
            if last_engine == 'crf':
                engine = crf_engine
                engine_name = 'crf'
                logger.info("🧭 라우팅: CRF DB (이전 맥락 유지 - 후속 질문)")
            else:
                engine = rag_engine
                engine_name = 'redmine'
                logger.info("🧭 라우팅: Redmine DB (이전 맥락 유지 - 후속 질문)")
        elif crf_engine is not None:
            # 애매한 경우: 벡터 유사도 비교
            logger.info("🧭 애매한 질문 → 벡터 유사도 비교 시작")

            crf_result = crf_engine.compare_collection_similarity(question)
            redmine_result = rag_engine.compare_collection_similarity(question)

            crf_distance = crf_result['distance']
            redmine_distance = redmine_result['distance']

            logger.info(f"  📊 CRF 거리: {crf_distance:.4f} vs Redmine 거리: {redmine_distance:.4f}")

            # 유사도 차이 임계값 (0.05 = 5% 차이)
            threshold = C.ROUTING_SIMILARITY_THRESHOLD

            if crf_distance < redmine_distance - threshold:
                engine = crf_engine
                engine_name = 'crf'
                logger.info(f"  ✅ CRF DB 선택 (거리 차이: {redmine_distance - crf_distance:.4f})")
            elif redmine_distance < crf_distance - threshold:
                engine = rag_engine
                engine_name = 'redmine'
                logger.info(f"  ✅ Redmine DB 선택 (거리 차이: {crf_distance - redmine_distance:.4f})")
            else:
                # 거리가 비슷하면 이전 엔진 우선, 없으면 Redmine 기본
                if last_engine == 'crf':
                    engine = crf_engine
                    engine_name = 'crf'
                    logger.info(f"  ⚖️ 유사도 비슷함 → CRF DB (이전 맥락 유지)")
                else:
                    engine = rag_engine
                    engine_name = 'redmine'
                    logger.info(f"  ⚖️ 유사도 비슷함 → Redmine DB (default)")
        else:
            engine = rag_engine
            engine_name = 'redmine'
            logger.info("🧭 라우팅: Redmine DB (CRF 엔진 없음)")

        result = engine.query(
            question,
            top_k=data.get('top_k'),
            chat_history=chat_history,
            session_id=session_id
        )

        engine.save_conversation(
            session_id=session_id,
            turn_index=turn_index,
            question=question,
            answer=result['answer']
        )

        chat_history.append({"question": question, "answer": result['answer']})
        session['chat_history'] = chat_history[-C.CHAT_HISTORY_CONFIG["max_turns_in_memory"]:]
        session['turn_index'] = turn_index + 1
        session['last_engine'] = engine_name  # 현재 사용한 엔진 저장
        session.modified = True

        logger.info(f"✅ 답변 완료 (메모리: {len(chat_history)}턴, 사용 엔진: {engine_name})")
        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    session.pop('chat_history', None)
    logger.info("🔄 대화 히스토리 초기화")
    return jsonify({"message": "대화 히스토리가 초기화되었습니다"})

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "vectordb_count": rag_engine.get_document_count()
    })

@app.route('/users', methods=['GET'])
def get_users():
    """사용자 목록 조회 (대화 로그에서 추출)"""
    try:
        users = rag_engine.get_user_list()
        return jsonify({"users": users})
    except Exception as e:
        logger.error(f"❌ 사용자 목록 조회 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/users/<user_name>', methods=['DELETE'])
def delete_user(user_name):
    """사용자 삭제 (대화 로그 삭제)"""
    try:
        success = rag_engine.delete_user(user_name)
        if success:
            return jsonify({"message": f"사용자 '{user_name}' 삭제 완료"})
        else:
            return jsonify({"error": "삭제 실패"}), 500
    except Exception as e:
        logger.error(f"❌ 사용자 삭제 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("🌐 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
