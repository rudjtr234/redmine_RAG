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
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RedmineRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.environ.get("SECRET_KEY", "redmine-rag-secret-key-change-in-production")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

logger.info("🚀 Redmine RAG 챗봇 초기화 중...")
rag_engine = RedmineRAG(
    vectordb_path=os.environ.get("VECTORDB_PATH", "/vectordb/chroma_db_v0.1.2"),
    collection_name=os.environ.get("COLLECTION_NAME", "redmine_issues_raw_v2"),
    gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    embedding_model=os.environ.get("EMBEDDING_MODEL", "gemini"),
    redmine_url=os.environ.get("REDMINE_URL", "https://redmine.192.168.20.150.nip.io:30443"),
    use_case=os.environ.get("USE_CASE", "redmine")
)
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

        session_id = f"mts_{user_name}"

        if session.get('user_name') != user_name:
            session.update({
                'session_id': session_id,
                'user_name': user_name,
                'chat_history': [],
                'turn_index': 0
            })
            logger.info(f"🆕 새 세션: {session_id} (사용자: {user_name})")

        chat_history = session.get('chat_history', [])
        turn_index = session.get('turn_index', 0)

        logger.info(f"📝 질문: {question} (히스토리: {len(chat_history)}턴)")

        result = rag_engine.query(
            question,
            top_k=data.get('top_k'),
            chat_history=chat_history,
            session_id=session_id
        )

        rag_engine.save_conversation(
            session_id=session_id,
            turn_index=turn_index,
            question=question,
            answer=result['answer']
        )

        chat_history.append({"question": question, "answer": result['answer']})
        session['chat_history'] = chat_history[-10:]
        session['turn_index'] = turn_index + 1
        session.modified = True

        logger.info(f"✅ 답변 완료 (메모리: {len(chat_history)}턴)")
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

if __name__ == '__main__':
    logger.info("🌐 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
