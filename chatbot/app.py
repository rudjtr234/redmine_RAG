"""
Redmine RAG 챗봇 웹 서비스
- Vector DB 읽기
- Gemini API 연동
- 웹 인터페이스 제공
- Multi-turn 대화 지원
"""
from flask import Flask, render_template, request, jsonify, session
import os
import logging
from rag_engine import RedmineRAG
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 세션 설정
app.secret_key = os.environ.get("SECRET_KEY", "redmine-rag-secret-key-change-in-production")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)  # 24시간 유지

# 환경변수
VECTORDB_PATH = os.environ.get("VECTORDB_PATH", "/vectordb/chroma_db_v0.1.2")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "redmine_issues_raw_v2")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
REDMINE_URL = os.environ.get("REDMINE_URL", "https://redmine.192.168.20.150.nip.io:30443")

# RAG 엔진 초기화
logger.info("="*80)
logger.info("🚀 Redmine RAG 챗봇 초기화 중...")
logger.info("="*80)

rag_engine = RedmineRAG(
    vectordb_path=VECTORDB_PATH,
    collection_name=COLLECTION_NAME,
    gemini_api_key=GEMINI_API_KEY,
    redmine_url=REDMINE_URL
)

logger.info("✅ RAG 엔진 준비 완료!")

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    """질의응답 API (Multi-turn 지원)"""
    try:
        data = request.json
        question = data.get('question', '')
        top_k = data.get('top_k', None)

        if not question:
            return jsonify({"error": "질문이 없습니다"}), 400

        # 세션에서 대화 히스토리 가져오기 (최대 10턴)
        if 'chat_history' not in session:
            session['chat_history'] = []

        chat_history = session['chat_history']

        logger.info(f"📝 질문: {question} (히스토리: {len(chat_history)}턴)")

        # RAG 실행 (히스토리 포함 - 최근 5턴만 프롬프트에 사용)
        result = rag_engine.query(question, top_k=top_k, chat_history=chat_history)

        # 대화 히스토리 업데이트 (최대 10턴만 유지)
        chat_history.append({
            "question": question,
            "answer": result['answer']
        })

        if len(chat_history) > 10:
            chat_history = chat_history[-10:]  # 최근 10턴만 유지

        session['chat_history'] = chat_history
        session.modified = True

        logger.info(f"✅ 답변 생성 완료 (히스토리: {len(chat_history)}턴 저장)")

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ 오류: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """대화 히스토리 초기화"""
    session.pop('chat_history', None)
    logger.info("🔄 대화 히스토리 초기화")
    return jsonify({"message": "대화 히스토리가 초기화되었습니다"})

@app.route('/health')
def health():
    """헬스체크"""
    return jsonify({
        "status": "healthy",
        "vectordb_count": rag_engine.get_document_count()
    })

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🌐 웹 서버 시작")
    logger.info("📍 http://0.0.0.0:5000")
    logger.info("="*80)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
