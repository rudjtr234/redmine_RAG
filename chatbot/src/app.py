"""Redmine RAG 챗봇 웹 서비스"""


"""
- vector db  읽기
- gemini api 연동
- 웹 인터페이스 연동
- multi-turn 대화 지원
- 세션 별 로그인 추가

"""

from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_session import Session
import os
import sys
import logging
import re
import threading
import time
import uuid
import json as _json
import base64 as _base64
import requests as _requests
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_engine import RedmineRAG
from utils.rag_utils import RAGHelperMixin
from utils.paperbanana_client import generate_diagram
from config import constants as C
from config import patterns as P
from config.diagram_config import (
    PAPERBANANA_BASE_URL,
    PAPERBANANA_TIMEOUT_SECONDS,
    PAPERBANANA_POLL_INTERVAL,
    PAPERBANANA_REFINEMENT_ITERATIONS,
    PAPERBANANA_DEFAULT_DIAGRAM_TYPE,
    PAPERBANANA_MAX_CONCURRENT,
)

# 도식화 동시 요청 제한 (Semaphore)
_visualize_semaphore = threading.Semaphore(PAPERBANANA_MAX_CONCURRENT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.environ.get("SECRET_KEY", "redmine-rag-secret-key-change-in-production")
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.environ.get("SESSION_FILE_DIR", "/vectordb/flask_sessions")
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_USE_SIGNER'] = True
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)
Session(app)

logger.info("🚀 Redmine RAG 챗봇 초기화 중...")
_REDMINE_API_KEY = os.environ.get("REDMINE_API_KEY", "")
_REDMINE_URL = os.environ.get("REDMINE_URL", "https://your-redmine.example.com")
conversation_db_path = os.environ.get("CONVERSATION_DB_PATH")
if not conversation_db_path:
    default_vectordb = os.environ.get("VECTORDB_PATH", "/vectordb/chroma_db_v0.1.2")
    conversation_db_path = os.path.join(os.path.dirname(default_vectordb), "conversation_db")

rag_engine = RedmineRAG(
    vectordb_path=os.environ.get("VECTORDB_PATH", "/vectordb/chroma_db_v0.2.0"),
    collection_name=os.environ.get("COLLECTION_NAME", "redmine_issues_raw_v4"),
    gemini_api_key=os.environ.get("GEMINI_API_KEY"),
    redmine_url=os.environ.get("REDMINE_URL", "https://your-redmine.example.com"),
    use_case=os.environ.get("USE_CASE", "redmine"),
    conversation_db_path=conversation_db_path
)
crf_engine = None
crf_db_path = os.environ.get(
    "CRF_VECTORDB_PATH",
    "/vectordb/crf_data/chroma_db_v0.3.0"
)
crf_collection_name = os.environ.get("CRF_COLLECTION_NAME", "crf_all_cancers_v0.3.0")
try:
    if os.path.exists(crf_db_path):
        crf_engine = RedmineRAG(
            vectordb_path=crf_db_path,
            collection_name=os.environ.get("COLLECTION_NAME", "redmine_issues_raw_v4"),  # 사용되지 않음 (use_case=crf이므로)
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            redmine_url=os.environ.get("REDMINE_URL", "https://your-redmine.example.com"),
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
        route_reason = None
        if is_redmine_query:
            # Redmine 이슈 번호나 명시적 키워드가 있으면 Redmine 확정
            engine = rag_engine
            engine_name = 'redmine'
            route_reason = 'explicit_keyword'
            logger.info("🧭 라우팅: Redmine DB (명시적 키워드)")
        elif is_crf_query and not is_redmine_query:
            # CRF 키워드만 있고 Redmine 키워드 없으면 CRF 확정
            engine = crf_engine
            engine_name = 'crf'
            route_reason = 'explicit_keyword'
            logger.info("🧭 라우팅: CRF DB (명시적 키워드)")
        elif last_engine and is_followup and crf_engine is not None:
            # 이전 엔진이 있고 후속 질문이면 이전 엔진 우선 사용
            if last_engine == 'crf':
                engine = crf_engine
                engine_name = 'crf'
                route_reason = 'followup_context'
                logger.info("🧭 라우팅: CRF DB (이전 맥락 유지 - 후속 질문)")
            else:
                engine = rag_engine
                engine_name = 'redmine'
                route_reason = 'followup_context'
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
                route_reason = 'vector_similarity'
                logger.info(f"  ✅ CRF DB 선택 (거리 차이: {redmine_distance - crf_distance:.4f})")
            elif redmine_distance < crf_distance - threshold:
                engine = rag_engine
                engine_name = 'redmine'
                route_reason = 'vector_similarity'
                logger.info(f"  ✅ Redmine DB 선택 (거리 차이: {crf_distance - redmine_distance:.4f})")
            else:
                # 거리가 비슷하면 이전 엔진 우선, 없으면 Redmine 기본
                if last_engine == 'crf':
                    engine = crf_engine
                    engine_name = 'crf'
                    route_reason = 'similarity_tie_context'
                    logger.info(f"  ⚖️ 유사도 비슷함 → CRF DB (이전 맥락 유지)")
                else:
                    engine = rag_engine
                    engine_name = 'redmine'
                    route_reason = 'similarity_tie_default'
                    logger.info(f"  ⚖️ 유사도 비슷함 → Redmine DB (default)")
        else:
            engine = rag_engine
            engine_name = 'redmine'
            route_reason = 'no_crf_engine'
            logger.info("🧭 라우팅: Redmine DB (CRF 엔진 없음)")

        result = engine.query(
            question,
            chat_history=chat_history,
            session_id=session_id,
            engine_name=engine_name,
            route_reason=route_reason
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

@app.route('/visualize', methods=['POST'])
def visualize():
    """
    1단계: LLM 재작성만 수행 → rewritten 데이터 반환
    프론트가 취소 여부 확인 후 /pb-start 호출

    Request JSON:
        {"question": str, "rag_answer": str, "user_name": str, "mode": str}
        mode: "patent" → 특허 청구항 계층 구조 재작성, 생략/기타 → 일반 도식화
    Response JSON:
        {"source_context": str, "communicative_intent": str}  or  {"error": str}
    """
    request_id = str(uuid.uuid4())[:8]
    t_start = time.time()

    try:
        data = request.json or {}
        question = data.get("question", "").strip()
        rag_answer = data.get("rag_answer", "").strip()
        user_name = data.get("user_name", "unknown")
        mode = data.get("mode", "default")

        if not question or not rag_answer:
            return jsonify({"error": "question과 rag_answer가 필요합니다"}), 400
        if len(question) > 500:
            return jsonify({"error": "question이 너무 깁니다 (최대 500자)"}), 400
        if len(rag_answer) > 5000:
            return jsonify({"error": "rag_answer가 너무 깁니다 (최대 5000자)"}), 400

        logger.info(f"[{request_id}] /visualize 요청: user={user_name}, question={question[:60]}...")

        acquired = _visualize_semaphore.acquire(blocking=False)
        if not acquired:
            logger.warning(f"[{request_id}] 도식화 동시 요청 초과")
            return jsonify({"error": "도식화 요청이 많습니다. 잠시 후 다시 시도하세요."}), 429

        try:
            t1 = time.time()
            rewritten = rag_engine._rewrite_for_diagram(question, rag_answer, mode=mode)
            if not rewritten:
                return jsonify({"error": "도식화 입력 변환에 실패했습니다"}), 500

            source_context = rewritten.get("source_context", "")
            communicative_intent = rewritten.get("communicative_intent", "")
            logger.info(f"[{request_id}] LLM 재작성 완료 ({time.time() - t1:.1f}s)")
        finally:
            _visualize_semaphore.release()

        elapsed = round(time.time() - t_start, 1)
        logger.info(f"[{request_id}] 재작성 완료 ({elapsed}s)")
        return jsonify({
            "source_context": source_context,
            "communicative_intent": communicative_intent,
            "request_id": request_id,
        })

    except Exception as e:
        logger.error(f"[{request_id}] /visualize 예외: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/pb-start', methods=['POST'])
def pb_start():
    """
    2단계: paperbanana task 시작 → task_id 반환
    프론트가 취소하지 않은 경우에만 호출

    Request JSON:
        {"source_context": str, "communicative_intent": str}
    Response JSON:
        {"task_id": str}  or  {"error": str}
    """
    request_id = str(uuid.uuid4())[:8]
    try:
        data = request.json or {}
        source_context = data.get("source_context", "")
        communicative_intent = data.get("communicative_intent", "")

        if not source_context or not communicative_intent:
            return jsonify({"error": "source_context와 communicative_intent가 필요합니다"}), 400

        payload = {
            "source_context": source_context,
            "communicative_intent": communicative_intent,
            "diagram_type": PAPERBANANA_DEFAULT_DIAGRAM_TYPE,
            "refinement_iterations": PAPERBANANA_REFINEMENT_ITERATIONS,
        }
        resp = _requests.post(
            f"{PAPERBANANA_BASE_URL}/api/generate",
            json=payload,
            timeout=15
        )
        resp.raise_for_status()
        task_id = resp.json().get("task_id")
        if not task_id:
            return jsonify({"error": "paperbanana task_id 없음"}), 502

        logger.info(f"[{request_id}] paperbanana task 시작: task_id={task_id}")
        return jsonify({"task_id": task_id})

    except _requests.exceptions.RequestException as e:
        return _pb_request_error(f"[{request_id}] paperbanana", e)
    except Exception as e:
        logger.error(f"[{request_id}] /pb-start 예외: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/pb-stream/<task_id>')
def pb_stream(task_id):
    """
    paperbanana SSE를 프록시하여 프론트에 실시간 진행 상태 전달
    완료 이벤트에 elapsed_seconds 필드 추가
    """
    t_start = time.time()

    def generate():
        stream_url = f"{PAPERBANANA_BASE_URL}/api/stream/{task_id}"
        try:
            with _requests.get(stream_url, stream=True, timeout=(10, None)) as sse_resp:
                sse_resp.raise_for_status()
                for raw_line in sse_resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        yield "\n"
                        continue
                    if raw_line.startswith(":"):
                        yield f"{raw_line}\n\n"
                        continue
                    if raw_line.startswith("data:"):
                        data_str = raw_line[5:].strip()
                        try:
                            event = _json.loads(data_str)
                        except _json.JSONDecodeError:
                            yield f"{raw_line}\n\n"
                            continue

                        # 완료 이벤트에 elapsed_seconds 추가
                        if event.get("done") or event.get("agent") == "Complete":
                            event["elapsed_seconds"] = round(time.time() - t_start, 1)

                        yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"

                        if event.get("done") or event.get("agent") == "Complete":
                            break
        except Exception as e:
            err = _json.dumps({"error": str(e), "done": True})
            yield f"data: {err}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx 버퍼링 비활성화
        }
    )


def _pb_request_error(label: str, e: Exception):
    """pb-* 엔드포인트 공통 RequestException → 502 응답 헬퍼"""
    logger.error(f"{label} 실패: {e}")
    return jsonify({"error": str(e)}), 502


@app.route('/pb-cancel/<task_id>', methods=['DELETE'])
def pb_cancel(task_id):
    """paperbanana 생성 작업 취소 요청을 프록시"""
    try:
        resp = _requests.delete(
            f"{PAPERBANANA_BASE_URL}/api/tasks/{task_id}",
            timeout=10
        )
        return jsonify(resp.json()), resp.status_code
    except _requests.exceptions.RequestException as e:
        return _pb_request_error(f"pb-cancel {task_id}", e)


@app.route('/pb-image/<task_id>')
def pb_image(task_id):
    """
    paperbanana 완성 이미지를 base64로 프록시
    Response JSON: {"image_base64": "data:image/png;base64,..."}
    """
    try:
        img_resp = _requests.get(
            f"{PAPERBANANA_BASE_URL}/api/images/{task_id}",
            timeout=30
        )
        img_resp.raise_for_status()
        b64 = _base64.b64encode(img_resp.content).decode("utf-8")
        return jsonify({"image_base64": f"data:image/png;base64,{b64}"})
    except _requests.exceptions.RequestException as e:
        return _pb_request_error(f"pb-image {task_id}", e)


@app.route('/redmine-image/<int:attachment_id>')
def redmine_image_proxy(attachment_id):
    """Redmine 첨부 이미지를 API 키 없이 브라우저에 제공하는 프록시"""
    if not session.get("user_name"):
        return "", 401
    try:
        url = f"{_REDMINE_URL}/attachments/download/{attachment_id}"
        resp = _requests.get(
            url,
            headers={"X-Redmine-API-Key": _REDMINE_API_KEY},
            timeout=15,
            verify=False,
        )
        if resp.status_code != 200:
            return "", resp.status_code
        content_type = resp.headers.get("Content-Type", "")
        if not content_type.startswith("image/"):
            return "", 403
        return Response(
            stream_with_context(iter([resp.content])),
            content_type=content_type
        )
    except Exception as e:
        logger.warning(f"이미지 프록시 오류 (id={attachment_id}): {e}")
        return "", 502


if __name__ == '__main__':
    logger.info("🌐 웹 서버 시작: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
