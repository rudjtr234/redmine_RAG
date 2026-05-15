"""
paperbanana_v2 API 클라이언트

BASE_URL 규칙: PAPERBANANA_BASE_URL = http://<host>:<port>  (trailing /api 없이)
이 모듈에서 /api/generate, /api/stream/{task_id}, /api/images/{task_id} 를 붙임

주요 함수:
    generate_diagram(...) -> dict
        {"success": bool, "image_base64": str, "task_id": str, "error": str|None, "error_type": str|None}
"""

import base64
import json
import logging
import time
import uuid

import requests

logger = logging.getLogger(__name__)


def generate_diagram(
    source_context: str,
    communicative_intent: str,
    diagram_type: str = "methodology",
    raw_data: str = "",
    refinement_iterations: int = 1,
    base_url: str = "http://localhost:50003",
    timeout: int = 180,
    poll_interval: int = 2,
    request_id: str = "",
) -> dict:
    """
    paperbanana_v2에 도식화 요청을 보내고 완성된 이미지를 base64로 반환.

    Args:
        source_context: 영문 방법론/알고리즘 설명 (paperbanana Planner 입력)
        communicative_intent: 도식 목적 한 줄 캡션
        diagram_type: "methodology" | "statistical_plot"
        raw_data: statistical_plot 용 JSON 데이터 문자열 (선택)
        refinement_iterations: paperbanana 반복 횟수 (1~5)
        base_url: paperbanana 서비스 base URL (trailing /api 없이)
        timeout: 전체 SSE 대기 타임아웃 (초)
        poll_interval: SSE 재연결 대기 (초)
        request_id: 로깅용 요청 식별자

    Returns:
        {
            "success": bool,
            "image_base64": "data:image/png;base64,...",  # 성공 시
            "task_id": str,
            "error": str | None,
            "error_type": "upstream" | "timeout" | None  # 에러 분류
        }
    """
    rid = request_id or str(uuid.uuid4())[:8]
    t_start = time.time()

    # ─── 1. POST /api/generate ───────────────────────────────────────────────
    payload = {
        "source_context": source_context,
        "communicative_intent": communicative_intent,
        "diagram_type": diagram_type,
        "refinement_iterations": refinement_iterations,
    }
    if raw_data:
        payload["raw_data"] = raw_data

    generate_url = f"{base_url}/api/generate"
    logger.info(f"[{rid}] paperbanana generate 요청: {generate_url}")

    try:
        resp = requests.post(generate_url, json=payload, timeout=30)
        resp.raise_for_status()
        task_data = resp.json()
    except requests.exceptions.ConnectionError as e:
        logger.error(f"[{rid}] paperbanana 연결 실패: {e}")
        return {"success": False, "image_base64": None, "task_id": None,
                "error": f"paperbanana 서버에 연결할 수 없습니다: {e}", "error_type": "upstream"}
    except requests.exceptions.HTTPError as e:
        logger.error(f"[{rid}] paperbanana HTTP 오류: {e}")
        return {"success": False, "image_base64": None, "task_id": None,
                "error": f"paperbanana 서버 오류 ({resp.status_code})", "error_type": "upstream"}
    except requests.exceptions.RequestException as e:
        logger.error(f"[{rid}] paperbanana generate 요청 실패: {e}")
        return {"success": False, "image_base64": None, "task_id": None,
                "error": str(e), "error_type": "upstream"}

    task_id = task_data.get("task_id")
    if not task_id:
        logger.error(f"[{rid}] task_id 없음: {task_data}")
        return {"success": False, "image_base64": None, "task_id": None,
                "error": "paperbanana 응답에 task_id 없음", "error_type": "upstream"}

    t_generate = time.time()
    logger.info(f"[{rid}] task 생성 완료: task_id={task_id} ({t_generate - t_start:.1f}s)")

    # ─── 2. GET /api/stream/{task_id} (SSE) ──────────────────────────────────
    stream_url = f"{base_url}/api/stream/{task_id}"
    deadline = time.time() + timeout
    completed = False

    while time.time() < deadline:
        try:
            # SSE는 연결 유지가 필요하므로 read timeout을 None(무한)으로 설정
            # (connect_timeout=10, read_timeout=None)
            with requests.get(stream_url, stream=True, timeout=(10, None)) as sse_resp:
                sse_resp.raise_for_status()
                for raw_line in sse_resp.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    # keepalive 라인 무시
                    if raw_line.startswith(":"):
                        continue
                    # "data: {...}" 파싱
                    if raw_line.startswith("data:"):
                        data_str = raw_line[5:].strip()
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        agent = event.get("agent", "")
                        progress = event.get("progress", 0)
                        message = event.get("message", "")
                        logger.info(f"[{rid}] SSE [{agent}] {progress}% - {message}")

                        # 완료 감지: done=true 또는 agent=="Complete"
                        if event.get("done") or agent == "Complete":
                            completed = True
                            break
                        # 오류 감지
                        if agent == "Error" or event.get("error"):
                            err_msg = event.get("error") or event.get("message", "Unknown error")
                            logger.error(f"[{rid}] paperbanana 오류 이벤트: {err_msg}")
                            return {"success": False, "image_base64": None, "task_id": task_id,
                                    "error": f"paperbanana 생성 오류: {err_msg}", "error_type": "upstream"}

                        if time.time() > deadline:
                            break

            if completed:
                break

        except requests.exceptions.RequestException as e:
            logger.warning(f"[{rid}] SSE 연결 오류 (재시도 대기): {e}")

        if not completed and time.time() < deadline:
            time.sleep(poll_interval)

    if not completed:
        elapsed = time.time() - t_start
        logger.error(f"[{rid}] paperbanana 타임아웃: {elapsed:.1f}s (한도: {timeout}s)")
        return {"success": False, "image_base64": None, "task_id": task_id,
                "error": f"도식화 생성 타임아웃 ({timeout}초)", "error_type": "timeout"}

    t_sse = time.time()
    logger.info(f"[{rid}] SSE 완료: task_id={task_id} ({t_sse - t_generate:.1f}s)")

    # ─── 3. GET /api/images/{task_id} → PNG → base64 ─────────────────────────
    image_url = f"{base_url}/api/images/{task_id}"
    logger.info(f"[{rid}] 이미지 다운로드: {image_url}")

    try:
        img_resp = requests.get(image_url, timeout=30)
        img_resp.raise_for_status()
        b64 = base64.b64encode(img_resp.content).decode("utf-8")
        image_base64 = f"data:image/png;base64,{b64}"
    except requests.exceptions.RequestException as e:
        logger.error(f"[{rid}] 이미지 다운로드 실패: {e}")
        return {"success": False, "image_base64": None, "task_id": task_id,
                "error": f"이미지 다운로드 실패: {e}", "error_type": "upstream"}

    t_end = time.time()
    logger.info(f"[{rid}] 도식화 완료: task_id={task_id} 총 {t_end - t_start:.1f}s")

    return {
        "success": True,
        "image_base64": image_base64,
        "task_id": task_id,
        "error": None,
        "error_type": None,
    }
