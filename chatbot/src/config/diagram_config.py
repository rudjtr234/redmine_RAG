import os

# paperbanana_v2 API 연동 설정
# BASE_URL 규칙: host:port만 (trailing /api 없이)
# 코드에서 /api/generate, /api/stream/{task_id}, /api/images/{task_id} 를 붙임
# Deployment env 예시:
#   PAPERBANANA_BASE_URL=http://<host>:<port>
PAPERBANANA_BASE_URL = os.environ.get("PAPERBANANA_BASE_URL")
if not PAPERBANANA_BASE_URL:
    raise RuntimeError(
        "PAPERBANANA_BASE_URL environment variable is required. "
        "Set to http://<host>:<port> (no trailing /api). "
        "Example: http://localhost:50003"
    )

PAPERBANANA_TIMEOUT_SECONDS = int(os.environ.get("PAPERBANANA_TIMEOUT_SECONDS", "180"))
PAPERBANANA_POLL_INTERVAL = 2
PAPERBANANA_REFINEMENT_ITERATIONS = 1
PAPERBANANA_DEFAULT_DIAGRAM_TYPE = "methodology"
PAPERBANANA_MAX_CONCURRENT = 6  # threading.Semaphore 동시 요청 한도 (초과 시 429)
