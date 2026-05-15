"""
Gunicorn 프로덕션 설정
"""
import multiprocessing
import os

# 서버 소켓
port = int(os.environ.get("PORT", 8080))
bind = f"0.0.0.0:{port}"
backlog = 2048

# 워커 프로세스
# gthread: SSE 장기 연결을 스레드로 처리 (sync worker는 SSE 연결 시 timeout 발생)
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "gthread"
threads = 4          # 워커당 스레드 수 (SSE + 일반 요청 동시 처리)
worker_connections = 1000
timeout = 300        # 워커 초기화 타임아웃 (모델 로드)
keepalive = 5

# Preload 모드 비활성화 (모델 로드 시간 때문에 timeout 발생)
# preload_app = True

# 로깅
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# 프로세스 이름
proc_name = "redmine_rag_chatbot"

# 재시작
max_requests = 1000
max_requests_jitter = 50

# 보안
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190