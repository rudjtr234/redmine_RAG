"""
Gunicorn 프로덕션 설정
"""
import multiprocessing
import os

# 서버 소켓
port = int(os.environ.get("PORT", 50001))
bind = f"0.0.0.0:{port}"
backlog = 2048

# 워커 프로세스
workers = int(os.environ.get("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
timeout = 300  # 임베딩 모델 로드 시간 고려 (5분)
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