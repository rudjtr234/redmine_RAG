# 멀티 스테이지 빌드
FROM python:3.11-slim as builder

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirement.txt .
RUN pip install --no-cache-dir --user -r requirement.txt


# 프로덕션 이미지
FROM python:3.11-slim

WORKDIR /app

# curl 설치 (헬스체크용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 시스템 사용자 생성 (보안) - 호스트 UID와 동일하게 설정
RUN useradd -m -u 1002 appuser && \
    chown -R appuser:appuser /app

# 빌더에서 설치된 패키지 복사
COPY --from=builder /root/.local /home/appuser/.local

# PATH 설정
ENV PATH=/home/appuser/.local/bin:$PATH

# 애플리케이션 코드 복사
COPY --chown=appuser:appuser ./chatbot .

# 로그 디렉토리 생성 및 권한 설정
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# 비root 사용자로 전환
USER appuser

# 포트 고정
ENV PORT=50001
EXPOSE 50001

# 헬스체크 - curl 사용
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:50001/health || exit 1

# Gunicorn으로 실행
CMD ["gunicorn", "--config", "src/config/gunicorn_config.py", "src.app:app"]
