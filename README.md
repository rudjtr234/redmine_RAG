# Redmine RAG 챗봇

MTS BIO-DT 팀의 Redmine 실험 데이터를 자연어로 검색할 수 있는 RAG 기반 챗봇 시스템

## 📁 디렉토리 구조

```
chatbot/
├── src/
│   ├── app.py                    # Flask 웹 애플리케이션
│   ├── rag_engine.py            # RAG 로직 (ChromaDB + Gemini)
│   └── config/
│       └── gunicorn_config.py   # Gunicorn 서버 설정
├── templates/
│   └── chat.html                # 웹 UI
├── logs/                         # 로그 파일 저장
├── Dockerfile                    # Docker 이미지 빌드 설정
└── requirement.txt              # Python 패키지 의존성
```

## 🚀 실행 방법

### Docker Compose 사용 (권장)

```bash
# 프로젝트 루트에서
cd /data/member/jks/redmine_RAG

# 서비스 시작
sudo docker compose up -d

# 로그 확인
sudo docker compose logs -f chatbot

# 서비스 중지
sudo docker compose down
```

### 로컬 실행

```bash
cd /data/member/jks/redmine_RAG/chatbot

# 의존성 설치
pip install -r requirement.txt

# 환경변수 설정
export VECTORDB_PATH=/vectordb/chroma_db_v0.1.2
export COLLECTION_NAME=redmine_issues_raw_v2
export GEMINI_API_KEY=your-api-key

# 개발 서버 실행
python src/app.py

# 또는 Gunicorn으로 실행
gunicorn --config src/config/gunicorn_config.py src.app:app
```

## 🔧 환경변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `VECTORDB_PATH` | ChromaDB 경로 | `/vectordb/chroma_db_v0.1.2` |
| `COLLECTION_NAME` | 컬렉션 이름 | `redmine_issues_raw_v2` |
| `GEMINI_API_KEY` | Google Gemini API 키 | (필수) |
| `REDMINE_URL` | Redmine 서버 URL | (사내 URL) |
| `PORT` | 서버 포트 | `50001` |
| `GUNICORN_WORKERS` | Gunicorn 워커 수 | `6` |
| `LOG_LEVEL` | 로그 레벨 | `info` |

## 📡 API 엔드포인트

### `GET /`
웹 UI 페이지

### `POST /chat`
질의응답 API (Multi-turn 지원)

**Request:**
```json
{
  "question": "최신 TSR 모델의 Dice Score는?"
}
```

**Response:**
```json
{
  "answer": "TSR v2.1 모델의 Dice Score는 0.8500입니다. (Issue #123)",
  "sources": [
    {
      "issue_id": 123,
      "url": "https://redmine.../issues/123"
    }
  ],
  "question": "최신 TSR 모델의 Dice Score는?"
}
```

### `POST /reset`
대화 히스토리 초기화

### `GET /health`
헬스체크

## 🔄 업데이트 및 재배포

```bash
# 코드 수정 후
cd /data/member/jks/redmine_RAG

# 컨테이너 재빌드 (캐시 없이)
sudo docker compose down
sudo docker compose up -d --build --no-cache

# 로그 확인
sudo docker compose logs -f chatbot
```

## 🎯 주요 기능

- **Multi-turn 대화**: 대화 맥락을 기억하여 후속 질문 가능
- **초심자 가이드**: 클릭 가능한 프롬프트 예시 제공
- **실시간 검색**: Vector DB 기반 의미론적 검색
- **참고 이슈 링크**: 답변 출처를 Redmine 이슈로 연결
- **마크다운 표 지원**: 복잡한 데이터를 표 형식으로 표시

## 📝 버전 정보

- **현재 버전**: v0.2.2
- **Python**: 3.11
- **Flask**: 3.1.0
- **ChromaDB**: 0.5.23
- **Gemini**: 2.0 Pro
- **임베딩 모델**: intfloat/multilingual-e5-large
