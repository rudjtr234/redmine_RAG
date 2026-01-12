"""정규식 패턴 정의 모음 - 중복 제거 및 중앙 관리"""
import re

# ============================================
# 컴파일된 정규식 패턴 (성능 최적화)
# ============================================

# 이슈 및 레코드 ID 추출
ISSUE_ID_PATTERN = re.compile(r'issue\s*#?(\d+)|이슈\s*#?(\d+)|#(\d+)', re.IGNORECASE)
CRF_RECORD_PATTERN = re.compile(r'(BC_\d+_\d+)', re.IGNORECASE)

# 버전 관련
VERSION_TOKEN_PATTERN = re.compile(r'(v?\d+\.\d+(?:\.\d+)?)', re.IGNORECASE)
VERSION_FILTER_PATTERN = re.compile(r"v?\d+(?:\.\d+)+", re.IGNORECASE)

# 모델 키워드
MODEL_KEYWORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

# ============================================
# 질문 분류 패턴 (문자열 리스트)
# ============================================

# 일반 대화 패턴
GENERAL_CONVERSATION_PATTERNS = [
    r"^(안녕|안녕하세요|하이|헬로|hi|hello|hey)[\?!.\s]*$",
    r"^(감사|고마워|고맙습니다|땡큐|thanks?|thx)[\?!.\s]*$",
    r"(내일|다음주|이번주|미래|계획|뭐하지|할까)",
    r"(날씨|맛집|카페|커피|영화|주식|코인|여행|연애|쇼핑)",
    r"((코드|스크립트|프로그램).*(작성|만들|짜|생성))|((작성|만들|짜|생성).*(코드|스크립트|프로그램))",
    r"^(이거|그거|저거|그것|성능)$",
]

# 대화 이력 조회 패턴
CONVERSATION_HISTORY_PATTERNS = [
    r"과거\s*(대화|질문|내용|이력|히스토리)",
    r"(대화|질문)\s*(목록|리스트|내역|이력)",
    r"전에\s*(물어본|질문한|했던)",
    r"이전\s*(대화|질문)",
    r"history|past\s*conversation",
    r"저장된\s*대화",
    r"대화\s*몇\s*개",
]

# 후속 질문 패턴 (라우팅용)
FOLLOWUP_PATTERNS = [
    r"^(각|전체|모든|모두)\s*(병원|데이터)",
    r"병원\s*별",
    r"비교",
    r"차이",
    r"^(그것|그거|저것|저거|이것|이거)",
    r"^(어떻게|왜|언제|어디)",
    r"(가능|되|할\s*수\s*있)",
    r"^(네|예|응|ㅇㅇ|yes)",
    r"^(더|추가|다른)",
]

# 버전/비교 질문 패턴
VERSION_COMPARISON_PATTERNS = [
    r"\b0\.\d",
    r"\bv\d",
    r"ver",
    r"version",
    r"버전",
    r"전체",
    r"목록",
    r"비교",
    r"차이",
    r"모든",
    r"최신",
    r"이전",
    r"변경",
    r"업데이트",
]

# 기술적 질문 패턴
TECHNICAL_QUERY_PATTERNS = [
    r"pytorch",
    r"tensorflow",
    r"cuda",
    r"framework",
    r"환경",
    r"설정",
    r"config",
    r"파라미터",
    r"하이퍼파라미터",
    r"gpu",
    r"cpu",
    r"메모리",
    r"배치",
    r"epoch",
    r"optimizer",
    r"learning.?rate",
    r"loss",
    r"metric",
    r"데이터셋",
    r"dataset",
    r"모델 구조",
    r"architecture",
]

# Redmine 이슈 질문 패턴
REDMINE_QUERY_PATTERNS = [
    r"\bredmine\b",
    r"이슈",
    r"issue",
    r"\b모델\b",
    r"\bmodel\b",
    r"실험",
    r"experiment",
    r"학습",
    r"train",
    r"training",
    r"테스트",
    r"test",
    r"평가",
    r"evaluation",
    r"결과",
    r"result",
    r"성능",
    r"performance",
    r"정확도",
    r"accuracy",
    r"f1[_\s-]?score",
    r"auc",
    r"auroc",
    r"\bmap\b",
    r"recall",
    r"precision",
]

# CRF 데이터 질문 패턴 (기본)
CRF_BASE_PATTERNS = [
    r"\bcrf\b",
    r"breast",
    r"유방",
    r"임상\s*데이터",
    r"환자",
    r"병원명",
    r"병원코드",
    r"병원번호",
    r"병리번호",
    r"수술연월일",
    r"진단명",
    r"record[_\s-]?id",
    r"bc_\d+_\d+",
    r"통합\s*데이터",
    r"코딩\s*설명",
    r"항목\s*용어",
    r"case\s*no",
    r"path\s*no",
    r"serial\s*no",
]

# CRF 병원 코드 패턴
CRF_HOSPITAL_CODE_PATTERNS = [
    r"병원\s*01",
    r"병원\s*02",
    r"병원\s*03",
    r"병원\s*04",
    r"병원\s*05",
    r"병원\s*06",
    r"병원\s*07",
]

# CRF 의료/임상 마커 패턴
CRF_MEDICAL_PATTERNS = [
    r"her2",
    r"ihc",
    r"fish",
    r"\ber\b",
    r"\bpr\b",
    r"림프절",
    r"전이",
    r"병기",
    r"stage",
    r"grade",
    r"조직학적",
    r"면역조직화학",
    r"수용체",
    r"양성|음성",
    r"판독",
]

# CRF 필드명 패턴
CRF_FIELD_PATTERNS = [
    r"follow[_\s-]?up",
    r"\bf[\s/.-]?u\b",
    r"followup",
    r"\bajcc\b",
    r"\bng\b",
    r"\bhg\b",
    r"\bdcis\b",
    r"\blcis\b",
    r"allred\s*score",
    r"\bki[-\s]?67\b",
    r"\bsish\b",
    r"neoadjuvant",
    r"adjuvant",
    r"\bctx\b",
    r"\brtx\b",
    r"\bbrca\b",
    r"mutation",
    r"\brcb\b",
    r"재발",
    r"recurrence",
    r"axillary",
    r"겨드랑이",
    r"절제연",
    r"margin",
    r"침윤",
    r"invasion",
]

# CRF 메타데이터 질문 패턴
METADATA_QUERY_PATTERNS = [
    r"어떤\s*병원",
    r"병원별\s*데이터\s*수",
    r"(가장\s*)?(오래된|최신|최근).*수술.*날짜",
    r"어떤\s*항목",
    r"데이터\s*수집\s*기간",
    r"필드|컬럼|column|field",
]

# CRF 통계 질문 패턴
STATISTICS_QUERY_PATTERNS = [
    r"통계",
    r"몇\s*(명|건|개)",
    r"총\s*(환자|데이터|수)",
    r"평균",
    r"분포",
    r"비율",
    r"현황",
    r"요약",
    r"전체",
    r"모든",
    r"all",
    r"summary",
    r"count",
    r"수집한\s*데이터",
    r"데이터\s*현황",
    # 바이오마커 기반 통계
    r"(er|pr|her2).*(양성|음성)",
    r"ki[-\s]?67.*\d+%?",
    r"폐경\s*(전|후).*(호르몬|er|pr|양성)",
    # 병원명 패턴
    r"(세브란스|계명대|분당차|강남세브란스|강남차|단국대|이화여대|이대목동)\s*(병원)?\s*(은|는|의)\??$",
    r"(세브란스|계명대|분당차|강남세브란스|강남차|단국대|이화여대|이대목동|병원\s*0[1-7]).*(있어|있나요|있습니까|있는지)",
    r"(데이터|환자|record|기록|자료).*(있어|있나요|있습니까|있는지)",
    r"(있어|있나요|있습니까|있는지)",
]

# CRF 샘플(소수 사례) 질문 패턴
SAMPLE_QUERY_PATTERNS = [
    r"사례",
    r"케이스",
    r"case",
    r"\d+\s*개\s*만",
    r"\d+\s*건\s*만",
    r"몇\s*개\s*만",
    r"몇\s*건\s*만",
    r"상위\s*\d+",
]

# CRF 수치 조건 질문 패턴 (통계/차트 분기 제외용)
CONDITIONAL_QUERY_PATTERNS = [
    r"\d+\s*%?\s*(이상|초과|보다\s*큰|이하|미만)",
    r"[<>]=?\s*\d",
]

# 최신 질문 패턴
RECENT_QUERY_PATTERN = re.compile(r"최신|최근|가장\s*새로운|latest|recent", re.IGNORECASE)
