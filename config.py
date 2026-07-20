# 프로젝트 전역 설정 파일
# PDF 경로, 모델명, 청킹 파라미터 등을 한 곳에서 관리한다.

import os

from dotenv import load_dotenv

# .env 파일에 저장된 환경변수(OPENAI_API_KEY, HF_TOKEN 등)를 불러온다.
load_dotenv()

# gpt-5-mini 비전 API 호출에 사용할 OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 커스텀 엔드포인트(프록시, 게이트웨이 등)를 통해 API 토큰을 쓰는 경우의 base URL.
# 표준 OpenAI 엔드포인트를 쓰면 비워두면 된다 (None이면 openai 패키지 기본값 사용).
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Hugging Face Hub 접근 토큰 (Dataset repo 업로드/다운로드에 사용, private repo는 필수)
HF_TOKEN = os.getenv("HF_TOKEN")

# TODO: 추후 실제 PDF 폴더 경로로 교체
PDF_DIR = r"D:\누리\이어드림 AI 교육\수업 자료\5강"  # 예: r"D:\docs\pdfs"  (임베딩할 PDF들이 들어있는 폴더)
# 컨테이너/클라우드 환경에서는 환경변수 OUTPUT_DIR로 저장 경로를 덮어쓸 수 있다 (예: /data/output).
OUTPUT_DIR = os.getenv("OUTPUT_DIR", r"D:\누리\이어드림 AI 교육\project\first_project\output")

# Hugging Face Dataset repo id (예: "your-username/pdf-dataset") — .env의 HF_REPO_ID에 한 번만 설정
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_LOCAL_SYNC_DIR = r"./hf_pdfs"  # HF Dataset repo에서 내려받은 PDF를 저장할 로컬 캐시 폴더

# 모델명은 비밀값은 아니라 코드에 둬도 되지만, .env에서 값을 주면 코드 수정 없이 바꿀 수 있다 (안 쓰면 기본값 사용).
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-5-mini")  # PDF 페이지 시각적 해석 (OpenAI API, 게이트웨이 요구 형식)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # 로컬 오픈소스 임베딩 모델

CHUNK_SIZE = 700       # 임베딩 청크 최대 토큰 수
CHUNK_OVERLAP = 70     # 청크 간 중복 토큰 수 (문맥 단절 방지용)
PDF_IMAGE_DPI = 200    # PDF -> 이미지 변환 해상도 (높을수록 선명하지만 처리 비용 증가)

# 이 길이(글자 수) 미만으로 텍스트 레이어가 추출되면 스캔 이미지 페이지로 간주해 비전 API로 처리한다.
MIN_EXTRACTABLE_TEXT_LENGTH = 30

# 삽입 이미지가 페이지 면적 대비 이 비율 이상을 차지해야 "의미 있는 그림"(차트/다이어그램 등)으로 간주한다.
# 로고·아이콘처럼 작은 장식 이미지는 이 값 미만이라 비전 API 추가 호출 대상에서 제외된다.
MIN_IMAGE_AREA_RATIO = 0.05

# BGE-M3/리랭커 실행 장치. DEVICE 환경변수를 명시하면 그 값을 그대로 쓰고,
# 명시하지 않으면 torch.cuda.is_available()로 자동 감지한다 — 로컬(GPU 없음)에서는 cpu,
# GPU 클라우드(CUDA 있음)에서는 자동으로 cuda를 쓰므로 환경마다 따로 설정할 필요가 없다.
_env_device = os.getenv("DEVICE")
if _env_device:
    DEVICE = _env_device
else:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 검색: bi-encoder(BGE-M3)+BM25를 RRF로 융합해 CANDIDATE_TOP_K개 후보를 뽑은 뒤,
# cross-encoder(RERANK_MODEL)로 재정렬해 RERANK_TOP_K개만 최종 반환한다.
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
CANDIDATE_TOP_K = 30  # 리랭킹 전, bi-encoder/BM25 단계에서 뽑을 후보 개수
RERANK_TOP_K = 10     # 크로스 인코더 리랭킹 후 최종적으로 반환할 개수

SECONDS_PER_PAGE_ESTIMATE = 6  # 실제 처리 전 대략적인 예상 시간 계산에 쓰는 가정치(초/페이지)
PERSIST_EVERY_N_PAGES = 1      # 몇 페이지마다 인덱스를 디스크에 저장할지 (1 = 매 페이지 즉시 저장)
