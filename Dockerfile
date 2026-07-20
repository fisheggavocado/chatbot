# PDF -> 비전 텍스트 추출 -> BGE-M3 임베딩 -> consultant_bot(챗봇+FAQ 멀티에이전트) 컨테이너
# gcube 등 GitHub 저장소 기반 빌드 환경에서 사용한다.
# 임베딩은 이미 완료되어 HF_REPO_ID의 embedding/에 백업돼 있으므로, 기본 실행은 재임베딩 없이
# consultant_bot이 그 백업을 복원해 바로 챗봇/FAQ로 동작한다 (재임베딩이 필요할 때만 main.py를 수동 실행).

FROM python:3.12-slim

WORKDIR /app

# 의존성 레이어를 먼저 캐시한다 (코드만 바뀌면 재설치 생략)
# torch는 CUDA 12용 빌드(cu128)로 먼저 설치한다 — PyPI 기본 torch는 CUDA 13용이라
# CUDA 12.x 드라이버 노드(gcube 최대 12.9)에서 동작하지 않는다.
COPY requirements.txt .
COPY consultant_bot/requirements.txt consultant_bot/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r consultant_bot/requirements.txt

COPY . .

# 컨테이너 기본 출력 경로. 개인 Storage를 /data에 마운트하면 인덱스/checkpoint가 영속된다.
ENV OUTPUT_DIR=/data/output \
    HF_HOME=/data/hf_cache \
    PYTHONUNBUFFERED=1

# gcube 등 배포 플랫폼이 이미지 메타데이터에서 서비스 포트를 읽을 수 있도록 선언한다.
EXPOSE 8000

# 기본 명령: HF_REPO_ID의 embedding/ 백업을 복원해 consultant_bot 챗봇(FAQ Agent 포함)을
# HTTP API(server.py, 포트 8000)로 바로 실행. 컨테이너에 대화형 stdin이 붙지 않는 배포 환경이라
# app.py(터미널 input() 루프)가 아니라 FastAPI 서버를 기본값으로 쓴다.
# 재임베딩이 필요하면(PDF 추가/변경 시) 워크로드 설정의 "컨테이너 명령"란에
# `python main.py --use-hf`를 넣어 이 기본값을 덮어쓴다.
CMD ["python", "-m", "uvicorn", "consultant_bot.server:app", "--host", "0.0.0.0", "--port", "8000"]
