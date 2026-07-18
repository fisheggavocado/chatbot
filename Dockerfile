# PDF -> 비전 텍스트 추출 -> BGE-M3 임베딩 파이프라인 컨테이너
# gcube 등 GitHub 저장소 기반 빌드 환경에서 사용한다.

FROM python:3.12-slim

WORKDIR /app

# 의존성 레이어를 먼저 캐시한다 (코드만 바뀌면 재설치 생략)
# torch는 CUDA 12용 빌드(cu128)로 먼저 설치한다 — PyPI 기본 torch는 CUDA 13용이라
# CUDA 12.x 드라이버 노드(gcube 최대 12.9)에서 동작하지 않는다.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 컨테이너 기본 출력 경로. 개인 Storage를 /data에 마운트하면 인덱스/checkpoint가 영속된다.
ENV OUTPUT_DIR=/data/output \
    HF_HOME=/data/hf_cache

# gcube 등 배포 플랫폼이 이미지 메타데이터에서 서비스 포트를 읽을 수 있도록 선언한다.
EXPOSE 8000

# 기본 명령: HF Dataset의 PDF 전체를 처리 후 인덱스를 HF에 업로드.
# (워크로드 설정의 "컨테이너 명령"란에 값을 넣으면 이 기본값 대신 그 명령이 실행된다.)
CMD ["python", "main.py", "--use-hf"]
