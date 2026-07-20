# chatbot — 이론 PDF RAG 파이프라인

챗봇 모델을 설계할 때 이론을 전부 암기하지 않고, 색인된 PDF에서 검색해 근거로 참조하기 위한 RAG 인덱싱 파이프라인.
"AI 챗봇 구축 방법론" PDF(약 3,000장, 30개 파일 예상)를 대상으로 하며, 챗봇 자체는 파인튜닝 없이 **LLM API 호출 + RAG** 방식으로 만든다.

## 파이프라인

| 단계 | 내용 |
|---|---|
| 1. 원본 PDF 확보 | 로컬 폴더(`PDF_DIR`) 또는 Hugging Face Dataset(`HF_REPO_ID`)에서 동기화 (하위 폴더의 PDF까지 재귀 탐색) |
| 2. 텍스트 추출 | pypdfium2로 텍스트 레이어 우선 추출. 텍스트가 부족한 페이지(스캔 이미지)는 전체를 `gpt-5-mini` 비전 API로 대체 추출하고, 텍스트는 충분해도 페이지 면적의 일정 비율 이상을 차지하는 삽입 이미지(차트/다이어그램)가 있으면 그림 설명만 비전 API로 보강해 덧붙임 |
| 3. 청킹 | `SentenceSplitter` (`CHUNK_SIZE=700`, `CHUNK_OVERLAP=70`) |
| 4. 임베딩 | `BAAI/bge-m3` (한국어/다국어 지원, 로컬 실행) |
| 5. 후보 검색 | BM25(kiwipiepy) + 벡터(BGE-M3, bi-encoder) 검색을 RRF(Reciprocal Rank Fusion)로 융합해 상위 `CANDIDATE_TOP_K`(기본 30)개 추출 |
| 6. 재정렬 | `BAAI/bge-reranker-v2-m3`(cross-encoder)로 후보를 재정렬해 최종 `RERANK_TOP_K`(기본 10)개만 반환 |
| 7. 저장·재개 | 페이지 단위 스트리밍 저장 + `checkpoint.json` 기반 중단/재개. **PDF 1개 완료 시마다 HF Dataset `embedding/`에 백업**하고, 시작 시 로컬 진행 기록이 없으면 그 백업을 복원해 이어서 처리 — 휘발성 컨테이너에서도 마운트 볼륨 없이 재개 가능 |
| 8. 챗봇 응답 생성 | `consultant_bot/` — 위 검색 파이프라인을 도구로 쓰는 6-에이전트 LangGraph 멀티에이전트 (자세히: [consultant_bot/README.md](consultant_bot/README.md)) |

## 설치

```bash
pip install -r requirements.txt
```

`.env.example`을 `.env`로 복사한 뒤 값을 채운다.

```bash
cp .env.example .env
```

| 변수 | 설명 |
|---|---|
| `OPENAI_API_KEY` | `gpt-5-mini` 비전 API 호출용 |
| `OPENAI_BASE_URL` | 프록시/게이트웨이 사용 시에만 설정 (표준 OpenAI 엔드포인트면 비움) |
| `HF_TOKEN` | Hugging Face Hub 토큰 (private dataset repo 사용 시 필수). PDF 동기화·임베딩 인덱스 복원 등 **읽기**에 사용 |
| `HF_REPO_ID` | PDF/임베딩 원본이 있는 HF Dataset repo id (읽기) |
| `HF_TOKEN_ORG` | (선택) `HF_TOKEN`이 읽기 전용이라 업로드가 막힐 때, 쓰기 권한이 있는 별도 토큰 |
| `HF_REPO_ID_ORG` | (선택) 임베딩 인덱스 백업/consultant_bot 체크포인트 백업 등 **쓰기**가 실제로 올라가는 HF Dataset repo id. 없으면 업로드 단계를 건너뜀 |

PDF 경로, 청킹 파라미터 등 나머지 설정은 [config.py](config.py)에서 관리한다.

## 사용법

### 1. 인덱스 생성 (`main.py`)

```bash
python main.py                 # PDF_DIR 전체 처리
python main.py --limit 1       # 시험 삼아 PDF 1개만 처리
python main.py --use-hf        # PDF_DIR 대신 HF_REPO_ID에서 PDF를 동기화해 처리
```

중간에 중단돼도 `checkpoint.json`을 보고 이어서 처리한다 (완료된 PDF/페이지는 건너뜀).

### 2. HF PDF 1개 격리 테스트 (`test_hf_one_pdf.py`)

실제 `output/`과 HF repo를 건드리지 않고 HF Dataset의 PDF 1개만 시험 처리한다.

```bash
python test_hf_one_pdf.py            # 결과는 output_test/에만 저장, HF 업로드 생략
python test_hf_one_pdf.py --clean    # 원상복구 (output_test/, hf_pdfs/ 삭제)
```

### 3. 검색 검증 (`verify_embeddings.py`)

```bash
python verify_embeddings.py "질문 내용"                          # 하이브리드(BM25+벡터 RRF) top 30 -> 리랭크 top 10
python verify_embeddings.py "질문 내용" --mode vector             # 벡터(BGE-M3)만 + 리랭크
python verify_embeddings.py "질문 내용" --mode bm25               # BM25(키워드)만 + 리랭크
python verify_embeddings.py "질문 내용" --candidates 20 --top_k 5 # 후보/최종 개수 직접 조절
python verify_embeddings.py "질문 내용" --no-rerank               # 리랭크 전/후 비교용
```

## 컨테이너 실행 (gcube 배포)

로컬 PC 제약(Windows Smart App Control의 네이티브 DLL 차단) 없이 클라우드 GPU에서 파이프라인을 돌리기 위한 구성.

- `Dockerfile` — `python:3.12-slim` 기반. torch는 **CUDA 12용 빌드(cu128)** 로 설치 (PyPI 기본 torch는 CUDA 13용이라 CUDA 12.x 드라이버 노드에서 동작하지 않음). `EXPOSE 8000`, 출력 경로는 `OUTPUT_DIR=/data/output`.
- `.github/workflows/docker-publish.yml` — main에 push할 때마다 이미지를 빌드해 `ghcr.io/<이 repo 이름>:latest`로 자동 배포 (패키지는 public). 본 repo 기준 `ghcr.io/fisheggavocado/chatbot:latest`.

### gcube 워크로드 설정값

| 항목 | 값 |
|---|---|
| 저장소 유형 | GitHub (인증 체크 불필요 — 이미지 public) |
| 컨테이너 이미지 | `fisheggavocado/chatbot:latest` |
| 컨테이너 포트 | `8000` |
| 컨테이너 명령 (1개 테스트) | `sh -c "python test_hf_one_pdf.py && python -m http.server 8000"` |
| 컨테이너 명령 (본 실행) | 비워둠 → 기본 명령 `python main.py --use-hf` 실행 (완료 시 HF 업로드) |
| 환경변수 | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `HF_TOKEN`, `HF_REPO_ID`, `HF_TOKEN_ORG`, `HF_REPO_ID_ORG`, `VISION_MODEL`, `DEVICE=cuda` |
| 개인 Storage | 선택 사항 — PDF별 HF 백업/복원이 재개를 담당하므로 없어도 됨. 마운트하면 모델 캐시(`/data/hf_cache`)가 유지돼 재시작이 빨라지는 정도 |
| CUDA 버전 | 12.9 (torch cu128 빌드와 호환) |
| 공유 메모리 | 1GB (단일 프로세스 파이프라인이라 크게 필요 없음) |
| GPU | RTX 3070 8GB급이면 충분 (BGE-M3 fp16 ≈ VRAM 2~3GB). 병목은 GPU가 아니라 비전 API 호출이므로 상위 GPU는 불필요. Tier 3 노드는 퇴출이 잦지만 HF 백업/복원 덕에 완료된 PDF 단위로 진행분이 보존됨 |

## 구성 파일

- `config.py` — 경로/모델명/청킹 파라미터 등 전역 설정
- `main.py` — 인덱싱 실행 진입점
- `llama_pdf_reader.py` — 하이브리드 OCR(pypdfium2 + 비전 API) PDF 리더
- `llama_embedding.py` — BGE-M3 임베딩 래퍼
- `llama_bm25_retriever.py` — 한국어 형태소 분석 기반 BM25 리트리버
- `checkpoint.py` — 인덱싱 진행 상황 저장/재개
- `hf_storage.py` — Hugging Face Dataset 동기화/백업
- `reranker.py` — `bge-reranker-v2-m3` 기반 cross-encoder 재정렬
- `verify_embeddings.py` — 저장된 인덱스 검색 검증 스크립트 (후보 검색 + 리랭크)

## 현재 상태

테스트 단계 — 그리드서치로 청킹/검색 파라미터는 확정했으나, 실제 코퍼스(30개 PDF) 대상 본 실행은 아직 진행 전.

- 로컬 PC(Windows 11)는 Smart App Control이 pyarrow 등 네이티브 DLL을 차단해 실행 불가 → **gcube 클라우드 GPU에서 실행하는 방향으로 전환** (2026-07).
- gcube 이미지 검증 통과, Tier 3 RTX 3070 8GB + CUDA 12.9로 워크로드 구성. 격리 테스트 성공 (2026-07-19).
- 본 실행에서 확인된 이슈와 대응 (2026-07-20):
  - HF Dataset 하위 폴더의 PDF 미탐색 → 재귀 탐색으로 수정 (실제 코퍼스: PDF 40개, 2,260페이지. 대부분 텍스트 레이어가 있어 페이지당 0.2~1.2초 수준)
  - 불안정한 노드에서 비전 API 호출이 수십 분 멈춤 → 시도당 120초 타임아웃 + 재시도 5회로 단축, 컨테이너 로그 실시간화(`PYTHONUNBUFFERED`)
  - Tier 3 노드 퇴출 시 진행분 소실(개인 Storage 미등록) → **PDF 1개 완료마다 HF에 백업하고 시작 시 복원하는 구조 도입**, 마운트 볼륨 불필요
- 다음 단계: 본 실행 완주 확인 → HF `embedding/` 결과 검증(`verify_embeddings.py`) → 챗봇 응답 생성 단계.
- 자세한 진행 상황은 `chat_log/project_status.md` 참고 (로컬 전용, `.gitignore`로 제외됨).
