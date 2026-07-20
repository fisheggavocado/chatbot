# consultant_bot — AI 설계 컨설턴트 봇 (Multi-Agent)

`../chat_log/AI 설계 컨설턴트 봇 — Multi Agent 구성안.py`에 정리된 6-에이전트 LangGraph 설계를 실제 코드로
옮긴 대화형 멀티에이전트. 상위 `first_project`가 만든 RAG 인덱스(PDF → gpt-5-mini 비전 추출 → BGE-M3 임베딩 →
BM25+벡터 RRF+rerank)를 `hybrid_search` 도구로 그대로 재사용해, 그 위에 Coordinator → FAQ Agent / Wizard
Supervisor(3턴, human-in-the-loop) → Research Worker(bounded ReAct) → Presenter → Guardrail 흐름을 얹었다.

아키텍처/라우팅/가드레일 4지점/observability/캐시 구조의 자세한 설명은 **[DESIGN.md](DESIGN.md)** 참고.

## 설치 및 실행

```bash
# first_project/.venv 활성화 상태에서
pip install -r consultant_bot/requirements.txt
python consultant_bot/app.py
```

`DEVICE`는 `config.py`가 `torch.cuda.is_available()`로 자동 감지한다 — 로컬(GPU 없음)에서는 CPU, gcube 등
CUDA가 있는 GPU 클라우드에서는 자동으로 GPU를 쓴다. `OUTPUT_DIR`에 로컬 인덱스가 없으면 `.env`의 `HF_REPO_ID`
에서 `embedding/` 백업을 자동 복원한다(백업도 없으면 먼저 상위 폴더에서 `python main.py`로 인덱스를 만들어야 함).

## 구성 파일

| 파일 | 역할 |
|---|---|
| `state.py` | `WizardState` + 노드 간 구조화 출력 스키마 |
| `llm.py` | 공용 `ChatOpenAI(gpt-5-mini)` 팩토리 (reasoning 모델 temperature 이슈 처리 포함) |
| `retrieval.py` | `hybrid_search` — 벡터+BM25 RRF 융합 후 rerank, HF Dataset 자동 복원 |
| `react_loop.py` | FAQ/Research가 공유하는 bounded-ReAct 엔진 (예산/중복/정체 3중 정지) |
| `cache.py` | `SearchCache`(검색 결과 캐시) / `FaqCache`(FAQ 시맨틱 캐시) |
| `chat_utils.py` | 메시지 목록에서 사용자 발화만 정확히 찾는 공유 헬퍼 |
| `observability.py` | 콜백 핸들러 + `OUTPUT_DIR/traces/{thread_id}.jsonl` 트레이스 로그 |
| `coordinator.py` / `faq_agent.py` / `wizard_supervisor.py` / `research_worker.py` / `presenter.py` / `guardrail.py` | 6개 그래프 노드 |
| `graph.py` | `StateGraph` 조립 + `SqliteSaver` checkpointer |
| `app.py` | 터미널 대화형 실행 진입점 |

## 현재 상태 — 진행 로그

- **2026-07-20**: 설계안 기반 6-에이전트 멀티에이전트 초안 구현 — `WizardState`, Coordinator/FAQ Agent/Wizard
  Supervisor/Research Worker/Presenter/Guardrail 6개 노드, `graph.py`(StateGraph + `SqliteSaver`
  checkpointer, `recursion_limit=8`), `app.py`(interrupt()/Command(resume=...) 기반 터미널 인터페이스) 작성.
- **2026-07-20**: "ReAct는 한 번 잘못 생각하면 계속 틀린다"는 문제와 observability·가드레일 커버리지를
  점검해달라는 요청에 따라 `react_loop.py`(공유 bounded-ReAct — 예산/중복 액션/정체(no-progress) 3중 정지,
  코드가 LLM 판단보다 우선 적용)와 `observability.py`(콜백 + JSONL 트레이스)를 추가하고, 설계안의
  "guardrail 4지점"을 코드 위치에 명시적으로 매핑(입력 가드=Coordinator, 추론 루프 가드=react_loop,
  출력 스키마 가드=Presenter, 출력 내용 가드=Guardrail). `DESIGN.md` 작성.
- **2026-07-20**: `DEVICE`를 `torch.cuda.is_available()`로 자동 감지하도록 `config.py` 수정 — 로컬(GPU 없음)은
  자동 CPU, gcube 같은 CUDA GPU 클라우드에서는 자동 GPU. 정확도 튜닝용 하드코딩 값들(`TEMPERATURE`,
  `NO_PROGRESS_LIMIT`, `NUM_QUERY_EXPANSIONS` 등)에 이름을 붙여 조정 가능하게 정리.
- **2026-07-20**: 데이터 유실 방지·토큰 절약을 위해 `cache.py` 추가 — `SearchCache`(hybrid_search 정확 일치
  캐시)와 `FaqCache`(BGE-M3 임베딩 유사도 기반 FAQ 캐시, guardrail 통과 답변만 저장). `react_loop.py`의 스텝별
  로그에 evidence 전체를 포함시켜 그래프 노드가 끝나기 전에 죽어도 트레이스에서 복구 가능하게 함. 여러 노드가
  "내용 있는 마지막 메시지"를 사용자 발화로 오인할 수 있던 부분을 `chat_utils.last_human_text()`로 통일.
- **2026-07-20**: gpt-5-mini(reasoning 계열 모델)가 `temperature`를 지원하지 않아(기본값 1 외에는 400 에러)
  `llm.py`가 실제 호출 시 깨질 수 있던 버그 발견 및 수정 — 모델명에서 reasoning 계열(gpt-5/o1/o3/o4)을 감지해
  `temperature` 대신 `reasoning_effort`를 사용하도록 변경.
- **2026-07-20**: `WizardState` 체크포인트를 임베딩 인덱스와 같은 패턴으로 HF Dataset에 백업하도록
  `hf_storage.py`에 `upload_checkpoint_to_hf`/`restore_checkpoint_from_hf`(경로: `checkpoints/`) 추가하고
  `graph.py`(시작 시 자동 복원, WAL 체크포인트 후 업로드)·`app.py`(매 턴 종료 후 백업 호출)에 연결. 실제
  HF repo로 왕복 테스트한 결과 코드 경로는 정상 동작하나, 현재 `.env`의 `HF_TOKEN`이 `RogersHun/lecture_pdf`에
  대한 쓰기 권한이 없어(`403 Forbidden`) 실제 업로드는 실패함을 확인.
- **2026-07-20**: 위 문제를 일단 읽기(`HF_TOKEN`/`HF_REPO_ID`)와 쓰기(`HF_TOKEN_ORG`/`HF_REPO_ID_ORG`)
  credential 분리로 해결했다가, 이후 `HF_REPO_ID`를 쓰기 권한이 있는 새 repo(`yeardream-toy-project/lecture_pdf`,
  PDF 40개 보유)로 교체하면서 ORG 분리가 불필요해져 되돌림 — `config.py`/`hf_storage.py`/`main.py`를 전부
  단일 `HF_TOKEN`/`HF_REPO_ID`만 쓰도록 원복. 임베딩 인덱스 백업·consultant_bot 체크포인트 백업·PDF 동기화가
  모두 이 하나의 repo로 통일됨.

## 알려진 한계 / 다음 단계

- **HF 업로드는 `HF_REPO_ID`에 쓰기 권한이 있는 토큰을 전제로 함**: 이전에 읽기 전용 repo(`RogersHun/lecture_pdf`)
  로 업로드 시도 시 `403 Forbidden`이 났던 적이 있음. 현재 repo(`yeardream-toy-project/lecture_pdf`)는 쓰기
  권한이 있는 것으로 확인됐으나, 만약 `HF_REPO_ID`를 다시 읽기 전용 repo로 바꾸면 `upload_output_to_hf`/
  `upload_checkpoint_to_hf`가 다시 403으로 실패할 수 있음(예외는 삼켜서 대화/처리는 안 막힘).
- **시스템 프롬프트(페르소나) 없음**: 각 노드가 개별 목적의 프롬프트만 사용하고, 봇 전체의 톤/역할을 정하는
  시스템 메시지는 없다 (범위 제한은 Coordinator의 규칙/LLM 분류가 대신 담당).
- 캐시(`cache.py`)에 무효화 로직 없음, E2E/LLM-as-Judge 평가 하네스·외부 트레이싱 대시보드 미구현.
