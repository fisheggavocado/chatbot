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
python consultant_bot/app.py                 # 로컬 터미널 대화형 (개발/디버그용)
uvicorn consultant_bot.server:app --reload    # 로컬에서 HTTP API로 띄워볼 때
```

`DEVICE`는 `config.py`가 `torch.cuda.is_available()`로 자동 감지한다 — 로컬(GPU 없음)에서는 CPU, gcube 등
CUDA가 있는 GPU 클라우드에서는 자동으로 GPU를 쓴다. `OUTPUT_DIR`에 로컬 인덱스가 없으면 `.env`의 `HF_REPO_ID`
에서 `embedding/` 백업을 자동 복원한다(백업도 없으면 먼저 상위 폴더에서 `python main.py`로 인덱스를 만들어야 함).

### HTTP API (`server.py`)

gcube처럼 컨테이너에 대화형 stdin이 붙지 않는 배포 환경에서는 `app.py`(터미널 `input()` 루프)가 시작하자마자
`EOFError`로 죽을 수 있어, 요청/응답 기반의 FastAPI 서버를 대신 쓴다. interrupt/resume 3턴 위저드는
**요청 1건 = `graph.invoke()` 1회**로 매핑해, app.py의 while 루프(터미널에서 그 자리에 입력받기)를 두
엔드포인트로 나눴다.

| 엔드포인트 | 설명 |
|---|---|
| `GET /health` | 상태 확인 |
| `POST /chat/message` `{message, thread_id?}` | 새 대화면 `thread_id` 생략(서버가 발급), 이어가는 대화면 이전 응답의 `thread_id`를 그대로 전달 |
| `POST /chat/resume` `{thread_id, resume}` | 직전 응답이 `status: "interrupt"`일 때만 호출. `resume` 형태는 `stage`별로 다름: `tech_select` → `["기술1", "기술2"]`, `pipeline_select` → `{"pipeline": "이름", "confirm": true}`, `compare` → `{"confirm": true}` |

두 엔드포인트 모두 `{status: "interrupt", stage, payload}` 또는 `{status: "done", answer}`를 반환한다.

### gcube 컨테이너 명령

루트 `Dockerfile`의 기본 명령이 위 `server.py`(uvicorn, 포트 8000)로 설정되어 있어, 임베딩이 이미 완료돼
HF `embedding/`에 백업돼 있으면 **재임베딩 없이 바로 이 챗봇(FAQ Agent 포함)이 HTTP로 실행**된다.

| 상황 | gcube 워크로드 "컨테이너 명령"란 |
|---|---|
| 챗봇/FAQ 실행 (기본값) | 비워둠 → 기본 명령 `uvicorn consultant_bot.server:app --host 0.0.0.0 --port 8000` 실행 |
| 재임베딩이 필요할 때만 (PDF 추가/변경 시) | `python main.py --use-hf` 로 덮어써서 실행 |

## 구성 파일

| 파일 | 역할 |
|---|---|
| `state.py` | `WizardState` + 노드 간 구조화 출력 스키마 |
| `llm.py` | 공용 `ChatOpenAI(gpt-5-mini)` 팩토리 (reasoning 모델 temperature 이슈 처리 포함) |
| `retrieval.py` | `hybrid_search` — 벡터+BM25 RRF 융합 후 rerank, HF Dataset 자동 복원 |
| `react_loop.py` | FAQ/Research가 공유하는 bounded-ReAct 엔진 (예산/중복/정체 3중 정지) |
| `cache.py` | `SearchCache`(검색 결과 캐시) / `FaqCache`(FAQ 시맨틱 캐시), 인덱스 지문 기반 자동 무효화 + TTL 만료 |
| `chat_utils.py` | 메시지 목록에서 사용자 발화만 정확히 찾는 공유 헬퍼 |
| `observability.py` | 콜백 핸들러 + `OUTPUT_DIR/traces/{thread_id}.jsonl` 트레이스 로그 + 선택적 LangSmith 연동 |
| `coordinator.py` / `faq_agent.py` / `wizard_supervisor.py` / `research_worker.py` / `presenter.py` / `guardrail.py` | 6개 그래프 노드 |
| `graph.py` | `StateGraph` 조립 + `SqliteSaver` checkpointer |
| `app.py` | 터미널 대화형 실행 진입점 (로컬 개발/디버그용) |
| `server.py` | FastAPI HTTP 실행 진입점 (gcube 등 배포용 기본 진입점) |
| `eval/` | E2E + LLM-as-Judge 평가 하네스 (`eval/run_eval.py`) — 아래 "품질 평가" 참고 |

## 품질 평가 (E2E + LLM-as-Judge)

`../test/`(run_scenarios.py·regression_faq.py)가 "배관이 실제로 동작하는가"를 확인하는 것과 달리,
`eval/run_eval.py`는 `eval/cases.py`의 시나리오(FAQ/out_of_scope/design)를 실제 그래프로 끝까지 실행한 뒤
**두 겹**으로 채점한다: ① 결정적 체크(라우팅 intent, 금지 표현, 근거 있으면 출처 인용/없으면 정직한 인정)와
② `eval/judge.py`의 LLM-as-Judge(같은 gpt-5-mini로 별도 호출해 `faithful`/`relevant`/1~5점 채점). 또한
`observability.py`가 남긴 `OUTPUT_DIR/traces/{thread_id}.jsonl`을 파싱해 검색 스텝 수·react_loop 정지
사유·guardrail 재시도 횟수 같은 궤적 지표도 리포트에 포함시킨다.

```bash
python consultant_bot/eval/run_eval.py                       # test/output_test 인덱스 기준 (없으면 --output-dir 지정)
python consultant_bot/eval/run_eval.py --only faq
python consultant_bot/eval/run_eval.py --update-baseline     # 이번 결과를 새 기준선(eval/eval_baseline.json)으로 저장
```

기준선 대비 회귀가 있거나 평균 judge 점수가 `--min-score`(기본 3.0/5) 미만이면 0이 아닌 종료 코드를 반환한다.
케이스별 답변·judge 판정·트레이스 요약은 `OUTPUT_DIR/eval_reports/{timestamp}.json`에 저장된다. 실제 OpenAI
API를 호출하므로(그래프 실행 + judge 호출) 비용이 발생한다.

`--only faq`/`--only out_of_scope`/`--only design`으로 각각 따로 돌리면 이 환경에서는 매번 끝까지 정상
완료됨을 확인했다. 반면 `--only` 없이 네 케이스를 한 프로세스에서 전부 돌리면 위 "OMP: Error #15" 계열과
같은 종류의 네이티브 라이브러리 충돌로 모델 로딩 도중 간헐적으로 세그멘테이션 폴트(exit 139)가 날 수 있다
(재현이 일정하지 않음 - `run_eval.py`가 만든 버그는 아니고, `app.py`/`server.py`에서 이미 겪은 것과 같은
환경 이슈 계열). 안정적으로 돌리려면 당분간 `--only`로 나눠서 실행하는 것을 권장한다.

## 외부 트레이싱 (LangSmith, 선택)

`.env`에 `LANGCHAIN_API_KEY`를 설정하면 `observability.py`가 시작 시 이를 감지해 LangChain/LangGraph의
내장 트레이싱을 켠다 — 추가 코드 없이 모든 노드/tool 호출이 자동으로 LangSmith 웹 대시보드
(project: `consultant-bot`, `LANGCHAIN_PROJECT`로 변경 가능)에 올라간다. 키가 없으면 조용히 건너뛰고
기존처럼 `OUTPUT_DIR/traces/{thread_id}.jsonl` 로컬 트레이스만 남으므로, 별도 계정 없이도 그대로 동작한다.

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
  `graph.py`(시작 시 자동 복원, WAL 체크포인트 후 업로드)·`app.py`(매 턴 종료 후 백업 호출)에 연결.
- **2026-07-20**: `HF_TOKEN`을 write 권한이 있는 새 토큰으로 교체 → 업로드→삭제 왕복 테스트 성공
  (`yeardream-toy-project/lecture_pdf`에 실제 쓰기 가능 확인됨). 임베딩 인덱스 백업·consultant_bot 체크포인트
  백업이 정상 동작함을 확인.
- **2026-07-20**: 실제 배관(plumbing) 검증을 위해 `../test/`(별도 폴더)에 PDF 1개 격리 임베딩
  (`embed_one_pdf.py`, 비전 API 없이 텍스트 전용·CPU) + 자동 시나리오 테스트(`run_scenarios.py`: FAQ /
  out_of_scope / design 3턴 위저드) 하네스를 만들고 실제로 돌려 3개 버그를 찾아 **본 코드에** 수정:
  - `coordinator.py`: `DESIGN_MARKERS`("추천해" 등)가 도메인 키워드 없이도 즉시 매칭되어 "저녁 메뉴
    추천해줘" 같은 무관한 발화가 `design`으로, 도메인 키워드 목록이 좁아 "허깅페이스가 뭐야?"가
    `out_of_scope`로 오분류되던 버그 → design/faq 마커에 도메인 키워드 존재를 AND 조건으로 요구하도록
    수정하고, `DOMAIN_KEYWORDS`를 강의 자료의 실제 범위(Hugging Face 생태계·LLM/챗봇 구축 전반)에 맞춰 확장
  - `retrieval.py`: `QueryFusionRetriever`가 `num_queries=1`(질의 재생성 없음)이라 실제로는 안 쓰는데도
    생성자에서 `Settings.llm`을 즉시 해석하려다 `llama-index-llms-openai` 미설치로 `ImportError` 발생 →
    실제 호출되지 않는 자리이므로 `llama_index.core.llms.MockLLM`로 채워 해결(불필요한 의존성 추가 없음)
  - 루트 `requirements.txt`: `transformers`가 버전 미고정이라 5.14.0이 설치돼 있었는데, `FlagEmbedding`의
    reranker가 5.x에서 제거된 `tokenizer.prepare_for_model()`을 호출해 크래시 → `transformers<5.0.0`으로
    고정, 4.57.6으로 재설치(부수적으로 `huggingface_hub`도 0.36.2로 내려갔으나 `hf_storage.py` 정상 동작 확인)
  - 부수적으로 Windows 콘솔(cp949)이 LLM 응답에 흔한 em-dash 등 일부 유니코드를 인코딩 못 해 `print()`가
    죽을 수 있던 문제도 발견해 `app.py`에 UTF-8 출력 강제 추가
  - 수정 후 재실행 결과: FAQ(`intent=faq`, 출처 7곳 인용) / out_of_scope(검색 0회) / design 위저드
    (`interrupt` 3턴 전부 완주, `stage=done`) 모두 정상 동작 확인, guardrail 재시도 0회
- **2026-07-20**: gcube 본 실행이 `openai.BadRequestError: Model 'gpt-5-mini' is not allowed. Allowed
  models: ['openai/gpt-5-mini']`로 중단되는 것을 발견 → 게이트웨이(mlapi.run)가 `openai/gpt-5-mini`
  형식만 허용하는데, `.env`는 이미 맞게 설정돼 있었지만 `.gitignore` 대상이라 GitHub 빌드 이미지에는
  반영되지 않아 `config.py`의 기본값(`gpt-5-mini`)이 그대로 쓰이고 있었음 → `config.py`의 `VISION_MODEL`
  기본값을 `openai/gpt-5-mini`로 수정(`.env.example`도 동일하게 갱신). `llm.py`의 `CHAT_MODEL = VISION_MODEL`도
  이 값을 그대로 쓰므로 같이 해결됨.
- **2026-07-20**: 임베딩이 완료되면 재임베딩 없이 바로 챗봇/FAQ로 넘어가도록 루트 `Dockerfile`의 기본
  컨테이너 명령을 `main.py --use-hf` → `consultant_bot/app.py`로 전환하고, 그동안 이미지 빌드에 빠져있던
  `consultant_bot/requirements.txt`(`langgraph` 등)를 설치 단계에 추가(안 그러면 새 기본 명령이 `ImportError`로
  즉시 죽음). `retrieval.py`가 이미 HF `embedding/` 백업을 자동 복원하므로 이쪽 코드는 변경 없음.
- **2026-07-20**: 위 전환 직후 `app.py`가 터미널 `input()` 루프라 gcube처럼 컨테이너에 대화형 stdin이 안
  붙는 배포 환경에서는 시작하자마자 `EOFError`로 죽을 수 있다는 문제를 확인 → `server.py`(FastAPI,
  `/chat/message`·`/chat/resume`)를 새로 추가해 요청 1건 = `graph.invoke()` 1회로 interrupt/resume을
  매핑하고, `Dockerfile` 기본 명령을 `uvicorn consultant_bot.server:app --host 0.0.0.0 --port 8000`으로
  최종 전환. `fastapi`/`uvicorn`을 `consultant_bot/requirements.txt`에 추가. `app.py`는 로컬 터미널
  테스트용으로 계속 남겨둠(코드 변경 없음, HF 체크포인트 백업 등 그래프 로직은 `server.py`와 100% 공유).
- **2026-07-21**: `server.py`/`app.py`가 실제로 로컬에서 동작하는지 직접 실행해 검증 — 두 가지 문제를 찾아
  각각 원인을 확인하고 해결:
  - `transformers 5.x`가 설치된 환경에서 재랭커가 이전에 이미 발견했던 `tokenizer.prepare_for_model()`
    버그로 다시 크래시 → `transformers<5.0.0`(4.57.6)으로 재설치해 해결 확인. 루트 `requirements.txt`에는
    이미 이 버전 고정이 있으므로, 그 파일 기준으로 설치된 환경이면 재발하지 않음
  - 위 문제를 해결한 뒤에도 그래프 전체(`coordinator`→`faq_agent`→`guardrail` 등)를 처음 실행하면
    **세그멘테이션 폴트(exit code 139)로 파이썬 프로세스 자체가 죽는** 새 문제 발견 — torch/BGE-M3/
    리랭커/kiwipiepy가 각자 들고 있는 OpenMP 런타임(Windows의 `libiomp5md.dll` 등)이 한 프로세스에서
    중복 로드되면서 나는 "OMP: Error #15" 계열 크래시로 확인. `KMP_DUPLICATE_LIB_OK=TRUE` 환경변수를 주고
    재실행하니 크래시 없이 정상 동작 → `app.py`/`server.py` 맨 앞(다른 import보다 먼저)에
    `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`를 추가해, 별도 환경변수 설정 없이도 항상
    적용되도록 코드에 반영. Linux(gcube 컨테이너)에서는 이 종류의 DLL 중복이 훨씬 드물어 영향이 적을
    것으로 예상되지만, 어느 플랫폼에서도 부작용 없는 안전한 기본값이라 무조건 설정하도록 함
  - 두 수정 후 실제 LLM까지 호출하는 전체 대화 흐름을 라이브로 검증: FAQ 질문("허깅페이스가 뭐야?")이
    `intent=faq`로 정확히 분류되고, 검색 근거 3건(실제 PDF 파일명·페이지 포함)으로 "정의 → 사용 상황 →
    출처" 형식의 답변이 정상 생성, `guardrail.verdict.passed=True`(재시도 0회) 확인. 별개 질문("오늘
    저녁 메뉴 추천해줘")도 `intent=out_of_scope`로 정확히 분류되어 범위 안내 메시지로 즉시 종료되는 것도 확인
- **2026-07-21**: README "알려진 한계"에 있던 세 항목을 구현:
  - `cache.py`: `OUTPUT_DIR`의 인덱스 파일(docstore.json 등) mtime/size로 지문을 만들어 `cache_meta.json`에
    저장해두고, 프로세스 시작 시 지문이 바뀌었으면(재임베딩 발생) 검색/FAQ 캐시를 통째로 비우도록
    `_invalidate_stale_index_cache()` 추가. 지문이 그대로여도 `CACHE_TTL_SECONDS`(기본 14일)가 지난 항목은
    조회 시점에 폐기(lazy eviction)하도록 `SearchCache`/`FaqCache` 모두 수정. 수동 무효화용 `clear()`/
    `invalidate_all()`도 추가
  - `eval/`: `eval/cases.py`(FAQ/out_of_scope/design 시나리오 데이터셋) + `eval/judge.py`(LLM-as-Judge —
    `faithful`/`relevant`/1~5점) + `eval/run_eval.py`(E2E 하네스 — 그래프를 실제로 끝까지 실행하고 결정적
    체크 + judge 채점 + `traces/*.jsonl` 기반 궤적 지표를 합쳐 리포트/기준선 회귀 게이트까지 수행)를 추가
  - `observability.py`: `configure_langsmith()` 추가 — `.env`에 `LANGCHAIN_API_KEY`가 있으면 LangChain/
    LangGraph 내장 트레이싱을 켜서 추가 코드 없이 LangSmith 웹 대시보드로 전송(`graph.py`의
    `make_run_config`에 `thread_id` 태그/메타데이터 추가). 키가 없으면 조용히 건너뛰고 기존 로컬 JSONL
    트레이스만 남아, 외부 계정 없이도 그대로 동작
- **2026-07-21**: README "알려진 한계"의 마지막 항목("시스템 프롬프트 없음")을 구현:
  - `llm.py`: FAQ 페르소나 `SYSTEM_PROMPT`(역할/답변 원칙/출력 형식/우선순위) + `OUT_OF_CONTEXT_MESSAGE`(근거
    부족 시 정직한 안내 문구, faq_agent.py·guardrail.py 공용 상수) + `FORBIDDEN_HEDGE_PHRASES`(금지형 가드용)
    추가. `faq_agent.py`의 최종 답변 생성 호출에만 `SystemMessage`로 씌운다 — react_loop.py의 Reflect 등
    내부 추론 호출에는 적용하지 않음(역할이 다름)
  - `test/regression_faq.py`: 이 페르소나가 실제로 지켜지는지 케이스 x 이슈(routing/no_hedge/honesty) 매트릭스로
    검증하고 `test/regression_baseline.json`으로 회귀 게이트를 거는 하네스 추가(자세한 설명은
    [`../test/README.md`](../test/README.md) 참고). 첫 실행에서 **실제 버그를 발견** — `guardrail.py`의
    LLM 판정이 `OUT_OF_CONTEXT_MESSAGE`(정직한 "모른다" 안내) 자체를 "근거 밖 확정적 주장"으로 오판해 매번
    불필요한 재검색 1회를 더 태우고 있었음. 이 판정은 LLM 판단이 아니라 코드로 고정해야 하는 지점이라
    판단해 `guardrail.py`에 `output_text == OUT_OF_CONTEXT_MESSAGE`면 LLM 호출 없이 즉시 `passed=True`로
    처리하는 분기를 추가 → 재검증 후 재시도 0회, 가드레일 통과 시간이 케이스당 약 150~250초에서 1초 미만으로
    단축됨을 확인
- **2026-07-21**: 위 `eval/` 하네스를 `test/output_test` 인덱스로 실제 실행해 검증 — `--only faq`/
  `--only out_of_scope`/`--only design` 각각 끝까지 정상 완료(라우팅/캐시히트/react_loop 정지 사유/guardrail
  판정까지 트레이스에서 정확히 집계됨, 리포트 JSON 저장 확인). judge 점수는 이 1-PDF 테스트 인덱스가 애초에
  "RAG가 뭐야?"·"임베딩이 뭐야?" 같은 질문의 근거를 담고 있지 않아 낮게 나왔다(정직한 미확인 응답이 나온 것
  자체는 guardrail이 의도대로 동작한 것) - 실제 서비스 인덱스로 돌리면 점수가 달라질 것으로 예상. `--only`
  없이 네 케이스를 한 번에 돌렸을 때는 위 "품질 평가" 절에 적은 대로 세그멘테이션 폴트가 발생해, 당분간
  `--only`로 나눠 돌리도록 문서화함

## 알려진 한계 / 다음 단계

- 통합 테스트(`../test/`)는 PDF 1개짜리 인덱스 기준이라 커버리지가 제한적 — 자세한 실행법은
  [`../test/README.md`](../test/README.md) 참고. `eval/run_eval.py`도 같은 인덱스를 기본값으로 쓰므로 동일한
  제약을 공유한다.
- FAQ 페르소나 `SYSTEM_PROMPT`는 `faq_agent.py` 경로에만 적용된다 — Wizard/Presenter 경로는 별도의 구조화
  출력 프롬프트를 그대로 쓰며 톤/페르소나를 공유하지 않는다.
