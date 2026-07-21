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
| `GET /` | `static/index.html` — 바닐라 JS 브라우저 채팅 화면. 아래 API를 그대로 호출하므로 별도 빌드 없이 gcube 워크로드의 공개 URL을 열면 바로 대화 가능 |
| `GET /health` | 상태 확인 |
| `POST /chat/message` `{message, thread_id?}` | 새 대화면 `thread_id` 생략(서버가 발급), 이어가는 대화면 이전 응답의 `thread_id`를 그대로 전달 |
| `POST /chat/resume` `{thread_id, resume}` | 직전 응답이 `status: "interrupt"`일 때만 호출. `resume` 형태는 `stage`별로 다름: `tech_select` → `["기술1", "기술2"]`, `pipeline_select` → `{"pipeline": "이름", "confirm": true}`, `compare` → `{"confirm": true}` |

`/chat/message`·`/chat/resume` 두 엔드포인트 모두 `{status: "interrupt", stage, payload}` 또는
`{status: "done", answer}`를 반환한다.

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
| `coordinator.py` / `faq_agent.py` / `lecture_agent.py` / `wizard_supervisor.py` / `research_worker.py` / `presenter.py` / `guardrail.py` | 7개 그래프 노드 — `lecture_agent.py`는 B형(추출형 강의자료 Q&A, react_loop 없이 검색 1회) |
| `graph.py` | `StateGraph` 조립 + `SqliteSaver` checkpointer |
| `app.py` | 터미널 대화형 실행 진입점 (로컬 개발/디버그용) |
| `server.py` | FastAPI HTTP 실행 진입점 (gcube 등 배포용 기본 진입점), `GET /`로 `static/index.html` 서빙 |
| `static/index.html` | 바닐라 JS 브라우저 채팅 UI (빌드 불필요, `server.py`의 `/chat/message`·`/chat/resume`만 호출) |
| `eval/` | E2E + LLM-as-Judge 평가 하네스 (`eval/run_eval.py`) — 아래 "품질 평가" 참고 |

## 프롬프트 위치

| 프롬프트 | 파일:줄 | 용도 |
|---|---|---|
| `SYSTEM_PROMPT` | `llm.py:61` | FAQ Agent 페르소나(역할/답변 원칙/출력 형식) — `faq_agent.py:65`에서 `SystemMessage`로 적용 |
| `OUT_OF_CONTEXT_MESSAGE` | `llm.py:56` | 근거 부족 시 정직한 안내 문구 (faq_agent·guardrail 공용 상수) |
| `FORBIDDEN_HEDGE_PHRASES` | `llm.py:59` | 금지 헤지 표현 목록 (평가 하네스의 결정적 체크용) |
| `ANSWER_PROMPT` | `faq_agent.py:24` | 근거 기반 "정의→사용 상황→출처" 답변 생성 프롬프트 (FAQ, 1턴) |
| `EXTRACT_ANSWER_PROMPT` | `lecture_agent.py:18` | 근거 기반 요약/예시/해석 답변 생성 프롬프트 (B형, 이전 답변 참조 포함) |
| `REFLECT_PROMPT` | `react_loop.py:23` | bounded-ReAct의 Reflect(다음 액션 판단) 단계 프롬프트 |
| `PROMPTS` (dict) | `presenter.py:15` | stage별(tech_select/pipeline_select/compare) 구조화 출력 프롬프트. pipeline_select/compare는 근거 사실을 조합한 합성 판단을 명시적으로 요구 |
| 의도 분류 프롬프트 (인라인) | `coordinator.py:84` | 규칙 기반 분류가 애매할 때 LLM 폴백용 faq/design/extract/out_of_scope 4지 분류 프롬프트 |
| `STRICT_VERDICT_PROMPT` | `guardrail.py:22` | faq/extract 경로용 — 근거 밖 확정 단언이면 무엇이든 실패 판정 |
| `SYNTHESIS_VERDICT_PROMPT` | `guardrail.py:36` | design 경로용 — 근거 사실을 조합한 합성 판단은 허용, 사실 날조만 실패 판정 |
| `JUDGE_PROMPT` | `eval/judge.py:16` | LLM-as-Judge 평가용 프롬프트 (품질 평가 하네스 전용, 런타임 아님) |

## 가드레일 위치 (4지점)

| # | 가드 지점 | 파일:줄 | 막는 대상 |
|---|---|---|---|
| 1 | 입력 가드 | `coordinator.py:69`, `coordinator.py:96` | 범위 밖 질문이 검색까지 도달하는 것 → `out_of_scope`면 검색 0회로 즉시 END |
| 2 | 추론 루프 가드 | `react_loop.py:94-138` | ReAct의 예산 초과(`budget_exhausted`)/반복(`duplicate_action`)/제자리맴돌기(`no_progress`) 3중 정지 (FAQ/design 경로. B형(`lecture_agent`)은 이 루프 대신 검색 1회로 고정) |
| 3 | 출력 스키마 가드 | `presenter.py:53` | 파싱 불가능한 형태의 응답 → `get_structured_llm`(Pydantic `with_structured_output`) 강제 |
| 4 | 출력 내용 가드 | `guardrail.py:92`(본 함수), `guardrail.py:131`(재시도 로직) | evidence에 없는 확정 표현/할루시네이션 → 규칙 우선 판정(`_rule_verdict`, `guardrail.py:58`) + 애매하면 intent별 LLM 검증(`faq`/`extract`는 엄격, `design`은 합성 허용) + `MAX_GUARDRAIL_RETRIES=1`회 재시도 후 안전 대체 응답 |

보조적으로 `llm.py:56-59`의 `OUT_OF_CONTEXT_MESSAGE`/`FORBIDDEN_HEDGE_PHRASES`가 위 4번 가드(`guardrail.py`)와 평가 하네스(`eval/run_eval.py:258`)의 판정 기준으로 공용 사용된다.

## 품질 평가 (E2E + LLM-as-Judge)

`../test/`(run_scenarios.py·regression_faq.py)가 "배관이 실제로 동작하는가"를 확인하는 것과 달리,
`eval/run_eval.py`는 `eval/cases.py`의 시나리오(FAQ/out_of_scope/design/extract)를 실제 그래프로 끝까지 실행한 뒤
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

`--only faq`/`--only out_of_scope`/`--only design`/`--only extract`으로 각각 따로 돌리면 이 환경에서는 매번
끝까지 정상 완료됨을 확인했다. 반면 `--only` 없이 네 케이스를 한 프로세스에서 전부 돌리면 위 "OMP: Error #15" 계열과
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
- **2026-07-21**: 로컬(Windows) 환경에서 torch/BGE-M3/reranker 로딩이 간헐적으로 세그폴트/`WinError 1455`
  (페이징 파일 부족)를 일으켜 gcube 클라우드 GPU 노드에서 돌리기로 결정 — 임베딩은 이미 HF에 백업돼 있으므로
  gcube 워크로드 "컨테이너 명령"란은 비워 기본 `CMD`(uvicorn `server:app`)만 실행하면 재임베딩 없이 바로
  챗봇/FAQ가 뜬다. `test/regression_faq.py`를 재실행해 `guardrail.py` 수정 반영 후 세 케이스 모두 통과함을
  확인하고 `test/regression_baseline.json`을 갱신
- **2026-07-21**: `server.py`에 `GET /` 추가 — `static/index.html`(바닐라 JS, 빌드 없음)을 서빙해 gcube
  워크로드의 공개 URL을 그냥 열면 브라우저에서 바로 대화할 수 있게 함. 기존 `/chat/message`·`/chat/resume`
  API를 그대로 호출하므로 서버 쪽 로직 변경은 없음
- **2026-07-21**: gcube 실사용 중 design 질문에서 `langgraph.errors.GraphRecursionError`(recursion_limit=8)
  발생 확인 — guardrail 재시도가 한 번만 걸려도(research_worker→wizard_supervisor→presenter→guardrail 4단계
  추가) happy path(6~7단계)만 겨우 맞던 8을 넘긴다는 걸 트레이스로 확인. 다만 근본 원인은 recursion 여유가
  아니라 "설계/FAQ 응답 자체가 느려서 재시도가 잦다"는 쪽으로 판단해 `recursion_limit`은 8로 유지하고
  속도 개선 쪽을 먼저 손대기로 함.
- **2026-07-21**: 응답 지연 원인 진단 — FAQ/design 둘 다 `hybrid_search`(BGE-M3 임베딩+BM25+cross-encoder
  리랭킹 30건)와 reasoning 모델(gpt-5-mini) structured-output 호출(reflect/presenter/guardrail)이 완전
  직렬로 여러 번 이어지는 구조라 정상 답변 케이스조차 순차 LLM 호출 3회 이상을 거침을 확인. 병렬화는 대부분
  단계 간 데이터 의존성(앞 단계 출력이 다음 단계 입력) 때문에 적용 불가하다고 판단하고, 대신 다음 3가지를
  적용:
  - `config.py`/`retrieval.py`: cross-encoder 리랭킹 대상을 `CANDIDATE_TOP_K`(30, RRF 융합 풀) 전체가 아니라
    RRF 점수 상위 `RERANK_CANDIDATE_LIMIT`(15)개로 축소 — RRF가 이미 점수 내림차순 정렬이라(LlamaIndex 소스
    확인) 정확도 손실 없이 리랭커 연산량만 절반 가까이 감소
  - `react_loop.py`: reflect 단계에 규칙 우선 판정(`_rule_sufficient`) 추가 — 이번 라운드 근거의 리랭커
    점수 상위 2건이 `RULE_SUFFICIENT_SCORE`(0.5) 이상이면 `reflect_llm` 호출 없이 즉시 종료, 애매할 때만
    기존처럼 LLM 폴백(coordinator.py의 "규칙 우선, 애매하면 LLM" 패턴과 동일). FAQ/design이 이 엔진을
    공유하므로 FAQ는 대부분 1라운드(reflect LLM 호출 0회)로 끝나되, 루프 자체는 유지해 첫 검색이 애매한
    질문에 대한 안전망은 보존
  - `guardrail.py`: verdict에도 같은 패턴으로 `_rule_verdict` 추가 — 응답에 인용된 `.pdf` 출처가 evidence
    목록과 일치하면 규칙으로 즉시 `passed=True/False`, 인용을 못 찾는 애매한 경우만 기존 `VERDICT_PROMPT`
    LLM 검증으로 폴백
  - 커밋 직후 kiwipiepy가 로컬 Windows에서 DLL 로딩 정책에 막혀 `eval/run_eval.py`를 못 돌리는 걸 확인해,
    kiwipiepy를 더미로 스텁하고 `_rule_sufficient`/`_rule_verdict`만 격리 단위 테스트(8케이스)로 검증하는
    과정에서 **실제 버그 발견**: `_rule_verdict`가 정규식(`\S+?\.pdf`)으로 출처를 추출했는데, 한국어 PDF
    파일명은 공백을 포함해서(`허깅페이스 개요 및 생태계 이해.pdf`) 마지막 단어만 잘려 정상 인용까지
    "근거 밖 출처"로 오판하는 상태였음 → 정규식 대신 evidence의 전체 출처 문자열을 텍스트에서 통째로
    제거해가며 남는 `.pdf` 언급이 있는지로 판정하도록 수정, 8케이스 전부 통과 확인. 전체 그래프 E2E는
    로컬 kiwipiepy 이슈로 미검증 — gcube 등 kiwipiepy가 정상 로드되는 환경에서 확인 필요

- **2026-07-21**: `chat_log/Atype faq_and_chatbot_scenarios.md`에서 챗봇 시나리오를 A형(추론형 설계
  컨설턴트)·B형(추출형 강의자료 Q&A) 이중 구조로 재설계한 것을 실제 코드에 반영:
  - B형(`lecture_agent.py` 신규): "강의 내용 요약해줘 → 예시 코드 작성해줘 → 이 코드 해석해줘" 같은 짧은
    연속 대화를 처리하는 7번째 그래프 노드 추가. `react_loop`의 반복 검색 대신 `hybrid_search` 1회만 호출하고
    (B형 설계: "검색 1회 + 원문 재구성 없는 서술"), `state.extract_source`에 첫 턴에서 찾은 문서 파일명을
    잠가 `retrieval.hybrid_search`의 신규 `source_filter` 인자로 이후 턴 검색 범위를 그 문서 하나로 좁힌다.
    `coordinator.py`는 `state.intent`를 4지(`faq`/`design`/`extract`/`out_of_scope`)로 확장하고, 도메인
    키워드가 없는 후속 턴("이 코드를 해석해줘")도 직전 턴이 `extract`였으면 최대 `EXTRACT_STICKY_LIMIT`(3)
    턴까지 계속 `extract`로 라우팅하는 sticky 규칙을 추가했다.
  - A형(위저드) 재조정: 기존 `guardrail.py`의 유일한 검증 프롬프트가 "근거 밖 확정 주장은 전부 실패"였는데,
    이 기준을 그대로 쓰면 위저드가 원래 해야 할 일(개별 기술 정의를 조합해 "이 서비스엔 어떤 파이프라인이
    맞는가"를 추천·비교하는 것)까지 근거 이탈로 오판해 차단할 위험이 있었다. `guardrail.py`를 `intent` 기반
    분기로 바꿔 `faq`/`extract`는 기존 엄격 판정(`STRICT_VERDICT_PROMPT`)을, `design`은 "근거 사실을 조합한
    새 추천/비교 판단은 허용하되 근거에 없는 기술 능력·사실을 지어내는 것만 차단"하는 신규
    `SYNTHESIS_VERDICT_PROMPT`를 쓰도록 분리했다. `presenter.py`의 `pipeline_select`/`compare` 프롬프트에도
    "근거를 조합해 새로운 판단을 스스로 추론하라"는 지시를 명시적으로 추가.
  - `eval/cases.py`에 `EXTRACT_CASES`, `eval/run_eval.py`에 `--only extract`(interrupt 없는 순차 3턴 러너)
    추가, `test/run_scenarios.py`에 `scenario_lecture_qa()`/`--only extract` 추가.
  - `test/output_test` 인덱스로 A형(`--only design`)·B형(`--only extract`) 실측 검증 중 `guardrail._rule_verdict`
    (인용 출처 문자열 대조 규칙)에서 실제 버그 2건 발견 및 수정: ① compare 단계처럼 같은 긴 한국어 파일명을
    여러 번 반복 인용할 때 LLM이 단 한 글자만 오타내도(예: "허깅페이스"->"허깱페이스") 나머지 정상 인용까지
    몽땅 "근거 밖 출처"로 오판해 재시도 2회를 다 태우고 안전 대체 응답으로 빠지던 문제 → 최소 한 번은 정확히
    인용됐다면 규칙으로 확정 실패시키지 않고 LLM 검증으로 폴백하도록 수정. ② `lecture_agent`(B형) 단일 인용
    응답에서 한 글자도 안 틀렸는데 실패 처리되는 사례 발견 → LLM이 화면상 동일한 한글을 완성형(NFC)이
    아니라 자모 분해형(NFD)으로 생성해 문자열 비교가 깨진 것으로 확인, 비교 전 양쪽을
    `unicodedata.normalize("NFC", ...)`로 정규화하도록 수정. 두 수정 모두 합성 데이터로 단위 검증 완료.
  - 위 수정 반영 후 재검증 중, `lecture_agent`가 `faq_agent`의 FAQ 페르소나(`llm.SYSTEM_PROMPT`, "5문장
    이내" 등 정의형 질문 전용 제약 포함)를 그대로 재사용하고 있어 요약처럼 분량이 필요한 요청을 근거가
    충분한데도 회피하는 현상을 발견 — `lecture_agent.py`에 분량 제약 없는 별도 `EXTRACT_SYSTEM_PROMPT`를
    추가해 교체.

## 알려진 한계 / 다음 단계

- 통합 테스트(`../test/`)는 PDF 1개짜리 인덱스 기준이라 커버리지가 제한적 — 자세한 실행법은
  [`../test/README.md`](../test/README.md) 참고. `eval/run_eval.py`도 같은 인덱스를 기본값으로 쓰므로 동일한
  제약을 공유한다.
- FAQ 페르소나 `SYSTEM_PROMPT`(`llm.py`)는 `faq_agent.py` 경로에만 적용된다. `lecture_agent.py`(B형)는 별도의
  `EXTRACT_SYSTEM_PROMPT`를 쓴다 — FAQ 페르소나의 "5문장 이내" 등 정의형 질문 전용 제약을 그대로 씌우면
  실측 검증(2026-07-21)에서 요약처럼 분량이 필요한 요청까지 근거가 충분한데도 위축돼 회피 응답을 내는
  현상이 관찰됐기 때문. Wizard/Presenter 경로는 이 둘과도 다른 구조화 출력 프롬프트를 쓴다.
- `lecture_agent.py`(B형)는 v1에서 캐시가 없다 — 3턴 안팎의 짧은 교환이라 FAQ류 반복 질문 캐싱 이득이 작고,
  임베딩 유사도 캐시가 "예시 코드 작성해줘" 같은 모호한 후속 질문을 다른 문서와 혼동시킬 위험이 더 크다고
  판단했다.
