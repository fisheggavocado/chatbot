# Consultant Bot — Multi-Agent 구현 설계

원본 설계안: `../chat_log/AI 설계 컨설턴트 봇 — Multi Agent 구성안.py`

`first_project`의 기존 파이프라인(PDF → gpt-5-mini 비전 추출 → BGE-M3 임베딩 → LlamaIndex 저장 → HF Dataset
`embedding/` 폴더 백업)에서 만든 인덱스를 그대로 검색 도구로 재사용해, 그 위에 6-에이전트 LangGraph 대화형
멀티에이전트(Wizard 플로우)를 얹은 것이 이 폴더다.

범위: 핵심 파이프라인(①~⑥ 에이전트 + WizardState + Guardrail) + 시맨틱 캐시(`cache.py`, 인덱스 지문 기반
자동 무효화 + TTL 만료) + E2E·LLM-as-Judge 평가 하네스(`eval/`) + 선택적 LangSmith 외부 트레이싱까지 구현.
`trace` 필드와 `observability.py`의 JSONL 로그가 `eval/run_eval.py`가 궤적 지표(검색 스텝 수, react_loop
정지 사유, guardrail 재시도 횟수)를 뽑아내는 원자료로도 쓰인다.

## 파일 구조 및 역할

| 파일 | 역할 |
|---|---|
| `state.py` | `WizardState` TypedDict + 노드 간 구조화 출력 스키마(Pydantic) |
| `llm.py` | 공용 `ChatOpenAI(gpt-5-mini)` 팩토리. 부모 `config.py`의 `OPENAI_API_KEY`/`VISION_MODEL` 재사용 |
| `retrieval.py` | `hybrid_search` — 벡터(BGE-M3)+BM25를 RRF로 융합 후 cross-encoder로 rerank. `verify_embeddings.py`의 `build_retriever(mode="hybrid")` 패턴 재사용. 로컬 인덱스가 없으면 `hf_storage.restore_output_from_hf()`로 HF Dataset의 `embedding/` 백업에서 자동 복원 |
| `react_loop.py` | FAQ Agent/Research Worker가 공유하는 bounded-ReAct 엔진 (아래 "ReAct 폭주 제어" 참고) |
| `observability.py` | `TraceCallbackHandler` + `log_event()` — 노드/tool 호출을 콘솔과 `OUTPUT_DIR/traces/{thread_id}.jsonl`에 기록 |
| `cache.py` | `SearchCache`(hybrid_search 정확 일치 캐시) + `FaqCache`(FAQ 임베딩 유사도 캐시). 둘 다 `OUTPUT_DIR`에 JSON으로 write-through 저장, 인덱스 지문 기반 자동 무효화 + TTL 만료 (아래 "중간 저장" 참고) |
| `chat_utils.py` | `last_human_text(state)` — 메시지 목록에서 가장 최근 사용자 발화만 정확히 찾는 공유 헬퍼 |
| `coordinator.py` | ① 진입 노드. 규칙 기반 우선 + 애매하면 LLM 폴백으로 `faq`/`design`/`extract`/`out_of_scope` 4분기 |
| `faq_agent.py` | ② bounded ReAct(≤2회) → "정의+사용 상황+출처" 포맷 응답 (A형과 별개의 1턴 개념 정의) |
| `wizard_supervisor.py` | ③ stage 커서 관리 + `interrupt()`로 사용자 선택 대기 (human-in-the-loop) — A형(추론형 설계) |
| `research_worker.py` | ④ bounded ReAct(≤3회)로 위저드 각 턴의 근거 수집, 항상 Supervisor로 복귀 |
| `presenter.py` | ⑤ stage별 Pydantic structured output (기술 체크박스 표 / 파이프라인 2개 / 비교표). pipeline_select·compare는 근거 사실을 조합한 새 추천/비교 판단(합성)을 명시적으로 요구 |
| `lecture_agent.py` | ⑦ B형(추출형 강의자료 Q&A). react_loop 없이 `hybrid_search` 1회만 호출, `extract_source`로 단일 문서 범위를 잠가 연속 턴(요약→예시→해석)을 처리 |
| `guardrail.py` | ⑥ 최종 출력 검증. `faq`/`extract`는 근거 밖 확정 주장을 전부 차단하는 엄격 판정, `design`은 근거 사실을 조합한 합성은 허용하고 사실 날조만 차단하는 판정 — intent별로 분리. 실패 시 1회 재시도 후 정직한 대체 응답 |
| `graph.py` | `StateGraph` 조립, `SqliteSaver` checkpointer, `recursion_limit=8` |
| `app.py` | 터미널 대화형 실행 진입점 (`interrupt()`/`Command(resume=...)` 루프 포함) |

## 그래프 라우팅

```
coordinator ─faq──────────────▶ faq_agent ─────▶ guardrail ──▶ END
            ─extract──────────▶ lecture_agent ─▶ guardrail ──▶ END
            ─design───────────▶ wizard_supervisor
            ─out_of_scope─────▶ END (검색 0회)

wizard_supervisor(초기) ──────▶ research_worker ──▶ wizard_supervisor ──▶ presenter ──▶ guardrail
                                                                                          │
                                                            pass ─────────────────────────┤
                                                                                          ▼
                                                                                wizard_supervisor
                                                                          (interrupt, 사용자 선택 대기)
                                                                                          │
                                                            stage 전진 ────────────────────┤
                                                            (다음 stage 필요시) ──▶ research_worker
                                                            (같은 stage 재확인) ──▶ presenter
                                                            (stage="done") ──▶ END
```

모든 노드가 `Command(goto=...)`로 스스로 다음 노드를 지정한다(LangManus 패턴). `guardrail` 실패 시에는
`faq`→`faq_agent`, `extract`→`lecture_agent`, `design`→`research_worker`로 1회만 재시도한다
(`guardrail_retries` 카운터).

`extract`(B형)는 `faq`처럼 1회 통과 후 바로 END지만, `lecture_agent`는 `react_loop`를 쓰지 않고
`hybrid_search`를 직접 1회만 호출하며, `WizardState.extract_source`에 첫 턴에서 찾은 문서 파일명을
잠가 이후 턴("예시 코드 작성해줘"처럼 주제어가 없는 후속 질문)의 검색 범위를 그 문서로 좁힌다.
`coordinator.py`는 `prev_intent=="extract"`이고 도메인 키워드가 없어도(최대 `EXTRACT_STICKY_LIMIT`턴까지)
`extract`로 계속 라우팅하는 sticky 규칙으로 이 연속 대화를 이어준다.

## ReAct 폭주 제어 (`react_loop.py`)

"ReAct는 한 번 생각을 잘못하면 계속 틀린다"는 문제를 프롬프트 지시가 아니라 **코드로 강제**한다.
`create_react_agent` 같은 블랙박스 prebuilt 대신 명시적 Python 루프로 구현했고, 3가지 정지 조건이
LLM의 판단(Reflect 단계)과 무관하게 우선 적용된다:

1. **`budget_exhausted`** — `budget_remaining`이 0이 되면 즉시 종료 (FAQ 2회 / Research 3회 상한)
2. **`duplicate_action`** — 정규화된 검색어를 반복하면 "같은 생각에 갇혔다"고 보고 즉시 종료
3. **`no_progress`** — 2턴 연속 새로운 출처가 하나도 안 나오면("표현만 바꿔가며 제자리를 맴돎") 조기 종료

세 조건 중 하나로 멈추면 예외를 던지지 않고 `sufficient=False`와 `stopped_reason`을 반환한다. 상위 노드는
이를 보고 확정 답변을 지어내지 않고 "자료에서 확인 불가/제한적 근거"로 정직하게 응답한다 — 설계안의
"Error as State" 원칙을 정지 사유에도 동일하게 적용한 것이다.

## 가드레일 4지점

설계안 표(⑥ Guardrail 행)의 "guardrail 4지점"을 실제 코드 위치로 매핑:

| # | 가드 지점 | 막는 대상 | 위치 |
|---|---|---|---|
| 1 | 입력 가드 | 범위 밖 질문이 검색까지 도달하는 것 | `coordinator.py` (`out_of_scope` → 검색 0회, 즉시 END) |
| 2 | 추론 루프 가드 | ReAct가 예산 초과/반복/제자리맴돌기 하는 것 | `react_loop.py` (budget/중복/정체 3중 정지) |
| 3 | 출력 스키마 가드 | 파싱 불가능한 형태의 응답 | `presenter.py` (Pydantic `with_structured_output` 강제) |
| 4 | 출력 내용 가드 | evidence에 없는 확정 표현/할루시네이션 | `guardrail.py` (LLM 검증 + 1회 재시도). `faq`/`extract`는 근거 밖 확정 주장을 전부 차단(`STRICT_VERDICT_PROMPT`), `design`은 근거 사실을 조합한 추천/비교 판단(합성)은 통과시키고 사실 날조만 차단(`SYNTHESIS_VERDICT_PROMPT`) — intent별로 판정 기준이 다르다 |

## Observability

`TraceCallbackHandler`가 체인/tool 호출을 콘솔에 자동 출력하고, `log_event()`가 콜백으로 못 잡는 지점
(Coordinator 라우팅 결정, react_loop 정지 사유, Guardrail pass/fail, `interrupt()` 진입/재개)까지
`OUTPUT_DIR/traces/{thread_id}.jsonl`에 append한다. 이 로그만으로 "이 선택지가 어떤 검색에서 나왔는지"
사후 복원과 "tool을 몇 번 불렀는지" 궤적 점검이 가능하다.

**외부 트레이싱(LangSmith, 선택)**: `.env`에 `LANGCHAIN_API_KEY`가 있으면 `observability.configure_langsmith()`가
이를 감지해 `LANGCHAIN_TRACING_V2`/`LANGCHAIN_PROJECT` 환경변수를 세팅한다 — LangChain/LangGraph가 이 두
환경변수만으로 전역 트레이서를 활성화하므로, 별도 코드 없이 모든 노드/tool 실행이 LangSmith 웹 대시보드로
자동 전송된다(`graph.py`의 `make_run_config`가 `thread_id`를 tag/metadata로 붙여 필터링을 돕는다). 키가
없으면 `configure_langsmith()`가 조용히 아무 것도 하지 않고, 위 JSONL 로컬 트레이스만 남는다 — LangSmith
계정이 없는 환경(로컬 개발, course-project 채점 등)에서도 동일하게 동작한다.

## 품질/정확도 튜닝 파라미터

| 파일 | 파라미터 | 효과 |
|---|---|---|
| `config.py` | `EMBEDDING_MODEL`, `RERANK_MODEL` | 더 큰/좋은 임베딩·리랭커로 교체 시 검색 품질↑ (속도·메모리↑) |
| `config.py` | `CHUNK_SIZE`/`CHUNK_OVERLAP` | 청크를 작게 하면 정밀도↑ 문맥 손실 위험↑, 크게 하면 반대 (재인덱싱 필요) |
| `config.py` | `CANDIDATE_TOP_K`/`RERANK_TOP_K` | 후보/최종 근거 개수. 늘리면 재현율↑, LLM 컨텍스트·비용↑ |
| `llm.py` | `CHAT_MODEL`, `REASONING_EFFORT` | 더 큰 모델일수록 정확도↑. gpt-5 계열은 `temperature`를 지원하지 않아(400 에러) `reasoning_effort`(minimal~high)로 대신 조절 — 높일수록 정확도↑ 지연·비용↑ |
| `faq_agent.py`/`research_worker.py` | `MAX_STEPS` | ReAct 재검색 상한. 늘리면 어려운 질문에 강해지나 지연·비용↑ |
| `react_loop.py` | `NO_PROGRESS_LIMIT` | 낮추면 빨리 포기, 높이면 더 끈질기게 재검색(비용↑) |
| `retrieval.py` | `NUM_QUERY_EXPANSIONS` | 2~3으로 올리면 LLM이 질의를 여러 각도로 확장 (재현율↑, LLM 호출 추가) |
| `guardrail.py` | `MAX_GUARDRAIL_RETRIES` | 재시도 횟수. 늘리면 방어력↑, 지연·비용↑ |
| `cache.py` | `FAQ_SIMILARITY_THRESHOLD` | 낮추면 캐시 히트가 늘어 토큰 절약↑, 오답 재사용 위험↑ |
| `graph.py` | `RECURSION_LIMIT` | 그래프 전체 스텝 상한 |

## 중간 저장 — 데이터 유실 방지 & 토큰 절약

네 겹으로 저장된다:

1. **그래프 상태 (SqliteSaver)** — `graph.py`의 checkpointer가 노드가 끝날 때마다(superstep 단위)
   `WizardState` 전체를 `thread_id`별로 로컬 `OUTPUT_DIR/consultant_bot_checkpoints.sqlite`에 저장. 프로세스가
   죽어도 같은 thread_id로 재개하면 그 시점부터 이어진다. **로컬 기본 경로**(환경변수 `OUTPUT_DIR` 미설정 시):
   `first_project/output/consultant_bot_checkpoints.sqlite`. 클라우드/컨테이너에서는 `OUTPUT_DIR` 환경변수가
   가리키는 위치(예: `/data/output`)의 같은 파일명.
2. **HF 백업 (checkpoints/)** — 임베딩 인덱스(`hf_storage.upload_output_to_hf`/`restore_output_from_hf`,
   `embedding/` 경로)와 같은 패턴으로, 이 sqlite 파일도 `hf_storage.upload_checkpoint_to_hf`/
   `restore_checkpoint_from_hf`가 `HF_REPO_ID`의 `checkpoints/` 경로와 주고받는다. `graph.py`가 로컬에
   체크포인트가 없고 `HF_REPO_ID`가 설정돼 있으면 시작 시 자동 복원하고, `app.py`가 매 턴이 끝날 때마다
   `backup_checkpoint()`로 재업로드한다(WAL 모드라 업로드 전에 `PRAGMA wal_checkpoint(TRUNCATE)`로 먼저 메인
   파일에 합침). `HF_REPO_ID`가 없으면 조용히 건너뛰고, 업로드 실패도 예외를 삼켜 대화를 막지 않는다.

   PDF/임베딩 인덱스 읽기와 쓰기 모두 같은 `HF_TOKEN`/`HF_REPO_ID`를 쓴다(이 repo에 쓰기 권한이 있는 토큰을
   쓰는 것을 전제로 함 — 읽기 전용 토큰이면 업로드가 403으로 막히니 쓰기 권한이 있는 repo/토큰으로 바꿔야 함).
3. **ReAct 스텝별 스냅샷 (JSONL)** — `react_loop.py`가 매 스텝마다 그 스텝에서 얻은 evidence 전체를
   `observability.log_event()`를 통해 `OUTPUT_DIR/traces/{thread_id}.jsonl`에 append한다. 그래프 노드가
   끝나기 전(예: Research Worker의 3스텝 중 2번째)에 프로세스가 죽어도, 이미 쓴 검색/Reflect 토큰의 결과물인
   evidence 자체는 이 로그에서 복구할 수 있다.
4. **캐시 (`cache.py`, JSON, write-through)**
   - `SearchCache`: `hybrid_search(query)` 결과를 정규화된 쿼리 문자열 기준 정확 일치로
     `OUTPUT_DIR/search_cache.json`에 캐싱. 같은 문구로 재검색(다른 노드, 다른 세션 포함)하면 벡터/BM25/rerank
     연산을 건너뛴다. LLM 토큰 자체를 아끼진 않지만 지연시간과 compute를 아낀다.
   - `FaqCache`: FAQ 질문을 BGE-M3(이미 로컬에 로드되어 있어 추가 비용 없음) 임베딩으로 벡터화해
     `OUTPUT_DIR/faq_cache.json`에 저장. 코사인 유사도가 `FAQ_SIMILARITY_THRESHOLD`(기본 0.92) 이상인 질문이
     재입력되면 `faq_agent.py`가 react_loop(검색+Reflect LLM)와 답변 생성 LLM 호출을 **모두 생략**하고 즉시
     응답한다 — 세 저장 방식 중 토큰 절약 효과가 가장 크다. 캐시에는 `guardrail.py`가 evidence 검증을 통과시킨
     답변만 들어간다(미검증 답변이 캐시에 쌓이는 것을 방지).
   - 무효화(invalidation): 두 겹으로 처리한다. ① `OUTPUT_DIR`의 인덱스 파일(`docstore.json` 등) mtime/size로
     지문을 만들어 `cache_meta.json`에 저장해두고, 프로세스 시작 시 지문이 바뀌었으면(재임베딩 발생) 두 캐시
     파일을 통째로 비운다 — PDF 자료가 갱신됐는데 과거 캐시가 재사용되는 것을 막는다. ② 지문이 그대로여도
     `CACHE_TTL_SECONDS`(기본 14일)가 지난 항목은 조회 시점에 폐기한다(lazy eviction) — 자료는 안 바뀌었지만
     오래된 캐시를 계속 재사용하는 것을 막는 보수적 안전장치다. `cache.invalidate_all()`로 수동 초기화도
     가능하다.

## 설치 및 실행

```bash
# first_project/.venv 활성화 상태에서
pip install -r consultant_bot/requirements.txt
python consultant_bot/app.py
```

`DEVICE`는 `config.py`가 `torch.cuda.is_available()`로 자동 감지한다 — 로컬(GPU 없음)에서는 자동으로 CPU,
CUDA가 있는 GPU 클라우드에서는 자동으로 GPU를 쓴다. 필요하면 `.env`에 `DEVICE=cpu`/`DEVICE=cuda`로 강제 지정 가능.

`OUTPUT_DIR`(`../output`)에 로컬 인덱스가 없으면, `.env`의 `HF_REPO_ID`에서 `embedding/` 백업을 자동
복원한다. 복원할 백업도 없다면 먼저 `python main.py` (또는 `--use-hf`)로 인덱스를 만들어야 한다.

## 검증 체크리스트

- FAQ성 질문 → FAQ Agent 경로로 1~2회 검색 후 출처 포함 응답
- 설계 관련 질문(A형) → 1턴(기술 선택) → `interrupt()` → 2턴(파이프라인+예/아니오) → 3턴(비교표) 순서로 진행,
  2·3턴의 추천·비교 문장이 근거 문장의 재진술이 아니라 합성된 판단인지 확인
- 강의자료 요약/예시/해석 요청(B형) → `lecture_agent`가 interrupt 없이 1턴씩 응답, 2·3턴(주제어 없는 후속
  질문)이 1턴에서 잠긴 `extract_source` 문서 범위 안에서만 검색되는지 확인
- 범위 밖 질문 → Coordinator가 검색 없이 즉시 안내 후 종료
- PDF에 없는 내용 질문(FAQ) → `react_loop`가 `budget_exhausted`/`no_progress`로 조기 종료하고 "확인 불가"로
  응답, `traces/{thread_id}.jsonl`에 정지 사유가 기록되는지 확인 / PDF에 없는 내용 질문(B형) → `lecture_agent`의
  단일 검색 결과가 비면 동일하게 "확인 불가"로 응답
- evidence 밖 확정 표현을 유도하는 질문 → `guardrail.py`가 재시도 후에도 실패하면 안전한 대체 응답으로 교체
  (A형은 `SYNTHESIS_VERDICT_PROMPT`가 "합성은 허용, 사실 날조만 차단"으로 판정하므로 정상 추천까지 막히지
  않는지도 함께 확인)

## 알려진 단순화 지점 (TODO)

- `stage="detail"`은 `state.py`의 `Stage` 타입에는 남겨뒀지만 `wizard_supervisor.py`는 아직 라우팅하지 않음
  (설계안 3턴 흐름에는 없고, 4턴 확장 시 사용할 자리로 예약)
- `eval/cases.py`의 시나리오는 소수(FAQ 2개/out_of_scope 1개/design 1개/extract 1개)라 커버리지가 제한적 —
  새 케이스는 파일에 항목만 추가하면 된다(하네스 코드 변경 불필요)
- `lecture_agent.py`(B형)는 v1에서 캐시를 두지 않는다 — 3턴 안팎의 짧은 교환이라 FAQ류 "반복 질문" 캐싱
  이득이 작고, 임베딩 유사도 캐시가 "예시 코드 작성해줘" 같은 모호한 후속 질문을 다른 문서와 혼동시킬 위험이
  더 크다고 판단
- `retrieval.hybrid_search`의 `source_filter`는 사전 필터링만 하고, 필터 후 후보 수가 `RERANK_CANDIDATE_LIMIT`
  보다 적게 남는 경우(해당 문서 청크가 원래 top-K 후보에 적게 뽑힌 경우)에 대한 재현율 보강은 하지 않음
- (수정 완료, 기록용) `guardrail._rule_verdict`가 긴 한국어 파일명을 여러 행에 걸쳐 반복 인용하는 compare
  단계에서, LLM이 단 한 글자만 오타내도(예: "허깅페이스"->"허깱페이스") 나머지 정상 인용까지 몽땅
  "근거 밖 출처"로 오판해 재시도를 소진시키던 버그를 실측 검증(2026-07-21) 중 발견해 수정 — 최소 한 번
  정확히 인용된 적이 있으면 규칙으로 확정 실패시키지 않고 LLM 검증으로 폴백하도록 변경
- (수정 완료, 기록용) 같은 검증에서 `lecture_agent`(B형) 단일 인용 응답이 한 글자도 안 틀렸는데도
  `_rule_verdict`가 실패 처리하는 사례 발견 — LLM이 화면상 동일한 한글을 완성형(NFC)이 아니라 자모
  분해형(NFD)으로 생성해 문자열 비교가 깨졌던 것. 비교 전 양쪽을 `unicodedata.normalize("NFC", ...)`로
  정규화하도록 수정
