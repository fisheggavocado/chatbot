# test/ — 멀티에이전트 통합 테스트 (PDF 1개 임베딩 → consultant_bot 시나리오)

## 목적
40개 PDF 전체를 gcube에서 본 실행하기 전에, **PDF 1개만으로** 전체 배관(pipeline: 임베딩 → 하이브리드 검색 →
6-에이전트 멀티에이전트)이 실제로 맞물려 동작하는지 **로컬(CPU, GPU 없음)에서** 빠르고 저비용으로 확인한다.
특히 FAQ 경로와 Wizard(설계 상담) 경로가 실제로 챗봇 응답을 만들어내는지 눈으로 확인하는 것이 목표.

## 범위/한계 (미리 알아둘 것)
- **비전 API 호출 없음**: 임베딩 단계(`embed_one_pdf.py`)는 `text_only_pdf_reader.TextOnlyPDFReader`로
  pypdfium2 텍스트 레이어만 추출한다 — gpt-5-mini 비전 API를 아예 타지 않는다. 로컬에 GPU가 없고, 이미지를
  LLM으로 처리하는 단계에서 문제가 생길 수 있는 환경(Windows Smart App Control 등)을 피하기 위함. 스캔
  이미지 위주 페이지는 텍스트가 거의/전혀 없이 인덱싱되지만, 멀티에이전트 배관 테스트가 목적이라 무방하다
  (실제 서비스용 인덱스는 항상 `main.py`의 비전 하이브리드 추출로 만들어야 함).
- 임베딩(BGE-M3)·리랭킹은 `DEVICE` 자동 감지로 CPU에서 그대로 동작한다 (`config.py`).
- PDF **딱 1개**만 임베딩하므로 검색 근거가 매우 제한적이다. 질문이 그 PDF 내용과 무관하면 `react_loop`가
  `budget_exhausted`/`no_progress`로 멈추고 "확인 불가"류 응답이 나올 수 있다 — 이건 실패가 아니라 정직한
  폴백이 정상 작동한다는 뜻이다 (`consultant_bot/DESIGN.md`의 "ReAct 폭주 제어" 참고).
- 2단계(`run_scenarios.py`)는 여전히 실제 OpenAI API(gpt-5-mini, 텍스트 전용)를 호출하므로 비용이 발생한다
  (Coordinator/Reflect/Presenter/Guardrail LLM 호출) — 1단계와 달리 이건 비전 API가 아니라 채팅 API다.
- 진짜 `output/`, HF의 `embedding/`/`checkpoints/`는 전혀 건드리지 않는다 — 모두 `test/output_test/`에
  격리된다 (`.gitignore`로 제외돼 있어 커밋되지 않음).
- PDF 자체는 `HF_REPO_ID`에서 목록을 동기화(`snapshot_download`)한 뒤 그중 1개만 처리한다 — 첫 실행 시 PDF
  40개 전체를 내려받는 시간이 걸릴 수 있다 (이후엔 캐시돼 빠름).

## 사전 준비
```bash
pip install -r consultant_bot/requirements.txt   # 아직 안 했다면
```
`.env`에 `OPENAI_API_KEY`, `HF_TOKEN`, `HF_REPO_ID`가 이미 설정돼 있어야 한다 (루트 `.env` 그대로 사용).

## 실행 순서 (first_project/ 에서 실행)

### 1단계 — PDF 1개 임베딩 (텍스트 전용, 비전 API 없음, CPU)
```bash
python test/embed_one_pdf.py
```
`HF_REPO_ID`에서 PDF를 동기화해 그중 1개만 텍스트 레이어만으로 처리하고 `test/output_test/`에 인덱스를
만든다. 콘솔에 어떤 PDF가 처리됐는지 파일명이 출력되니, 2단계에서 질문을 그 내용에 맞게 조정하고 싶으면
참고한다. (텍스트 레이어가 거의 없는 스캔 위주 PDF가 뽑히면 검색 근거가 아주 적을 수 있다 — 그럴 땐
`--clean` 후 다시 실행하면 다른 순서/다른 PDF로 재시도할 수 있다.)

### 2단계 — 멀티에이전트 시나리오 실행
```bash
python test/run_scenarios.py
```
세 가지 시나리오를 순서대로 자동 실행한다:

| # | 시나리오 | 확인하는 것 |
|---|---|---|
| 1 | **FAQ** | 정의형 질문 → Coordinator가 `faq_agent`로 라우팅하는지, bounded ReAct로 검색해 출처 포함 답변을 만드는지 |
| 2 | **out_of_scope** | 범위 밖 질문 → 검색 없이 즉시 안내 메시지로 끝나는지 |
| 3 | **design (Wizard)** | 설계 상담 질문 → `interrupt()`로 1턴(기술 선택)에 멈추는지, 제시된 선택지 중 첫 번째를 자동으로 골라 재개했을 때 2턴(파이프라인)·3턴(비교)까지 진행되는지 (내용에 의존하지 않는 일반적 선택이라 어떤 PDF든 동작) |

각 시나리오 끝에 `test/output_test/traces/{thread_id}.jsonl`에서 상세 트레이스(어떤 tool을 몇 번 불렀는지,
정지 사유 등)를 확인할 수 있다.

특정 시나리오만 돌리거나 질문을 바꾸고 싶으면:
```bash
python test/run_scenarios.py --only faq --faq-question "임베딩이 뭐야?"
python test/run_scenarios.py --only design --design-question "..."
```

## 성공 기준
- **FAQ**: `intent=faq`로 라우팅되고, 답변에 출처(PDF 파일명)가 포함됨 (근거가 없으면 "확인 불가" 정직한
  폴백도 정상 — 위 "범위/한계" 참고)
- **out_of_scope**: `intent=out_of_scope`로 라우팅되고 검색 없이(트레이스에 `hybrid_search` 호출 0회) 즉시
  안내 메시지
- **design**: interrupt가 최소 1번은 발생하고(1턴), 자동 재개 후 stage가 전진함. 근거 부족으로 2~3턴에서
  "확인 불가"류 폴백이 나올 수도 있음 (PDF 1개뿐이므로 정상)

## 원상복구
```bash
python test/embed_one_pdf.py --clean
```
`test/output_test/`를 삭제한다 (진짜 `output/`이나 HF는 애초에 안 건드렸으므로 별도 정리 불필요. PDF
캐시 `hf_pdfs/`는 다른 스크립트와 공유하는 폴더라 그대로 둔다).
