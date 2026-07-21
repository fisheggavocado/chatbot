# ⑥ Guardrail — 최종 출력 가드(가드레일 지점 4). intent별로 판정 기준이 다르다:
#   - faq/extract: 근거 밖 확정 주장은 무엇이든 차단(STRICT_VERDICT_PROMPT) — B형은 원문 재진술만 허용.
#   - design: 근거 사실을 조합한 새 추천/비교 판단(합성)은 허용하되 사실 날조만 차단(SYNTHESIS_VERDICT_PROMPT).
# 검증 실패 시 1회 재시도(faq->faq_agent / extract->lecture_agent / design->research_worker),
# 재실패 시 예외를 던지지 않고 "자료에서 확인 불가"로 정직하게 대체한다 (Error as State 원칙).

import re
import unicodedata
from typing import Literal, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from cache import faq_cache
from chat_utils import last_human_text
from llm import OUT_OF_CONTEXT_MESSAGE, get_structured_llm
from observability import log_event
from state import GuardrailVerdict, WizardState

# faq/extract 경로용 — 근거를 벗어난 어떤 확정 주장도 차단하는 엄격 판정. B형(lecture_agent)은 원문
# 재진술만 허용해야 하므로 faq와 동일한 엄격도를 쓴다.
STRICT_VERDICT_PROMPT = """아래 응답이 근거(evidence)에 없는 내용을 확정적으로 단언하고 있는지 검증하세요.
근거에 있는 내용만 다루면 passed=true, 근거를 벗어난 확정적 주장이 있으면 passed=false로 답하세요.

응답:
{output}

근거:
{evidence}
"""

# design(위저드) 경로 전용 — 개별 기술 근거를 조합한 새로운 추천/비교 판단(합성)은 정상 추론이므로 통과시키되,
# 근거 어디에도 없는 기술의 능력·사실을 지어내는 것만 차단한다. presenter.py가 위저드에 종합 판단을 명시적으로
# 요구하므로, guardrail도 "근거 문장의 재진술이 아니면 실패"였던 기존 엄격 기준을 그대로 쓰면 정상 응답까지
# 차단하게 된다.
SYNTHESIS_VERDICT_PROMPT = """아래 응답은 여러 기술에 대한 근거(evidence)를 조합해 사용자의 요청에 맞는
추천·비교·트레이드오프 판단을 내놓은 것입니다. 근거의 개별 사실을 조합해 새로운 추천/비교 판단을 내리는 것은
정상적인 추론이므로 통과시키세요(passed=true). 다만 evidence 어디에도 등장하지 않는 기술의 능력·성능수치·사실
관계를 지어내 단언하면 passed=false로 답하세요.

응답:
{output}

근거:
{evidence}
"""

MAX_GUARDRAIL_RETRIES = 1
# 위저드(design) 경로 전용 대체 문구. FAQ 경로는 llm.OUT_OF_CONTEXT_MESSAGE로 통일해
# faq_agent.py의 근거 부족 메시지와 문구가 갈라지지 않게 한다.
SAFE_FALLBACK_TEXT = "이 부분은 강의 자료 근거가 충분하지 않아 확정적으로 답변드리기 어렵습니다. 자료에서 확인이 필요합니다."


def _format_evidence(evidence: list[dict]) -> str:
    return "\n".join(f"- [{e['source']} p.{e['page']}] {e['text'][:300]}" for e in evidence) or "(없음)"


# 강의 자료 파일명 앞에 흔히 붙는 "7-5. ", "[강의자료] " 같은 번호/대괄호 라벨 접두어.
_FILENAME_PREFIX_RE = re.compile(r"^\s*(?:\d+(?:-\d+)?\.\s*|\[[^\]]*\]\s*)")


def _strip_filename_prefixes(source: str) -> str:
    """출처 파일명 앞의 번호/대괄호 라벨 접두어를 모두 벗긴 핵심 파일명을 반환한다.

    접두어가 없으면(스트리핑해도 원본과 같으면) 빈 문자열을 반환해, 호출부가 "접두어가 있던
    출처"만 골라 2차 매칭에 쓰도록 한다.
    """
    core = source
    while True:
        stripped = _FILENAME_PREFIX_RE.sub("", core)
        if stripped == core:
            break
        core = stripped
    return core if core != source else ""


def _rule_verdict(output_text: str, evidence: list[dict]) -> Optional[bool]:
    """규칙 우선 판정 (coordinator.py의 "규칙 우선, 애매할 때만 LLM" 패턴과 동일).

    presenter/faq_agent 프롬프트가 출처를 evidence의 파일명 그대로 쓰도록 강제하므로
    (presenter.py PROMPTS, faq_agent.py ANSWER_PROMPT), 응답에 등장하는 ".pdf" 인용이 실제
    evidence 출처 목록에 있는지만 봐도 "근거에 없는 걸 지어냄"의 흔한 형태(가짜 출처 인용)를
    LLM 호출 없이 잡아낼 수 있다.

    한국어 PDF 파일명은 공백을 포함하므로("허깅페이스 개요 및 생태계 이해.pdf") 정규식으로
    토큰 하나만 잘라내는 방식은 실제 출처 문자열을 반토막 내 오탐을 낸다 — 대신 알려진 evidence
    출처 문자열을 통째로 텍스트에서 제거해가며, 그러고도 ".pdf" 언급이 남는지로 판정한다.
    인용을 아예 못 찾으면(형식이 다르거나 근거 자체가 없음) None을 반환해 기존 VERDICT_PROMPT
    LLM 검증으로 폴백한다 — 애매한 경우까지 규칙으로 확정하지 않는다.

    한 응답 안에서 같은 긴 한국어 파일명을 여러 행에 걸쳐 반복 인용하는 compare/pipeline_select 같은
    stage에서는, LLM이 한 번이라도 한 글자를 오타내면(예: "허깅페이스"->"허깱페이스") 그 한 줄만 못
    지워 나머지 정상 인용까지 몽땅 "근거 밖 출처"로 오판되는 문제가 실측 검증(2026-07-21, design
    compare stage)에서 발견됐다. 이미 같은 문서를 최소 한 번 정확히 인용했다면(matched_known_source)
    가짜 출처를 지어냈다기보다 오탈자일 가능성이 높으므로, 이 경우는 규칙으로 확정(False)하지 않고
    LLM 검증(VERDICT_PROMPT류)으로 폴백한다. 반대로 알려진 출처가 단 한 번도 안 나왔는데 ".pdf"가
    있으면(완전히 낯선 출처를 지어낸 경우) 그대로 확정 실패(False)로 둔다.

    같은 검증(2026-07-21, lecture_agent 단일 인용 응답)에서 한 글자도 안 틀렸는데 문자열 비교가 실패하는
    사례도 발견됐다 — LLM이 한글 문자열을 생성할 때 완성형(NFC)이 아니라 자모 분해형(NFD)으로 내보내는
    경우가 있어, 화면에는 똑같이 보여도 `==`/`in` 비교로는 다른 문자열이 된다. 그래서 비교 전에 양쪽을
    NFC로 정규화한다 — 눈에 보이는 문자만 같으면 항상 참이 되도록.

    실측 검증(2026-07-21, faq_agent "rag가 뭐야?")에서 세 번째 문제도 발견됐다: evidence 출처가
    "4-3. [강의자료] RAG (검색 증강 생성) 기초.pdf"처럼 번호/대괄호 라벨 접두어를 달고 있는데, LLM이
    답변 내용 자체는 완전히 근거에 부합하면서도 인용할 때 그 접두어만 생략하고("RAG (검색 증강 생성)
    기초.pdf") 핵심 파일명만 쓰는 경우가 있다 — 정상 인용인데도 원본 문자열과 정확히 안 맞아 "근거 밖
    출처를 지어냄"으로 오판되고, 재시도까지 소진되면 멀쩡한 답변이 안전 대체 응답으로 버려진다. 그래서
    원본 문자열이 그대로 없으면, 그 앞의 번호("7-5. ")·대괄호 라벨("[강의자료] ") 접두어를 벗긴 핵심
    파일명으로도 한 번 더 찾아본다 — 이 핵심 파일명은 접두어가 있던 출처에만 존재하므로(접두어가 없는
    출처는 원본 그대로가 이미 핵심 파일명이라 이 2차 매칭에서 다시 걸리지 않는다), 짧아서 다른 문맥을
    잘못 지우는 오탐 위험도 낮다.
    """
    normalize = lambda s: unicodedata.normalize("NFC", s)  # noqa: E731
    output_norm = normalize(output_text)
    evidence_sources = sorted(
        {normalize(e["source"]) for e in evidence if e.get("source")}, key=len, reverse=True
    )
    if not evidence_sources:
        return None

    remaining = output_norm
    matched_known_source = False
    for source in evidence_sources:
        if source in remaining:
            matched_known_source = True
            remaining = remaining.replace(source, "")

    # 원본 그대로는 없었지만, 번호/대괄호 라벨 접두어만 뺀 핵심 파일명으로 인용한 경우도 정상 인용으로 인정.
    core_sources = sorted(
        {core for core in (_strip_filename_prefixes(s) for s in evidence_sources) if core},
        key=len, reverse=True,
    )
    for core in core_sources:
        if core in remaining:
            matched_known_source = True
            remaining = remaining.replace(core, "")

    if not matched_known_source:
        # 알려진 출처가 단 한 번도 안 나옴: .pdf 인용 자체가 없으면(비교/판단할 인용이 없음) 이 규칙이
        # 판단할 게 없으므로 None(폴백), .pdf 인용은 있는데 전혀 안 맞으면 낯선 출처를 지어낸 것이므로 False.
        return None if ".pdf" not in remaining else False
    # 최소 한 번은 알려진 출처를 정확히 인용함: 남는 .pdf가 없으면 완전히 정상(True), 있으면 위에서 설명한
    # "오탈자로 인한 오탐" 가능성이 있으므로 규칙으로 확정하지 않고 LLM 검증으로 폴백(None).
    return None if ".pdf" in remaining else True


# intent별로 다른 판정/재시도/캐시 규칙을 적용하기 위한 3-way 테이블 (design만 presenter_output을 씀).
_RETRY_NODE = {"faq": "faq_agent", "extract": "lecture_agent", "design": "research_worker"}


def guardrail(
    state: WizardState, config: RunnableConfig
) -> Command[Literal["faq_agent", "lecture_agent", "research_worker", "wizard_supervisor", "__end__"]]:
    thread_id = config["configurable"]["thread_id"]
    intent = state.get("intent")
    is_wizard_path = intent == "design"
    retries = state.get("guardrail_retries", 0)

    output_text = str(state.get("presenter_output")) if is_wizard_path else str(state["messages"][-1].content)

    if intent in ("faq", "extract") and output_text.strip() == OUT_OF_CONTEXT_MESSAGE:
        # "모른다"는 정직한 안내 자체는 근거를 벗어난 확정적 주장이 아니다. 그런데 LLM judge에게 맡기면
        # "질문이 없다고 확정적으로 단언한다"는 표면적 문구만 보고 종종 오판해(회귀 테스트로 실제 확인됨)
        # 불필요한 재검색 루프를 태운다 — 이 판정은 LLM 판단이 아니라 코드로 고정한다.
        verdict = GuardrailVerdict(passed=True, reason="정직한 안내 문구는 확정적 주장이 아니므로 코드로 즉시 통과")
    else:
        rule_result = _rule_verdict(output_text, state.get("evidence", []))
        if rule_result is not None:
            verdict = GuardrailVerdict(
                passed=rule_result,
                reason=(
                    "규칙 판정: 인용된 출처(.pdf)가 모두 근거 목록에 존재함"
                    if rule_result
                    else "규칙 판정: 근거 목록에 없는 출처를 인용함"
                ),
            )
        else:
            prompt = SYNTHESIS_VERDICT_PROMPT if is_wizard_path else STRICT_VERDICT_PROMPT
            verdict = get_structured_llm(GuardrailVerdict).invoke(
                prompt.format(output=output_text, evidence=_format_evidence(state.get("evidence", [])))
            )
    log_event(thread_id, "guardrail", "verdict", {"passed": verdict.passed, "reason": verdict.reason, "retries": retries})

    if verdict.passed:
        if intent == "faq":
            # guardrail을 통과한 FAQ 답변만 시맨틱 캐시에 저장 — 검증 안 된 답을 캐싱해 계속 재사용하는 것을 방지.
            # (자료 자체가 바뀌면 과거 캐시가 stale해질 수 있다는 트레이드오프는 감수한다 — course-project 범위)
            faq_cache.store(last_human_text(state), output_text, state.get("evidence", []))
        goto = "wizard_supervisor" if is_wizard_path else END
        return Command(goto=goto, update={"guardrail_retries": 0, "trace": ["guardrail: pass"]})

    if retries < MAX_GUARDRAIL_RETRIES:
        goto = _RETRY_NODE[intent]
        return Command(
            goto=goto,
            update={
                "guardrail_retries": retries + 1,
                "last_error": {"type": "guardrail_fail", "retryable": True, "reason": verdict.reason},
                "presenter_output": None,  # 재시도 후 presenter가 다시 생성하도록 stale 값 제거
                "trace": [f"guardrail: fail ({verdict.reason}) -> {goto} 재시도"],
            },
        )

    # 재실패: 정직한 대체 응답
    if not is_wizard_path:
        return Command(
            goto=END,
            update={
                "guardrail_retries": 0,
                "messages": [AIMessage(content=OUT_OF_CONTEXT_MESSAGE)],
                "trace": ["guardrail: 재실패 -> 안전한 대체 응답"],
            },
        )
    return Command(
        goto="wizard_supervisor",
        update={
            "guardrail_retries": 0,
            "presenter_output": {"rows": [], "options": [], "prompt": SAFE_FALLBACK_TEXT},
            "trace": ["guardrail: 재실패 -> 안전한 대체 응답으로 위저드 계속"],
        },
    )
