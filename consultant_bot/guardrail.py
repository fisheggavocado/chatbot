# ⑥ Guardrail — 최종 출력 가드(가드레일 지점 4). 근거 없는 확정 표현/evidence에 없는 기술 추천을 차단한다.
# 검증 실패 시 1회 재시도(FAQ -> faq_agent 재검색 / 위저드 -> research_worker 재검색),
# 재실패 시 예외를 던지지 않고 "자료에서 확인 불가"로 정직하게 대체한다 (Error as State 원칙).

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

VERDICT_PROMPT = """아래 응답이 근거(evidence)에 없는 내용을 확정적으로 단언하고 있는지 검증하세요.
근거에 있는 내용만 다루면 passed=true, 근거를 벗어난 확정적 주장이 있으면 passed=false로 답하세요.

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
    """
    evidence_sources = sorted({e["source"] for e in evidence if e.get("source")}, key=len, reverse=True)
    if not evidence_sources:
        return None

    remaining = output_text
    matched_known_source = False
    for source in evidence_sources:
        if source in remaining:
            matched_known_source = True
            remaining = remaining.replace(source, "")

    if not matched_known_source and ".pdf" not in remaining:
        return None  # 인용 자체를 못 찾음 -> LLM 검증으로 폴백
    return ".pdf" not in remaining  # 알려진 출처를 지우고도 .pdf가 남으면 근거 밖 출처 인용


def guardrail(
    state: WizardState, config: RunnableConfig
) -> Command[Literal["faq_agent", "research_worker", "wizard_supervisor", "__end__"]]:
    thread_id = config["configurable"]["thread_id"]
    is_faq_path = state.get("presenter_output") is None
    retries = state.get("guardrail_retries", 0)

    output_text = (
        str(state["messages"][-1].content)
        if is_faq_path
        else str(state.get("presenter_output"))
    )

    if is_faq_path and output_text.strip() == OUT_OF_CONTEXT_MESSAGE:
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
            verdict = get_structured_llm(GuardrailVerdict).invoke(
                VERDICT_PROMPT.format(output=output_text, evidence=_format_evidence(state.get("evidence", [])))
            )
    log_event(thread_id, "guardrail", "verdict", {"passed": verdict.passed, "reason": verdict.reason, "retries": retries})

    if verdict.passed:
        if is_faq_path:
            # guardrail을 통과한 FAQ 답변만 시맨틱 캐시에 저장 — 검증 안 된 답을 캐싱해 계속 재사용하는 것을 방지.
            # (자료 자체가 바뀌면 과거 캐시가 stale해질 수 있다는 트레이드오프는 감수한다 — course-project 범위)
            faq_cache.store(last_human_text(state), output_text, state.get("evidence", []))
        goto = END if is_faq_path else "wizard_supervisor"
        return Command(goto=goto, update={"guardrail_retries": 0, "trace": ["guardrail: pass"]})

    if retries < MAX_GUARDRAIL_RETRIES:
        goto = "faq_agent" if is_faq_path else "research_worker"
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
    if is_faq_path:
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
