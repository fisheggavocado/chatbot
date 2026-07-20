# ⑥ Guardrail — 최종 출력 가드(가드레일 지점 4). 근거 없는 확정 표현/evidence에 없는 기술 추천을 차단한다.
# 검증 실패 시 1회 재시도(FAQ -> faq_agent 재검색 / 위저드 -> research_worker 재검색),
# 재실패 시 예외를 던지지 않고 "자료에서 확인 불가"로 정직하게 대체한다 (Error as State 원칙).

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from cache import faq_cache
from chat_utils import last_human_text
from llm import get_structured_llm
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
SAFE_FALLBACK_TEXT = "이 부분은 강의 자료 근거가 충분하지 않아 확정적으로 답변드리기 어렵습니다. 자료에서 확인이 필요합니다."


def _format_evidence(evidence: list[dict]) -> str:
    return "\n".join(f"- [{e['source']} p.{e['page']}] {e['text'][:300]}" for e in evidence) or "(없음)"


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

    verdict: GuardrailVerdict = get_structured_llm(GuardrailVerdict).invoke(
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
                "messages": [AIMessage(content=SAFE_FALLBACK_TEXT)],
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
