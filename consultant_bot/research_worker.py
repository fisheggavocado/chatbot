# ④ Research Worker — bounded ReAct(상한 3회)로 위저드 각 턴에 필요한 근거를 수집한다.
# 결과는 항상 Wizard Supervisor로 복귀한다. 근거 수집 로직 자체(예산/중복/정체 제어)는 react_loop.py 공유.

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from chat_utils import last_human_text
from react_loop import run_bounded_react
from state import WizardState

MAX_STEPS = 3


def _build_query(state: WizardState) -> str:
    """현재 stage에 맞춰 검색 쿼리를 구성한다."""
    stage = state.get("stage", "tech_select")
    if stage == "pipeline_select":
        techs = ", ".join(state.get("selected_techs", []))
        return f"{techs} 기술을 조합한 에이전트 파이프라인 구성 방법"
    if stage == "compare":
        return f"{state.get('selected_pipeline', '')} 파이프라인 방식의 장단점 비교"
    return last_human_text(state)  # tech_select 등 기본값: 원 사용자 요청


def research_worker(state: WizardState, config: RunnableConfig) -> Command[Literal["wizard_supervisor"]]:
    thread_id = config["configurable"]["thread_id"]
    query = _build_query(state)

    result = run_bounded_react(
        node_name="research_worker",
        thread_id=thread_id,
        question=query,
        max_steps=MAX_STEPS,
        budget_remaining=MAX_STEPS,
    )

    last_error = None
    if not result.sufficient:
        last_error = {"type": result.stopped_reason, "retryable": True}

    return Command(
        goto="wizard_supervisor",
        update={
            "evidence": result.evidence,
            "trace": result.trace,
            "budget_remaining": result.budget_remaining,
            "last_error": last_error,
        },
    )
