# ③ Wizard Supervisor — 3턴 위저드의 지휘자. stage 커서를 읽어 다음 담당을 결정하고,
# 선택지 제시 후 interrupt()로 실행을 멈춰 사용자 선택을 기다린다 (human-in-the-loop).
#
# 이 노드는 사이클마다 3가지 다른 시점에 재진입한다:
#   (a) Coordinator에서 최초 진입 -> stage 초기화 후 research_worker로
#   (b) research_worker 완료 직후(presenter_output 없음) -> presenter로
#   (c) guardrail 통과 직후(presenter_output 있음) -> interrupt()로 사용자 선택 대기, 재개 시 stage 전진

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, interrupt

from chat_utils import last_human_text
from document_export import save_document
from observability import log_event
from state import Stage, WizardState

RESEARCH_BUDGET = 3


def _advance(stage: Stage, resume_value) -> tuple[Stage, dict, bool]:
    """(다음 stage, state 업데이트, 새 검색 필요 여부)를 반환한다."""
    if stage == "tech_select":
        techs = resume_value if isinstance(resume_value, list) else (resume_value or {}).get("selected", [])
        return "pipeline_select", {"selected_techs": techs}, True

    if stage == "pipeline_select":
        data = resume_value if isinstance(resume_value, dict) else {}
        pipeline = data.get("pipeline")
        confirm = data.get("confirm", True)
        if confirm:
            return "compare", {"selected_pipeline": pipeline}, True
        # 아니오 -> 같은 stage에서 이미 확보한 evidence로 다시 제시 (재검색 없음)
        return "pipeline_select", {"selected_pipeline": pipeline}, False

    if stage == "compare":
        return "done", {}, False

    return "done", {}, False


def wizard_supervisor(
    state: WizardState, config: RunnableConfig
) -> Command[Literal["research_worker", "presenter", "__end__"]]:
    thread_id = config["configurable"]["thread_id"]
    stage = state.get("stage")

    if stage is None or stage == "done":
        # stage="done"은 이전 위저드가 이미 끝났다는 뜻 — 같은 thread_id로 새 design 질문이 들어와도
        # 체크포인트에 "done"이 그대로 남아있어 `stage is None` 조건만으로는 재초기화가 안 되고,
        # presenter_output이 없다는 이유만으로 research_worker 없이 곧장 presenter로 잘못 넘어간다.
        log_event(thread_id, "wizard_supervisor", "init", {})
        return Command(
            goto="research_worker",
            update={"stage": "tech_select", "budget_remaining": RESEARCH_BUDGET, "trace": ["supervisor: init -> research_worker"]},
        )

    if state.get("presenter_output") is None:
        log_event(thread_id, "wizard_supervisor", "route_to_presenter", {"stage": stage})
        return Command(goto="presenter", update={"trace": [f"supervisor: stage={stage} -> presenter"]})

    # presenter_output이 guardrail을 통과해 돌아온 상태 -> 사용자 선택 대기
    log_event(thread_id, "wizard_supervisor", "interrupt_wait", {"stage": stage})
    resume_value = interrupt(state["presenter_output"])
    log_event(thread_id, "wizard_supervisor", "interrupt_resume", {"stage": stage, "resume_value": resume_value})

    stage_result = {"stage": stage, "output": state["presenter_output"], "resume": resume_value}
    next_stage, stage_updates, needs_research = _advance(stage, resume_value)
    update = {
        "stage": next_stage,
        "presenter_output": None,
        "trace": [f"supervisor: resume stage={stage} -> {next_stage}"],
        "wizard_results": [stage_result],
        **stage_updates,
    }

    if next_stage == "done":
        all_results = state.get("wizard_results", []) + [stage_result]
        doc_path = save_document(thread_id, last_human_text(state), all_results)
        log_event(thread_id, "wizard_supervisor", "document_saved", {"path": str(doc_path)})
        update["messages"] = [
            AIMessage(
                content=(
                    "여기까지 설계 상담을 정리했습니다. 지금까지의 선택 내용을 문서로 정리해 두었으니 "
                    "아래에서 다운로드하실 수 있습니다. 더 궁금한 점이 있으면 이어서 질문해 주세요."
                ),
                additional_kwargs={"download_path": str(doc_path)},
            )
        ]
        return Command(goto=END, update=update)

    if needs_research:
        update["budget_remaining"] = RESEARCH_BUDGET
        return Command(goto="research_worker", update=update)

    return Command(goto="presenter", update=update)
