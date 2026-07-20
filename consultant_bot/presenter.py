# ⑤ Presenter — Research가 모은 evidence로 턴별 출력물 생성.
# 1턴: 기술+역할 체크박스 표 / 2턴: 파이프라인 2개 + 예/아니오 / 3턴: 비교표.
# 가드레일 지점 3(출력 스키마 가드): Pydantic structured output을 강제해 UI가 파싱 가능한 형태만 통과시킨다.

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from chat_utils import last_human_text
from llm import get_structured_llm
from observability import log_event
from state import CompareOutput, PipelineSelectOutput, TechSelectOutput, WizardState

PROMPTS = {
    "tech_select": (
        "아래 근거를 바탕으로 사용자의 요청에 맞는 기술/컴포넌트 후보와 각 역할을 표로 정리하세요. "
        "각 행의 source는 반드시 근거의 출처(PDF 파일명)를 그대로 쓰세요.\n\n요청: {question}\n\n근거:\n{evidence}"
    ),
    "pipeline_select": (
        "선택된 기술({selected_techs})을 조합한 파이프라인 후보 2개를 근거 기반으로 제시하세요. "
        "각 옵션의 source는 근거의 출처를 그대로 쓰세요.\n\n근거:\n{evidence}"
    ),
    "compare": (
        "선택된 파이프라인({selected_pipeline})을 다른 대안과 근거 기반으로 비교하는 표를 만드세요. "
        "각 행의 source는 근거의 출처를 그대로 쓰세요.\n\n근거:\n{evidence}"
    ),
}
SCHEMAS = {
    "tech_select": TechSelectOutput,
    "pipeline_select": PipelineSelectOutput,
    "compare": CompareOutput,
}


def _format_evidence(evidence: list[dict]) -> str:
    return "\n".join(f"- [{e['source']} p.{e['page']}] {e['text'][:300]}" for e in evidence) or "(없음)"


def presenter(state: WizardState, config: RunnableConfig) -> Command[Literal["guardrail"]]:
    thread_id = config["configurable"]["thread_id"]
    stage = state.get("stage", "tech_select")
    schema = SCHEMAS.get(stage, TechSelectOutput)

    question = last_human_text(state)
    prompt = PROMPTS.get(stage, PROMPTS["tech_select"]).format(
        question=question,
        evidence=_format_evidence(state.get("evidence", [])),
        selected_techs=", ".join(state.get("selected_techs", [])),
        selected_pipeline=state.get("selected_pipeline", ""),
    )

    output = get_structured_llm(schema).invoke(prompt)

    log_event(thread_id, "presenter", "generated", {"stage": stage})

    return Command(
        goto="guardrail",
        update={
            "presenter_output": output.model_dump(),
            "trace": [f"presenter: stage={stage} 구조화 출력 생성"],
        },
    )
