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
        "선택된 기술({selected_techs}) 중 2개 이상을 조합한 파이프라인 후보 2개를 제시하세요. 근거에는 각 기술의 개별 정의만"
        "있을 뿐 '이 조합이 왜 맞는지'는 나와 있지 않습니다 — 근거에 있는 사실들을 조합해 이 요청에 맞는 새로운 "
        "구성 판단을 스스로 추론해서 만드세요(근거 문장을 그대로 옮기는 것으로 끝내지 마세요). 다만 각 기술 "
        "자체의 능력·특성은 근거에 있는 것만 사용하고 지어내지 마세요. 각 옵션의 source는 근거의 출처를 그대로 "
        "쓰세요.\n\n근거:\n{evidence}"
    ),
    "compare": (
        "선택된 파이프라인({selected_pipeline})을 다른 대안과 비교하는 표를 만드세요. 근거에는 개별 기술 정의만 "
        "있으므로, 두 파이프라인의 장단점·트레이드오프는 근거 사실을 조합해 직접 추론해서 채우세요(근거 문장의 "
        "재진술이 아니라 이 비교를 위한 새로운 판단이어야 합니다). 다만 각 기술 자체의 능력·특성은 근거에 있는 "
        "것만 사용하고 지어내지 마세요. 각 행의 source는 근거의 출처를 그대로 쓰세요.\n\n근거:\n{evidence}"
    ),
}
SCHEMAS = {
    "tech_select": TechSelectOutput,
    "pipeline_select": PipelineSelectOutput,
    "compare": CompareOutput,
}

# LLM이 생성하는 schema.prompt 필드는 PROMPTS(위 LLM 지시문)를 그대로 베끼는 경향이 있어(실측 확인),
# 화면에 노출되는 안내 문구는 LLM 출력에 맡기지 않고 고정 문구로 덮어쓴다.
UI_PROMPTS = {
    "tech_select": "포함하고 싶은 기술/컴포넌트를 모두 선택하고, 아래의 '선택완료'를 클릭해주세요.",
    "pipeline_select": "선택하신 기술을 조합한 설계안입니다. 원하시는 설계안을 선택하고, '이 파이프라인으로 진행'을 클릭해주세요.",
    "compare": "선택하신 설계안으로 진행하였을 때의 장단점 비교표입니다. 확인을 클릭해주세요.",
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
    output_data = output.model_dump()
    output_data["prompt"] = UI_PROMPTS.get(stage, output_data.get("prompt", ""))

    log_event(thread_id, "presenter", "generated", {"stage": stage})

    return Command(
        goto="guardrail",
        update={
            "presenter_output": output_data,
            "trace": [f"presenter: stage={stage} 구조화 출력 생성"],
        },
    )
