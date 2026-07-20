# ① Coordinator — 진입점. faq / design / out_of_scope 3분기 + 입력 가드(가드레일 지점 1).
# 설계안 6번 우선순위대로 규칙 기반 분류를 먼저 적용하고, 애매한 경우에만 LLM structured output으로 폴백한다.

from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from chat_utils import last_human_text
from llm import get_structured_llm
from observability import log_event
from state import IntentDecision, WizardState

FAQ_MARKERS = ("란?", "이란", "뭐야", "무엇", "정의가", "머야", "차이가", "차이는")
DESIGN_MARKERS = ("설계", "만들고 싶", "만들어", "구성해", "구축", "파이프라인 추천", "아키텍처", "추천해")
# 강의 자료가 "에이전트 설계"뿐 아니라 Hugging Face 생태계·LLM/챗봇 구축 전반을 다루므로 넓게 잡는다.
DOMAIN_KEYWORDS = (
    "agent", "에이전트", "llm", "sllm", "rag", "임베딩", "embedding", "벡터", "vector", "bm25",
    "리트리버", "retriever", "langgraph", "langchain", "가드레일", "guardrail", "reAct",
    "react", "프롬프트", "prompt", "파이프라인", "체인", "chain", "멀티 에이전트", "supervisor",
    "허깅페이스", "hugging face", "huggingface", "허브", "hub", "모델", "model", "데이터셋", "dataset",
    "토크나이저", "tokenizer", "트랜스포머", "transformer", "파인튜닝", "fine-tuning", "챗봇", "chatbot", "mcp",
    "openai", "gpt", "ai",
)
OUT_OF_SCOPE_MESSAGE = (
    "이 챗봇은 AI 챗봇 구축/에이전트 설계 강의 자료 범위 안에서만 답할 수 있어요. "
    "강의 관련 용어 설명이나 에이전트·파이프라인 설계 상담을 요청해 주세요."
)


def _rule_based_intent(text: str) -> str | None:
    lowered = text.lower()
    has_design = any(marker in text for marker in DESIGN_MARKERS)
    has_faq = any(marker in text for marker in FAQ_MARKERS)
    has_domain = any(kw.lower() in lowered for kw in DOMAIN_KEYWORDS)

    # design/faq 마커든 도메인 키워드가 함께 있어야 확정한다 — "추천해줘"/"만들어줘" 같은 일반 동사
    # 어미만으로 무관한 발화("저녁 메뉴 추천해줘")가 design으로 오분류되는 것을 막는다.
    if has_design and has_domain:
        return "design"
    if has_faq and has_domain:
        return "faq"
    if not has_domain:
        return "out_of_scope"
    return None  # 도메인은 맞는데 faq/design 마커가 애매함 -> LLM 폴백


def coordinator(
    state: WizardState, config: RunnableConfig
) -> Command[Literal["faq_agent", "wizard_supervisor", "__end__"]]:
    text = last_human_text(state)
    thread_id = config["configurable"]["thread_id"]

    intent = _rule_based_intent(text)
    if intent is None:
        decision: IntentDecision = get_structured_llm(IntentDecision).invoke(
            f"다음 사용자 발화의 의도를 faq/design/out_of_scope 중 하나로 분류하세요.\n\n발화: {text}"
        )
        intent = decision.intent

    log_event(thread_id, "coordinator", "route", {"intent": intent, "text": text[:200]})

    if intent == "out_of_scope":
        return Command(
            goto=END,
            update={
                "intent": intent,
                "messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)],
                "trace": [f"coordinator: out_of_scope -> END (검색 0회)"],
            },
        )

    goto = "faq_agent" if intent == "faq" else "wizard_supervisor"
    return Command(goto=goto, update={"intent": intent, "trace": [f"coordinator: intent={intent} -> {goto}"]})
