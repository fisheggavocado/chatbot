# ① Coordinator — 진입점. faq / design / extract / out_of_scope 4분기 + 입력 가드(가드레일 지점 1).
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

FAQ_MARKERS = ("란?", "이란", "뭐야", "무엇", "정의가", "머야", "차이가", "차이는", "원리는", "원리가", "개념", "정의")
DESIGN_MARKERS = ("설계", "만들고 싶", "만들어", "구성해", "구축", "파이프라인 추천", "아키텍처", "추천해", "프로세스")
# B형(추출형 강의자료 Q&A) 트리거 — 특정 강의자료 하나를 요약/예시/해석해달라는 요청.
# "설명해줘"처럼 FAQ와 겹칠 수 있는 범용 동사는 넣지 않는다 (요약/예시/해석은 FAQ_MARKERS와 겹치지 않음).
EXTRACT_MARKERS = (
    "요약해줘", "요약해서", "정리해줘", "예시 코드", "예시를 작성", "코드 작성", "코드를 작성",
    "이 코드", "코드를 해석", "해석해줘", "강의 내용을",
)
# 이 턴 수 미만에서만 도메인 키워드 없이도 "이전 턴이 extract였으면 계속 extract" sticky 라우팅을 허용한다.
# 무한정 허용하면 주제 없는 잡담이 extract에 갇힐 수 있어 상한을 둔다.
EXTRACT_STICKY_LIMIT = 3
# 강의 자료가 "에이전트 설계"뿐 아니라 Hugging Face 생태계·LLM/챗봇 구축 전반을 다루므로 넓게 잡는다.
DOMAIN_KEYWORDS = (
    "agent", "에이전트", "llm", "sllm", "rag", "임베딩", "embedding", "벡터", "vector", "bm25",
    "리트리버", "retriever", "langgraph", "langchain", "가드레일", "guardrail", "reAct", "react",
    "langmanus", "프롬프트", "prompt", "파이프라인", "pipeline", "체인", "chain", "멀티 에이전트", "supervisor",
    "허깅페이스", "hugging face", "huggingface", "허브", "hub", "모델", "model", "데이터셋", "dataset",
    "토크나이저", "tokenizer", "트랜스포머", "transformer", "파인튜닝", "fine-tuning", "챗봇", "chatbot", "mcp",
    "openai", "gpt", "ai",
)
OUT_OF_SCOPE_MESSAGE = (
    "이 챗봇은 AI 챗봇 구축/에이전트 설계 강의 자료 범위 안에서만 답할 수 있어요. "
    "강의 관련 용어 설명이나 에이전트·파이프라인 설계 상담을 요청해 주세요."
)


def _rule_based_intent(text: str, prev_intent: str | None, extract_turns: int) -> str | None:
    lowered = text.lower()
    has_design = any(marker in text for marker in DESIGN_MARKERS)
    has_faq = any(marker in text for marker in FAQ_MARKERS)
    has_extract = any(marker in text for marker in EXTRACT_MARKERS)
    has_domain = any(kw.lower() in lowered for kw in DOMAIN_KEYWORDS)

    # design/faq/extract 마커든 도메인 키워드가 함께 있어야 확정한다 — "추천해줘"/"만들어줘" 같은 일반 동사
    # 어미만으로 무관한 발화("저녁 메뉴 추천해줘")가 design으로 오분류되는 것을 막는다.
    if has_design and has_domain:
        return "design"
    if has_faq and has_domain:
        return "faq"
    if has_extract and has_domain:
        return "extract"

    # B형 연속 턴("이 코드를 해석해줘" 등)은 도메인 키워드가 전혀 없는 경우가 많다 — 직전 턴이 extract였고
    # 다른 의도로 명확히 전환되지 않았다면 도메인 키워드 없이도 이어간다 (상한 EXTRACT_STICKY_LIMIT까지만).
    if prev_intent == "extract" and extract_turns < EXTRACT_STICKY_LIMIT and not has_design and not has_faq:
        return "extract"

    if not has_domain:
        return "out_of_scope"
    return None  # 도메인은 맞는데 faq/design/extract 마커가 애매함 -> LLM 폴백


def coordinator(
    state: WizardState, config: RunnableConfig
) -> Command[Literal["faq_agent", "wizard_supervisor", "lecture_agent", "__end__"]]:
    text = last_human_text(state)
    thread_id = config["configurable"]["thread_id"]
    prev_intent = state.get("intent")
    extract_turns = state.get("extract_turns", 0)

    intent = _rule_based_intent(text, prev_intent, extract_turns)
    if intent is None:
        decision: IntentDecision = get_structured_llm(IntentDecision).invoke(
            "다음 사용자 발화의 의도를 faq/design/extract/out_of_scope 중 하나로 분류하세요.\n\n"
            "faq: 개념 정의 질문(1턴, 예: 'RAG가 뭐야?')\n"
            "design: 서비스 설계/추천 요청(여러 턴에 걸친 위저드, 예: '~봇을 만들고 싶어. 설계해줘')\n"
            "extract: 특정 강의자료 하나의 요약/예시코드/해석을 요구하는 짧은 연속 대화 "
            "(예: '~강의 내용을 요약해줘', '이 코드를 해석해줘')\n"
            "out_of_scope: 강의 범위 밖 요청\n\n"
            f"발화: {text}"
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

    route = {"faq": "faq_agent", "extract": "lecture_agent", "design": "wizard_supervisor"}[intent]
    update = {"intent": intent, "trace": [f"coordinator: intent={intent} -> {route}"]}
    if intent == "extract":
        if prev_intent != "extract":
            update["extract_source"] = None
            update["extract_turns"] = 1
        else:
            update["extract_turns"] = extract_turns + 1

    return Command(goto=route, update=update)
