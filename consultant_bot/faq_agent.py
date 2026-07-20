# ② FAQ Agent — 정의형 질문 처리. bounded ReAct(상한 2회)로 검색 -> "정의 + 사용 상황 + 출처" 포맷으로 종결.
# 위저드 진입 없음. react_loop.run_bounded_react가 예산/중복/정체 3중 정지를 강제하므로,
# 여기서는 결과를 포맷팅하는 역할만 한다.
#
# 시맨틱 캐시(cache.FaqCache): 비슷한 질문이 이미 guardrail을 통과해 캐시에 저장된 적 있으면
# 검색(react_loop)과 답변 생성 LLM 호출을 모두 건너뛰고 즉시 END로 응답한다 (토큰 절약 핵심 지점).

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from cache import faq_cache
from chat_utils import last_human_text
from llm import OUT_OF_CONTEXT_MESSAGE, SYSTEM_PROMPT, get_llm
from observability import log_event
from react_loop import run_bounded_react
from state import WizardState

MAX_STEPS = 2

ANSWER_PROMPT = """아래 근거만 사용해 질문에 답하세요. "정의 -> 사용 상황 -> 출처" 순서로 구성하고,
마지막 문장은 반드시 [출처: 파일명 p.페이지] 형식으로 끝내세요. 근거에 없는 내용은 절대 추가하지 마세요.

질문: {question}

근거:
{evidence}
"""


def _format_evidence(evidence: list[dict]) -> str:
    return "\n".join(f"- [{e['source']} p.{e['page']}] {e['text'][:300]}" for e in evidence) or "(없음)"


def faq_agent(state: WizardState, config: RunnableConfig) -> Command[Literal["guardrail", "__end__"]]:
    thread_id = config["configurable"]["thread_id"]
    question = last_human_text(state)

    cache_hit = faq_cache.lookup(question)
    if cache_hit is not None:
        log_event(thread_id, "faq_agent", "cache_hit", {"score": cache_hit["score"]})
        return Command(
            goto=END,
            update={
                "evidence": cache_hit["evidence"],
                "trace": [f"faq_agent: 시맨틱 캐시 히트(score={cache_hit['score']:.3f}) -> 검색/LLM 생략"],
                "messages": [AIMessage(content=cache_hit["answer"])],
            },
        )

    result = run_bounded_react(
        node_name="faq_agent",
        thread_id=thread_id,
        question=question,
        max_steps=MAX_STEPS,
        budget_remaining=MAX_STEPS,
    )

    if result.sufficient:
        answer = get_llm().invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=ANSWER_PROMPT.format(question=question, evidence=_format_evidence(result.evidence))),
            ]
        ).content
    else:
        answer = OUT_OF_CONTEXT_MESSAGE

    return Command(
        goto="guardrail",
        update={
            "evidence": result.evidence,
            "trace": result.trace,
            "budget_remaining": result.budget_remaining,
            "last_error": None if result.sufficient else {"type": result.stopped_reason, "retryable": False},
            "messages": [AIMessage(content=answer)],
        },
    )
