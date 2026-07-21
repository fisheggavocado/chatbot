# ⑦ Lecture Agent — B형(추출형 강의자료 Q&A) 처리. 특정 강의자료 하나를 요약/예시/해석해달라는
# 짧은 연속 대화(보통 3턴)를 다룬다. A형(위저드)과 달리 새로운 조합·판단을 만들지 않고, 검색된 원문
# 범위 안에서만 답한다 — 그래서 react_loop의 반복 검색·자기교정 루프를 쓰지 않고 검색을 1회만 한다
# (chat_log/Atype faq_and_chatbot_scenarios.md의 B형 설계: "검색 1회 + 원문 발췌·재구성 없는 서술").
#
# 단일 문서 범위 고정: 첫 턴에서 찾은 문서(state.extract_source)로 이후 턴의 검색을 좁혀, 턴2("예시 코드
# 작성해줘")처럼 주제어가 빠진 후속 질문도 엉뚱한 문서에서 검색되지 않게 한다.

from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command

from chat_utils import last_ai_text, last_human_text
from llm import OUT_OF_CONTEXT_MESSAGE, get_llm
from observability import log_event
from retrieval import hybrid_search
from state import WizardState

# faq_agent.py의 SYSTEM_PROMPT("FAQ 상담 챗봇", 5문장 이내 등)를 그대로 재사용하지 않는다 — 요약/예시코드
# 작성/코드 해석은 정의형 질문 1개에 답하는 FAQ와 형태가 달라서, 그 페르소나(특히 문장 수 제한)를 씌우면
# 실측 검증(2026-07-21)에서 실제로 LLM이 "요약해달라"는 정상 요청까지 필요 이상으로 위축돼 근거가 충분한데도
# "확인 불가"로 회피하는 사례가 관찰됐다. 안전 원칙(근거 밖 내용 금지, 모르면 인정)은 유지하되 분량 제약은 뺀다.
EXTRACT_SYSTEM_PROMPT = """당신은 강의자료 하나의 내용을 요약·예시 제시·해석해주는 보조원입니다.
아래 근거(해당 강의자료 검색 결과)에 실제로 있는 내용만 사용해 답하십시오. 근거에 없는 예시·코드·설명·수치를
새로 지어내지 마십시오. 근거만으로 충분히 답할 수 있는 요청이라면 회피하지 말고 성실하게 답하십시오 — 요약은
여러 근거 조각을 하나의 글로 자연스럽게 엮는 것이지, 근거에 없는 내용을 추가하는 것이 아닙니다. 근거가 요청과
무관하거나 근거만으로 답할 수 없을 때만 정직하게 "확인 불가"로 안내하십시오.
"""

EXTRACT_ANSWER_PROMPT = """아래 근거(해당 강의자료 검색 결과)에 실제로 있는 내용만 사용해 사용자의 요청에 답하세요.
근거에 없는 예시·코드·설명을 새로 만들어 덧붙이지 마세요. 사용자가 "이 코드"/"방금 답변"처럼 직전 답변을
가리키면, 아래 "이전 답변"에 있는 내용을 대상으로 삼되 그 해석도 근거에 있는 설명만 사용하세요.
마지막 문장은 반드시 [출처: 파일명 p.페이지] 형식으로 끝내세요.

이전 답변: {prior_answer}

사용자 요청: {question}

근거:
{evidence}
"""


def _format_evidence(evidence: list[dict]) -> str:
    return "\n".join(f"- [{e['source']} p.{e['page']}] {e['text'][:300]}" for e in evidence) or "(없음)"


def lecture_agent(state: WizardState, config: RunnableConfig) -> Command[Literal["guardrail"]]:
    thread_id = config["configurable"]["thread_id"]
    question = last_human_text(state)
    extract_source = state.get("extract_source")

    evidence = hybrid_search(question, source_filter=extract_source)
    log_event(
        thread_id, "lecture_agent", "search",
        {"query": question, "source_filter": extract_source, "hits": len(evidence)},
    )

    if not evidence:
        return Command(
            goto="guardrail",
            update={
                "trace": ["lecture_agent: 근거 없음 -> 정직한 안내"],
                "last_error": {"type": "no_evidence", "retryable": False},
                "messages": [AIMessage(content=OUT_OF_CONTEXT_MESSAGE)],
            },
        )

    # 새 진입(아직 문서가 안 잠김)이면 이번 검색의 최상위 근거 출처로 이후 턴의 검색 범위를 고정한다.
    resolved_source = extract_source or evidence[0]["source"]

    prior_answer = last_ai_text(state) or "(없음)"
    answer = get_llm().invoke(
        [
            SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
            HumanMessage(
                content=EXTRACT_ANSWER_PROMPT.format(
                    prior_answer=prior_answer, question=question, evidence=_format_evidence(evidence)
                )
            ),
        ]
    ).content

    return Command(
        goto="guardrail",
        update={
            "evidence": evidence,
            "extract_source": resolved_source,
            "trace": [f"lecture_agent: query='{question}' source={resolved_source} -> {len(evidence)}건"],
            "last_error": None,
            "messages": [AIMessage(content=answer)],
        },
    )
