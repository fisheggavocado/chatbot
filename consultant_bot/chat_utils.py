# 여러 노드가 공유하는 자잘한 메시지 조회 헬퍼.

from langchain_core.messages import AIMessage, HumanMessage


def last_human_text(state: dict) -> str:
    """state.messages에서 가장 최근 사용자 발화를 찾는다 (AIMessage가 뒤에 섞여 있어도 정확히 사람 메시지만 찾음)."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def last_ai_text(state: dict) -> str:
    """state.messages에서 가장 최근 챗봇 답변을 찾는다.

    lecture_agent(B형)가 "이 코드를 해석해줘" 같은, 직전 자기 답변을 가리키는 후속 질문을 처리할 때
    그 답변 내용을 프롬프트에 넣기 위해 쓴다.
    """
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""
