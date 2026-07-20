# 여러 노드가 공유하는 자잘한 메시지 조회 헬퍼.

from langchain_core.messages import HumanMessage


def last_human_text(state: dict) -> str:
    """state.messages에서 가장 최근 사용자 발화를 찾는다 (AIMessage가 뒤에 섞여 있어도 정확히 사람 메시지만 찾음)."""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""
