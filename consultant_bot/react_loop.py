# 공유 bounded-ReAct 엔진 (설계안 4번: "두 곳 모두 강의의 Bounded ReAct Loop 규율을 따릅니다")
#
# ReAct의 전형적 실패 모드 — "한 번 잘못된 생각에 갇히면 그 안에서 계속 맴돈다" — 를
# 프롬프트 지시가 아니라 코드로 강제로 끊기 위해, LangGraph의 블랙박스 create_react_agent 대신
# Thought(next_query) -> Action(hybrid_search) -> Observation -> Reflect(충분성 판단) 루프를 직접 구현한다.
#
# 3중 정지 조건 (LLM의 판단과 무관하게 코드가 우선 적용):
#   1) budget_exhausted  - state.budget_remaining 소진
#   2) duplicate_action  - 정규화된 쿼리를 반복 (같은 생각을 다시 함)
#   3) no_progress       - 2턴 연속 새로운 출처가 하나도 안 나옴 (표현만 바꿔 제자리를 맴돎)
#
# 위 조건으로 멈추면 예외를 던지지 않고 stopped_reason과 함께 "가능한 만큼의 근거"를 반환한다.
# 상위 노드는 이를 보고 확정 답변 대신 "자료에서 확인 불가/제한적 근거"로 정직하게 응답한다 (Error as State 원칙).

from dataclasses import dataclass, field
from typing import Optional

from llm import get_structured_llm
from observability import log_event
from retrieval import hybrid_search
from state import Evidence, ReflectDecision

REFLECT_PROMPT = """당신은 강의 자료 검색 에이전트입니다. 아래는 원 질문과, 지금까지 검색으로 모은 근거입니다.
이 근거만으로 원 질문에 출처를 밝히며 답할 수 있으면 sufficient=true로 답하세요.
부족하다면 sufficient=false로 답하고, 이전과는 다른 각도의 next_query를 제안하세요 (완전히 같은 검색어는 금지).

원 질문: {question}

지금까지 모은 근거:
{evidence_summary}
"""

# --- 품질/폭주 제어 튜닝 파라미터 (필요에 따라 조정) ---
NO_PROGRESS_LIMIT = 2  # 연속 몇 턴 "신규 출처 0개"면 정체로 보고 조기 종료할지. 낮출수록 보수적(빨리 포기),
# 높일수록 더 끈질기게 재검색을 시도함 (그만큼 LLM 호출 비용 증가)

# 규칙 우선 sufficient 판정 (coordinator.py의 "규칙 우선, 애매할 때만 LLM" 패턴과 동일) —
# 이번 라운드 근거의 리랭커 점수가 이미 충분히 높으면 reflect LLM 호출 없이 즉시 종료한다.
# FAQ(MAX_STEPS=2)는 첫 검색이 이 조건을 만족하는 흔한 경우 사실상 1라운드(LLM 호출 0회)로 끝나고,
# 점수가 낮거나 애매한 경우에만 기존처럼 reflect_llm으로 폴백한다 — 루프 자체는 유지해 안전망은 보존한다.
RULE_SUFFICIENT_SCORE = 0.5  # bge-reranker-v2-m3 normalize=True 점수(0~1) 임계값. 낮출수록 LLM 호출↓ 재현율 리스크↑
RULE_MIN_MATCHES = 2  # 이 개수만큼의 근거가 임계값을 넘어야 규칙 통과 (근거 1건만으로 확정하지 않음)


def _rule_sufficient(batch: list[Evidence]) -> bool:
    """이번 라운드에서 새로 얻은 근거의 리랭커 점수가 충분히 높으면 규칙만으로 sufficient로 판단한다."""
    top_scores = sorted((e["score"] for e in batch), reverse=True)[:RULE_MIN_MATCHES]
    return len(top_scores) >= RULE_MIN_MATCHES and all(s >= RULE_SUFFICIENT_SCORE for s in top_scores)


@dataclass
class ReactResult:
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    stopped_reason: Optional[str] = None  # None이면 sufficient=True로 정상 종료
    sufficient: bool = False
    budget_remaining: int = 0
    steps_used: int = 0


def _normalize(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _evidence_summary(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(아직 없음)"
    lines = [f"- [{e['source']} p.{e['page']}] {e['text'][:200]}" for e in evidence]
    return "\n".join(lines)


def run_bounded_react(
    node_name: str,
    thread_id: str,
    question: str,
    max_steps: int,
    budget_remaining: int,
) -> ReactResult:
    """bounded ReAct 루프를 실행하고 ReactResult를 반환한다. 예외를 던지지 않는다."""
    reflect_llm = get_structured_llm(ReflectDecision)

    query = question
    seen_queries: set[str] = set()
    seen_sources: set[tuple] = set()
    all_evidence: list[Evidence] = []
    trace: list[str] = []
    consecutive_no_progress = 0
    stopped_reason: Optional[str] = None
    sufficient = False
    steps_used = 0

    for step in range(1, max_steps + 1):
        if budget_remaining <= 0:
            stopped_reason = "budget_exhausted"
            break

        normalized = _normalize(query)
        if normalized in seen_queries:
            stopped_reason = "duplicate_action"
            log_event(thread_id, node_name, "react_stop", {"reason": stopped_reason, "query": query})
            break
        seen_queries.add(normalized)

        batch = hybrid_search(query)
        budget_remaining -= 1
        steps_used = step

        batch_sources = {(e["source"], e["page"]) for e in batch}
        new_sources = batch_sources - seen_sources
        seen_sources |= batch_sources
        all_evidence.extend(batch)

        step_log = (
            f"{node_name} step{step}: query='{query}' -> {len(batch)}건 "
            f"({len(new_sources)}개 신규 출처)"
        )
        trace.append(step_log)
        log_event(
            thread_id,
            node_name,
            "react_step",
            {
                "step": step,
                "query": query,
                "hits": len(batch),
                "new_sources": len(new_sources),
                # 크래시 시에도 evidence 자체를 traces/{thread_id}.jsonl에서 복구할 수 있도록 전체 근거를 남긴다
                # (그래프 노드가 끝나기 전에 죽으면 SqliteSaver 체크포인트엔 아직 안 남기 때문).
                "evidence": [dict(e) for e in batch],
            },
        )

        if new_sources:
            consecutive_no_progress = 0
        else:
            consecutive_no_progress += 1
            if consecutive_no_progress >= NO_PROGRESS_LIMIT:
                stopped_reason = "no_progress"
                log_event(thread_id, node_name, "react_stop", {"reason": stopped_reason, "query": query})
                break

        if _rule_sufficient(batch):
            sufficient = True
            log_event(
                thread_id,
                node_name,
                "react_rule_sufficient",
                {"step": step, "top_scores": sorted((e["score"] for e in batch), reverse=True)[:RULE_MIN_MATCHES]},
            )
            break

        reflect: ReflectDecision = reflect_llm.invoke(
            REFLECT_PROMPT.format(question=question, evidence_summary=_evidence_summary(all_evidence))
        )
        if reflect.sufficient:
            sufficient = True
            break
        query = reflect.next_query or query

    if not sufficient and stopped_reason is None:
        # max_steps를 다 썼는데도 sufficient 판정을 못 받은 경우
        stopped_reason = "budget_exhausted"
        log_event(thread_id, node_name, "react_stop", {"reason": stopped_reason, "query": query})

    if sufficient:
        log_event(thread_id, node_name, "react_done", {"steps_used": steps_used, "evidence": len(all_evidence)})

    return ReactResult(
        evidence=all_evidence,
        trace=trace,
        stopped_reason=None if sufficient else stopped_reason,
        sufficient=sufficient,
        budget_remaining=budget_remaining,
        steps_used=steps_used,
    )
