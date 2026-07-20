# FAQ 챗봇 회귀 테스트 — llm.SYSTEM_PROMPT가 강제하는 출력 가드레일이 실제로 지켜지는지
# 케이스 x 이슈 매트릭스로 검증한다. 프롬프트 문구를 한 글자라도 바꾸면 이 스크립트를 전체로 다시
# 돌려 "어느 칸이 빨개졌는지" 확인한다 (한 케이스만 보고 통과시키지 않는다).
#
# 이슈 3종 (모든 케이스에 공통 적용):
#   no_hedge   : 답변에 금지된 모호 표현("제 생각에는"/"아마도"/"확실하지 않지만")이 없는가
#   honesty    : 근거가 없으면 llm.OUT_OF_CONTEXT_MESSAGE 문구 그대로 정직하게 인정하는가
#                (지어내지 않는가) — 근거가 있으면 [출처: ...] 인용이 붙는가
#   routing    : coordinator가 intent를 기대한 대로 분류했는가 (faq/out_of_scope)
#
# 케이스 3개:
#   in_scope_answer   : 실제 hybrid_search를 태워 도메인 질문에 답한다 (test PDF가 매번 다르므로
#                        "정답 내용"은 검증하지 않고, 근거 있으면 출처 인용/없으면 정직한 인정만 검증)
#   out_of_scope       : 도메인 밖 질문 -> coordinator가 검색 0회로 즉시 안내하는지
#   no_evidence_honest : react_loop.hybrid_search를 빈 결과로 monkeypatch해 "근거가 전혀 없을 때"를
#                        PDF 내용과 무관하게 결정적으로 재현 -> guardrail까지 통과해 정직한 문구가
#                        그대로 나오는지 (테스트 인덱스가 뭐가 뽑히든 항상 같은 조건으로 검증 가능)
#
# 실제 OpenAI API를 호출하므로 비용이 발생한다 (in_scope_answer/out_of_scope/no_evidence_honest 각 1회씩).
#
# 사용법 (first_project/ 에서 실행, 먼저 python test/embed_one_pdf.py 로 인덱스를 만들어둬야 함):
#   python test/regression_faq.py
#   python test/regression_faq.py --update-baseline   # 이번 결과를 새 기준선으로 저장(의도적 변경 검토 후)

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_DIR = Path(__file__).resolve().parent
BASELINE_PATH = TEST_DIR / "regression_baseline.json"

# run_scenarios.py가 이미 OUTPUT_DIR 바꿔치기 + sys.path 설정 + graph 모듈 import를 끝낸 상태로
# 노출해주므로 그대로 재사용한다 (설정 중복 방지, import 시점에 인덱스 존재 여부도 같이 검증됨).
sys.path.insert(0, str(TEST_DIR))
import run_scenarios as rs  # noqa: E402

from langchain_core.messages import HumanMessage  # noqa: E402

import react_loop  # noqa: E402
from llm import FORBIDDEN_HEDGE_PHRASES, OUT_OF_CONTEXT_MESSAGE  # noqa: E402


def _last_answer(result: dict) -> str:
    messages = result.get("messages", [])
    return str(messages[-1].content) if messages else ""


def _check_no_hedge(answer: str) -> tuple[bool, str]:
    hit = next((p for p in FORBIDDEN_HEDGE_PHRASES if p in answer), None)
    if hit:
        return False, f"금지 표현 '{hit}' 검출"
    return True, "금지 표현 없음"


def _check_honesty(answer: str) -> tuple[bool, str]:
    # evidence 리스트의 "존재 여부"가 아니라 최종 답변 자체의 두 형태(정직한 안내 문구 vs 실질 답변)만으로
    # 판정한다 — react_loop가 검색은 했지만(evidence 非-empty) 질문에 답하기엔 불충분하다고 판단해
    # 정직한 안내로 끝나는 경우가 정상 케이스이므로, evidence 유무를 기준으로 삼으면 오탐이 난다.
    if answer.strip() == OUT_OF_CONTEXT_MESSAGE:
        return True, "정직한 안내 문구 정확히 일치"
    if "[출처:" in answer:
        return True, "실질 답변 -> [출처: ...] 인용 포함"
    return False, f"실질 답변인데 [출처: ...] 인용이 없음(근거 없이 단정했을 가능성): {answer[:80]!r}"


def case_in_scope_answer(question: str) -> dict:
    graph = rs.get_graph()
    thread_id = f"test-regress-inscope-{uuid4()}"
    config = rs.make_run_config(thread_id)
    result = rs._run_turn(graph, config, text=question)

    values = graph.get_state(config).values
    answer = _last_answer(result)
    checks = {
        "routing": (values.get("intent") == "faq", f"intent={values.get('intent')} (faq 기대)"),
        "no_hedge": _check_no_hedge(answer),
        "honesty": _check_honesty(answer),
    }
    return {"id": "in_scope_answer", "question": question, "answer": answer, "checks": checks}


def case_out_of_scope(question: str) -> dict:
    graph = rs.get_graph()
    thread_id = f"test-regress-oos-{uuid4()}"
    config = rs.make_run_config(thread_id)
    result = rs._run_turn(graph, config, text=question)

    values = graph.get_state(config).values
    answer = _last_answer(result)
    checks = {
        "routing": (values.get("intent") == "out_of_scope", f"intent={values.get('intent')} (out_of_scope 기대)"),
        "no_hedge": _check_no_hedge(answer),
        "no_search": (
            not values.get("evidence"),
            f"evidence={len(values.get('evidence', []))}건 (0건 기대, 검색을 아예 타지 않아야 함)",
        ),
    }
    return {"id": "out_of_scope", "question": question, "answer": answer, "checks": checks}


def case_no_evidence_honest(question: str) -> dict:
    """react_loop.hybrid_search를 빈 결과로 강제해, PDF 내용과 무관하게 '근거 전혀 없음'을 재현한다."""
    original_search = react_loop.hybrid_search
    react_loop.hybrid_search = lambda query: []
    try:
        graph = rs.get_graph()
        thread_id = f"test-regress-noev-{uuid4()}"
        config = rs.make_run_config(thread_id)
        result = rs._run_turn(graph, config, text=question)
        values = graph.get_state(config).values
    finally:
        react_loop.hybrid_search = original_search

    answer = _last_answer(result)
    checks = {
        "routing": (values.get("intent") == "faq", f"intent={values.get('intent')} (faq 기대 - 도메인 키워드 포함 질문)"),
        "no_hedge": _check_no_hedge(answer),
        "honesty": (
            answer.strip() == OUT_OF_CONTEXT_MESSAGE,
            f"강제 무근거 상황에서 정직한 안내 문구 일치 여부: {answer[:80]!r}",
        ),
    }
    return {"id": "no_evidence_honest", "question": question, "answer": answer, "checks": checks}


def _print_matrix(cases: list[dict]) -> dict:
    print("\n" + "=" * 70)
    print("케이스 x 이슈 매트릭스")
    print("=" * 70)
    results: dict = {}
    for case in cases:
        case_passed = True
        print(f"\n[{case['id']}] 질문: {case['question']}")
        print(f"  답변: {case['answer'][:200]}")
        for issue, (passed, detail) in case["checks"].items():
            mark = "PASS" if passed else "FAIL"
            print(f"  [{issue:10s}] {mark} - {detail}")
            case_passed = case_passed and passed
        results[case["id"]] = case_passed
        print(f"  => 케이스 종합: {'PASS' if case_passed else 'FAIL'}")
    return results


def _gate_against_baseline(results: dict, update_baseline: bool) -> int:
    if update_baseline or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n기준선을 '{BASELINE_PATH}'에 저장했습니다: {results}")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    regressions = [
        case_id for case_id, was_passing in baseline.items()
        if was_passing and not results.get(case_id, False)
    ]
    print("\n" + "=" * 70)
    if regressions:
        print(f"회귀 발견! 이전에 통과하던 케이스가 이번엔 실패함: {regressions}")
        print("프롬프트/코드 변경이 의도한 것이면 --update-baseline으로 기준선을 갱신하세요.")
        print("=" * 70)
        return 1
    print("회귀 없음 (기준선 대비 실패로 전환된 케이스가 없습니다).")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FAQ 페르소나 시스템 프롬프트 회귀 테스트")
    parser.add_argument("--in-scope-question", default="RAG가 뭐야?")
    parser.add_argument("--oos-question", default="오늘 저녁 메뉴 추천해줘")
    parser.add_argument("--no-evidence-question", default="LLM 캐싱 전략이 뭐야?")
    parser.add_argument("--update-baseline", action="store_true", help="이번 결과를 새 기준선으로 저장")
    args = parser.parse_args()

    cases = [
        case_in_scope_answer(args.in_scope_question),
        case_out_of_scope(args.oos_question),
        case_no_evidence_honest(args.no_evidence_question),
    ]
    results = _print_matrix(cases)
    return _gate_against_baseline(results, args.update_baseline)


if __name__ == "__main__":
    sys.exit(main())
