# consultant_bot E2E + LLM-as-Judge 평가 하네스.
#
# ../../test/run_scenarios.py·regression_faq.py가 "배관이 실제로 동작하는가"(intent 라우팅, interrupt
# 흐름, 금지 표현 여부)를 확인하는 것과 달리, 이 하네스는 cases.py의 시나리오를 실제 그래프로 끝까지
# 실행한 뒤 judge.py(LLM-as-Judge)로 "답변 품질이 쓸만한가"까지 채점하고, 이전 실행(기준선) 대비
# 회귀를 잡는다. observability.py가 남긴 OUTPUT_DIR/traces/{thread_id}.jsonl도 함께 읽어 궤적 지표
# (검색 스텝 수, react_loop 정지 사유, guardrail 재시도 횟수)를 리포트에 포함시킨다.
#
# 실제 OpenAI API를 호출하므로 비용이 발생한다 (케이스마다 그래프 실행 1회 + judge 호출 1회).
#
# 사용법 (first_project/ 에서 실행, 먼저 python test/embed_one_pdf.py 로 인덱스를 만들어둬야 함):
#   python consultant_bot/eval/run_eval.py
#   python consultant_bot/eval/run_eval.py --output-dir "D:/.../output" --min-score 3.5
#   python consultant_bot/eval/run_eval.py --only faq
#   python consultant_bot/eval/run_eval.py --update-baseline   # 이번 결과를 새 기준선으로 저장

import os

# torch/BGE-M3/리랭커/kiwipiepy가 각자 자체 OpenMP 런타임(libiomp5md.dll 등)을 들고 있어, 같은 프로세스에서
# 전부 로드되면(Windows에서 특히) "OMP: Error #15" 세그멘테이션 폴트로 죽는다(app.py/server.py와 동일 이슈).
# graph.py를 import하기 전에 반드시 설정해야 해서 다른 import보다 앞에 둔다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from uuid import uuid4  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EVAL_DIR = Path(__file__).resolve().parent
CONSULTANT_BOT_DIR = EVAL_DIR.parent
PROJECT_DIR = CONSULTANT_BOT_DIR.parent
DEFAULT_TEST_OUTPUT_DIR = PROJECT_DIR / "test" / "output_test"
BASELINE_PATH = EVAL_DIR / "eval_baseline.json"


def _resolve_output_dir(cli_value: str | None) -> Path:
    if cli_value:
        path = Path(cli_value)
    elif DEFAULT_TEST_OUTPUT_DIR.exists():
        path = DEFAULT_TEST_OUTPUT_DIR
    else:
        raise SystemExit(
            f"'{DEFAULT_TEST_OUTPUT_DIR}'에 인덱스가 없습니다. 먼저 `python test/embed_one_pdf.py`를 실행하거나 "
            "--output-dir로 인덱스가 있는 OUTPUT_DIR을 지정하세요."
        )
    if not (path / "docstore.json").exists():
        raise SystemExit(f"'{path}'에 인덱스(docstore.json)가 없습니다.")
    return path


def _trace_summary(output_dir: Path, thread_id: str) -> dict:
    """observability.log_event()가 남긴 JSONL에서 궤적 지표(검색 스텝 수/정지 사유/guardrail 재시도)를 뽑는다."""
    trace_path = output_dir / "traces" / f"{thread_id}.jsonl"
    summary = {"search_steps": 0, "stopped_reason": None, "guardrail_fail_count": 0, "cache_hit": False}
    if not trace_path.exists():
        return summary
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            event = record.get("event")
            if event == "react_step":
                summary["search_steps"] += 1
            elif event == "react_stop":
                summary["stopped_reason"] = record.get("payload", {}).get("reason")
            elif event == "verdict" and not record.get("payload", {}).get("passed", True):
                summary["guardrail_fail_count"] += 1
            elif event == "cache_hit":
                summary["cache_hit"] = True
    return summary


def _run_faq_or_oos_case(case: dict, harness, output_dir: Path) -> dict:
    graph = harness.get_graph()
    thread_id = f"eval-{case['id']}-{uuid4()}"
    config = harness.make_run_config(thread_id)
    result = harness.run_turn(graph, config, text=case["question"])

    values = graph.get_state(config).values
    messages = result.get("messages", [])
    answer = str(messages[-1].content) if messages else ""
    return {
        "id": case["id"],
        "question": case["question"],
        "intent": values.get("intent"),
        "answer": answer,
        "evidence": values.get("evidence", []),
        "turns": 1,
        "trace": _trace_summary(output_dir, thread_id),
    }


def _run_extract_case(case: dict, harness, output_dir: Path) -> dict:
    """B형(extract)은 interrupt() 없는 일반 대화 턴이라, design과 달리 위저드 재개 루프가 필요 없다 —
    case['questions']에 든 턴들을 그대로 순차 invoke하면 된다(coordinator의 sticky 라우팅이 이어준다)."""
    graph = harness.get_graph()
    thread_id = f"eval-{case['id']}-{uuid4()}"
    config = harness.make_run_config(thread_id)

    result = None
    for question in case["questions"]:
        result = harness.run_turn(graph, config, text=question)

    values = graph.get_state(config).values
    messages = result.get("messages", [])
    answer = str(messages[-1].content) if messages else ""
    return {
        "id": case["id"],
        "question": case["questions"][-1],
        "intent": values.get("intent"),
        "answer": answer,
        "evidence": values.get("evidence", []),
        "turns": len(case["questions"]),
        "trace": _trace_summary(output_dir, thread_id),
    }


def _run_design_case(case: dict, harness, output_dir: Path, max_turns: int = 3) -> dict:
    graph = harness.get_graph()
    thread_id = f"eval-{case['id']}-{uuid4()}"
    config = harness.make_run_config(thread_id)

    result = harness.run_turn(graph, config, text=case["question"])
    turns = 1
    while "__interrupt__" in result and turns <= max_turns:
        interrupt_obj = result["__interrupt__"][0]
        stage = graph.get_state(config).values.get("stage")
        resume_value = harness.pick_resume(stage, interrupt_obj.value)
        result = harness.run_turn(graph, config, resume=resume_value)
        turns += 1

    values = graph.get_state(config).values
    messages = result.get("messages", [])
    answer = str(messages[-1].content) if messages else ""
    return {
        "id": case["id"],
        "question": case["question"],
        "intent": values.get("intent"),
        "answer": answer,
        "evidence": values.get("evidence", []),
        "turns": turns,
        "final_stage": values.get("stage"),
        "trace": _trace_summary(output_dir, thread_id),
    }


def _deterministic_checks(case: dict, run: dict, forbidden_hedge, out_of_context_message) -> dict:
    checks = {}
    checks["routing"] = (
        run["intent"] == case["expected_intent"],
        f"intent={run['intent']} (기대: {case['expected_intent']})",
    )
    hit = next((p for p in forbidden_hedge if p in run["answer"]), None)
    checks["no_hedge"] = (hit is None, f"금지 표현 '{hit}' 검출" if hit else "금지 표현 없음")

    if case["expected_intent"] == "out_of_scope":
        checks["no_search"] = (
            not run["evidence"],
            f"evidence={len(run['evidence'])}건 (0건 기대, 검색을 아예 타지 않아야 함)",
        )
    elif case["expected_intent"] in ("faq", "extract"):
        if not run["evidence"]:
            checks["honesty"] = (
                run["answer"].strip() == out_of_context_message,
                "근거 없음 -> 정직한 안내 문구 일치 여부",
            )
        else:
            checks["citation"] = ("[출처:" in run["answer"], "근거 있음 -> [출처: ...] 인용 포함 여부")
    return checks


def _print_and_collect(cases_run: list[dict]) -> dict:
    print("\n" + "=" * 78)
    print("케이스별 결과 (결정적 체크 + LLM-as-Judge)")
    print("=" * 78)
    results = {}
    scores = []
    for run in cases_run:
        case_passed = all(passed for passed, _ in run["checks"].values())
        judge = run["judge"]
        judge_passed = judge.faithful and judge.relevant
        scores.append(judge.score)

        print(f"\n[{run['id']}] 질문: {run['question']}")
        print(f"  답변: {run['answer'][:200]}")
        for issue, (passed, detail) in run["checks"].items():
            print(f"  [{issue:10s}] {'PASS' if passed else 'FAIL'} - {detail}")
        print(
            f"  [judge] score={judge.score}/5 faithful={judge.faithful} relevant={judge.relevant} - {judge.reason}"
        )
        trace = run["trace"]
        print(
            f"  [trace] 검색 스텝={trace['search_steps']} 정지사유={trace['stopped_reason']} "
            f"guardrail 재시도={trace['guardrail_fail_count']} 캐시히트={trace['cache_hit']}"
        )
        overall = case_passed and judge_passed
        print(f"  => 종합: {'PASS' if overall else 'FAIL'}")
        results[run["id"]] = overall

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print("\n" + "-" * 78)
    print(f"평균 judge 점수: {avg_score:.2f}/5 ({len(scores)}개 케이스)")
    return {"results": results, "avg_score": avg_score}


def _gate(summary: dict, min_score: float, update_baseline: bool) -> int:
    results, avg_score = summary["results"], summary["avg_score"]
    exit_code = 0

    if avg_score < min_score:
        print(f"\n[실패] 평균 judge 점수 {avg_score:.2f} < 기준 {min_score}")
        exit_code = 1

    if update_baseline or not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"기준선을 '{BASELINE_PATH}'에 저장했습니다: {results}")
    else:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        regressions = [cid for cid, was_passing in baseline.items() if was_passing and not results.get(cid, False)]
        if regressions:
            print(f"\n[실패] 회귀 발견 - 이전에 통과하던 케이스가 이번엔 실패함: {regressions}")
            print("의도한 변경이면 --update-baseline으로 기준선을 갱신하세요.")
            exit_code = 1
        else:
            print("\n회귀 없음 (기준선 대비 실패로 전환된 케이스가 없습니다).")

    print("=" * 78)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="consultant_bot E2E + LLM-as-Judge 평가 하네스")
    parser.add_argument("--output-dir", default=None, help="인덱스가 있는 OUTPUT_DIR (기본: test/output_test)")
    parser.add_argument("--min-score", type=float, default=3.0, help="judge 평균 점수(1~5) 최저 통과선")
    parser.add_argument("--update-baseline", action="store_true", help="이번 결과를 새 기준선으로 저장")
    parser.add_argument(
        "--only", choices=["faq", "out_of_scope", "design", "extract"], default=None, help="해당 종류만 실행"
    )
    args = parser.parse_args()

    output_dir = _resolve_output_dir(args.output_dir)
    os.environ["OUTPUT_DIR"] = str(output_dir)  # config.py가 import 시점에 읽으므로 graph 등을 import하기 전에 설정

    sys.path.insert(0, str(CONSULTANT_BOT_DIR))
    sys.path.insert(0, str(PROJECT_DIR))

    # test/run_scenarios.py를 import해서 재사용하지 않는다 - 그 모듈은 자체적으로 test/output_test로
    # OUTPUT_DIR을 덮어쓰므로 --output-dir로 다른 인덱스를 지정해도 무시돼 버린다. 대신 graph.py를
    # 직접 import하고, _run_turn/_pick_resume 같은 최소 헬퍼만 이 파일 안에 인라인으로 둔다.
    from graph import get_graph, make_run_config  # noqa: E402
    from langchain_core.messages import HumanMessage  # noqa: E402
    from langgraph.types import Command  # noqa: E402
    from llm import FORBIDDEN_HEDGE_PHRASES, OUT_OF_CONTEXT_MESSAGE  # noqa: E402

    from cases import DESIGN_CASES, EXTRACT_CASES, FAQ_CASES, OUT_OF_SCOPE_CASES  # noqa: E402
    from judge import judge_answer  # noqa: E402

    def run_turn(graph, config, text: str = None, resume=None):
        if resume is not None:
            return graph.invoke(Command(resume=resume), config=config)
        return graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)

    def pick_resume(stage: str, payload: dict):
        """제시된 선택지 중 첫 번째를 자동으로 골라 위저드를 진행시킨다 (PDF 내용에 의존하지 않는 일반적 선택)."""
        if stage == "tech_select":
            rows = payload.get("rows", [])
            return [rows[0]["technology"]] if rows else []
        if stage == "pipeline_select":
            options = payload.get("options", [])
            name = options[0]["name"] if options else None
            return {"pipeline": name, "confirm": True}
        if stage == "compare":
            return {"confirm": True}
        return {}

    harness = SimpleNamespace(
        get_graph=get_graph, make_run_config=make_run_config, run_turn=run_turn, pick_resume=pick_resume
    )

    faq_cases = FAQ_CASES if args.only in (None, "faq") else []
    oos_cases = OUT_OF_SCOPE_CASES if args.only in (None, "out_of_scope") else []
    design_cases = DESIGN_CASES if args.only in (None, "design") else []
    extract_cases = EXTRACT_CASES if args.only in (None, "extract") else []

    cases_run = []
    for case in faq_cases + oos_cases:
        run = _run_faq_or_oos_case(case, harness, output_dir)
        run["checks"] = _deterministic_checks(case, run, FORBIDDEN_HEDGE_PHRASES, OUT_OF_CONTEXT_MESSAGE)
        run["judge"] = judge_answer(run["question"], run["answer"], run["evidence"])
        cases_run.append(run)

    for case in design_cases:
        run = _run_design_case(case, harness, output_dir)
        run["checks"] = _deterministic_checks(case, run, FORBIDDEN_HEDGE_PHRASES, OUT_OF_CONTEXT_MESSAGE)
        run["judge"] = judge_answer(run["question"], run["answer"], run["evidence"])
        cases_run.append(run)

    for case in extract_cases:
        run = _run_extract_case(case, harness, output_dir)
        run["checks"] = _deterministic_checks(case, run, FORBIDDEN_HEDGE_PHRASES, OUT_OF_CONTEXT_MESSAGE)
        run["judge"] = judge_answer(run["question"], run["answer"], run["evidence"])
        cases_run.append(run)

    summary = _print_and_collect(cases_run)

    report_dir = output_dir / "eval_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(
        json.dumps(
            {
                "avg_score": summary["avg_score"],
                "results": summary["results"],
                "cases": [
                    {
                        "id": r["id"],
                        "question": r["question"],
                        "answer": r["answer"],
                        "intent": r["intent"],
                        "checks": {k: v[0] for k, v in r["checks"].items()},
                        "judge": r["judge"].model_dump(),
                        "trace": r["trace"],
                    }
                    for r in cases_run
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n리포트 저장: {report_path}")

    return _gate(summary, args.min_score, args.update_baseline)


if __name__ == "__main__":
    sys.exit(main())
