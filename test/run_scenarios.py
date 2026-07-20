# test/output_test/ 인덱스(embed_one_pdf.py로 생성)를 대상으로 consultant_bot 멀티에이전트 그래프를
# 자동으로 세 시나리오(FAQ / out_of_scope / design 위저드)로 돌려보고 결과를 콘솔에 출력한다.
# 실제 OpenAI API를 호출하므로 비용이 발생한다.
#
# design 시나리오는 위저드가 제시하는 선택지 중 "첫 번째"를 그대로 골라 자동 진행한다 (PDF 내용에 의존하지 않음).
#
# 사용법 (first_project/ 에서 실행, 먼저 python test/embed_one_pdf.py 로 인덱스를 만들어둬야 함):
#   python test/run_scenarios.py
#   python test/run_scenarios.py --only faq --faq-question "임베딩이 뭐야?"
#   python test/run_scenarios.py --only design --design-question "..."

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

# Windows 콘솔의 기본 코드페이지(cp949)는 em-dash 등 일부 유니코드 문자를 인코딩하지 못해 print()가
# 죽을 수 있다. LLM이 생성하는 답변에는 이런 문자가 흔히 섞여 나오므로, 출력 인코딩을 UTF-8로 강제한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
TEST_OUTPUT_DIR = TEST_DIR / "output_test"

if not (TEST_OUTPUT_DIR / "docstore.json").exists():
    raise SystemExit(f"'{TEST_OUTPUT_DIR}'에 인덱스가 없습니다. 먼저 `python test/embed_one_pdf.py`를 실행하세요.")

# consultant_bot이 이 테스트 인덱스/체크포인트/캐시/트레이스를 보도록 OUTPUT_DIR을 먼저 바꿔치기한다.
# config.py가 import 시점에 os.getenv("OUTPUT_DIR")를 읽으므로, consultant_bot 모듈을 import하기 전에 설정해야 한다.
os.environ["OUTPUT_DIR"] = str(TEST_OUTPUT_DIR)

sys.path.insert(0, str(PROJECT_DIR / "consultant_bot"))
sys.path.insert(0, str(PROJECT_DIR))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from graph import get_graph, make_run_config  # noqa: E402


def _run_turn(graph, config, text: str = None, resume=None):
    if resume is not None:
        return graph.invoke(Command(resume=resume), config=config)
    return graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)


def _pick_resume(stage: str, payload: dict):
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


def scenario_faq(question: str) -> None:
    print("\n" + "#" * 70)
    print(f"[시나리오: FAQ] 질문: {question}")
    print("#" * 70)
    graph = get_graph()
    thread_id = f"test-faq-{uuid4()}"
    config = make_run_config(thread_id)
    result = _run_turn(graph, config, text=question)

    intent = graph.get_state(config).values.get("intent")
    print(f"\n[결과] intent={intent} (faq 기대)")
    messages = result.get("messages", [])
    if messages:
        print(f"[답변] {messages[-1].content}")
    print(f"[trace] {TEST_OUTPUT_DIR}/traces/{thread_id}.jsonl 에서 상세 로그 확인 가능")


def scenario_out_of_scope(question: str) -> None:
    print("\n" + "#" * 70)
    print(f"[시나리오: out_of_scope] 질문: {question}")
    print("#" * 70)
    graph = get_graph()
    thread_id = f"test-oos-{uuid4()}"
    config = make_run_config(thread_id)
    result = _run_turn(graph, config, text=question)

    intent = graph.get_state(config).values.get("intent")
    print(f"\n[결과] intent={intent} (out_of_scope 기대, 검색 0회여야 함)")
    messages = result.get("messages", [])
    if messages:
        print(f"[답변] {messages[-1].content}")


def scenario_design_wizard(question: str, max_turns: int = 3) -> None:
    print("\n" + "#" * 70)
    print(f"[시나리오: design 위저드] 질문: {question}")
    print("#" * 70)
    graph = get_graph()
    thread_id = f"test-design-{uuid4()}"
    config = make_run_config(thread_id)

    result = _run_turn(graph, config, text=question)
    turns_seen = 0
    while "__interrupt__" in result and turns_seen < max_turns:
        turns_seen += 1
        interrupt_obj = result["__interrupt__"][0]
        stage = graph.get_state(config).values.get("stage")
        print(f"\n[{turns_seen}턴] stage={stage}")
        print(f"  presenter 출력: {interrupt_obj.value}")
        resume_value = _pick_resume(stage, interrupt_obj.value)
        print(f"  자동 선택(첫 옵션): {resume_value}")
        result = _run_turn(graph, config, resume=resume_value)

    final_stage = graph.get_state(config).values.get("stage")
    print(f"\n[결과] 총 {turns_seen}턴 진행 (interrupt 발생), 최종 stage={final_stage}")
    messages = result.get("messages", [])
    if messages:
        print(f"[최종 메시지] {messages[-1].content}")
    print(f"[trace] {TEST_OUTPUT_DIR}/traces/{thread_id}.jsonl 에서 상세 로그 확인 가능")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="test/output_test/ 인덱스로 consultant_bot 멀티에이전트 시나리오 테스트"
    )
    parser.add_argument("--faq-question", default="RAG가 뭐야?")
    parser.add_argument("--design-question", default="AI 챗봇 파이프라인을 어떻게 설계하면 좋을지 추천해줘")
    parser.add_argument("--oos-question", default="오늘 저녁 메뉴 추천해줘")
    parser.add_argument(
        "--only", choices=["faq", "out_of_scope", "design"], default=None, help="지정하면 해당 시나리오만 실행"
    )
    args = parser.parse_args()

    if args.only in (None, "faq"):
        scenario_faq(args.faq_question)
    if args.only in (None, "out_of_scope"):
        scenario_out_of_scope(args.oos_question)
    if args.only in (None, "design"):
        scenario_design_wizard(args.design_question)

    print("\n" + "=" * 70)
    print("모든 시나리오 실행 완료.")
    print("=" * 70)


if __name__ == "__main__":
    main()
