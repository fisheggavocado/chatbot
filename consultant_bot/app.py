# 터미널 대화형 실행 진입점.
# Coordinator -> FAQ Agent 또는 Wizard Supervisor(3턴, interrupt() 기반 human-in-the-loop)로 라우팅되는
# 전체 그래프를 콘솔에서 실행해본다.
#
# 사용법:
#   python app.py
#   (첫 실행 시 로컬 인덱스가 없으면 HF_REPO_ID의 embedding/ 백업에서 자동 복원을 시도한다.
#    복원할 백업도 없다면 먼저 상위 폴더에서 `python main.py`로 인덱스를 만들어야 한다.
#    대화 체크포인트(WizardState)도 같은 방식으로 HF_REPO_ID의 checkpoints/ 경로에서 자동 복원되고,
#    매 턴이 끝날 때마다 그 경로로 다시 백업된다.)

import os

# torch/BGE-M3/리랭커/kiwipiepy가 각자 자체 OpenMP 런타임(libiomp5md.dll 등)을 들고 있어, 같은 프로세스에서
# 전부 로드되면(Windows에서 특히) "OMP: Error #15" 세그멘테이션 폴트로 죽는다. graph.py를 import하기 전에
# 반드시 설정해야 해서 다른 import보다 앞에 둔다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from uuid import uuid4  # noqa: E402

# Windows 콘솔의 기본 코드페이지(cp949)는 em-dash 등 일부 유니코드 문자를 인코딩하지 못해 print()가
# 죽을 수 있다. LLM이 생성하는 답변에는 이런 문자가 흔히 섞여 나오므로, 출력 인코딩을 UTF-8로 강제한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from graph import backup_checkpoint, get_graph, make_run_config  # noqa: E402


def _print_presenter_payload(stage: str, payload: dict) -> None:
    print("\n" + "=" * 60)
    if stage == "tech_select":
        print("[1턴] 아래 기술/역할 후보 중 사용할 기술을 골라주세요.")
        for row in payload.get("rows", []):
            print(f" - {row.get('technology')} : {row.get('role')}  (출처: {row.get('source')})")
    elif stage == "pipeline_select":
        print("[2턴] 아래 파이프라인 후보를 확인하고 하나를 선택해주세요.")
        for i, opt in enumerate(payload.get("options", []), start=1):
            print(f" {i}. {opt.get('name')} (출처: {opt.get('source')})")
            for step in opt.get("steps", []):
                print(f"     - {step}")
    elif stage == "compare":
        print("[3턴] 아래 비교표를 확인해주세요.")
        for row in payload.get("rows", []):
            print(f" - {row.get('aspect')}: A={row.get('option_a')} / B={row.get('option_b')} (출처: {row.get('source')})")
    print(payload.get("prompt", ""))
    print("=" * 60)


def _read_resume(stage: str):
    if stage == "tech_select":
        raw = input("선택한 기술(쉼표로 구분) > ").strip()
        return [t.strip() for t in raw.split(",") if t.strip()]
    if stage == "pipeline_select":
        pipeline = input("선택한 파이프라인 이름 > ").strip()
        confirm = input("이 파이프라인으로 진행할까요? (y/n) > ").strip().lower().startswith("y")
        return {"pipeline": pipeline, "confirm": confirm}
    if stage == "compare":
        input("확인했으면 Enter > ")
        return {"confirm": True}
    return {}


def main() -> None:
    graph = get_graph()
    thread_id = str(uuid4())
    config = make_run_config(thread_id)
    print(f"[consultant_bot] 세션 시작 (thread_id={thread_id}). 종료하려면 'exit' 입력.")

    while True:
        text = input("\n질문 > ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if not text:
            continue

        result = graph.invoke({"messages": [HumanMessage(content=text)]}, config=config)

        while "__interrupt__" in result:
            interrupt_obj = result["__interrupt__"][0]
            snapshot = graph.get_state(config)
            stage = snapshot.values.get("stage")
            _print_presenter_payload(stage, interrupt_obj.value)
            resume_value = _read_resume(stage)
            result = graph.invoke(Command(resume=resume_value), config=config)

        final_messages = result.get("messages", [])
        if final_messages:
            print(f"\n[답변] {final_messages[-1].content}")
            download_path = final_messages[-1].additional_kwargs.get("download_path")
            if download_path:
                print(f"[문서 저장] {download_path}")

        backup_checkpoint()  # 이번 턴까지의 WizardState를 HF checkpoints/에 백업 (HF_REPO_ID 없으면 조용히 건너뜀)


if __name__ == "__main__":
    main()
