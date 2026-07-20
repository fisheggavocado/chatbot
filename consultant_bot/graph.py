# StateGraph 조립. 노드들이 모두 Command(goto=...)로 스스로 라우팅하므로(LangManus/LangGraph Command 패턴),
# 여기서는 노드 등록 + 진입점 + checkpointer만 담당한다.
#
# checkpointer: SqliteSaver로 OUTPUT_DIR/consultant_bot_checkpoints.sqlite에 저장 -> thread_id 기반 durable
# persistence (설계안 3번: "위저드 도중 새로고침돼도 선택 상태가 유지·재개됨").
#
# HF 백업: 임베딩 인덱스(hf_storage.upload_output_to_hf/restore_output_from_hf, embedding/ 경로)와 같은
# 패턴으로, 이 체크포인트 sqlite 파일도 HF_REPO_ID의 checkpoints/ 경로에 백업한다 — 휘발성 컨테이너가
# 재시작돼도 대화(WizardState)가 유실되지 않도록. app.py가 매 턴 종료 후 backup_checkpoint()를 호출한다.

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.graph import START, StateGraph  # noqa: E402

from config import HF_REPO_ID, OUTPUT_DIR  # noqa: E402
from hf_storage import restore_checkpoint_from_hf, upload_checkpoint_to_hf  # noqa: E402

from coordinator import coordinator  # noqa: E402
from faq_agent import faq_agent  # noqa: E402
from guardrail import guardrail  # noqa: E402
from observability import TraceCallbackHandler  # noqa: E402
from presenter import presenter  # noqa: E402
from research_worker import research_worker  # noqa: E402
from state import WizardState  # noqa: E402
from wizard_supervisor import wizard_supervisor  # noqa: E402

RECURSION_LIMIT = 8  # 설계안 4번: 그래프 전체 recursion_limit=8 (LangManus는 5)
CHECKPOINT_DB_PATH = Path(OUTPUT_DIR) / "consultant_bot_checkpoints.sqlite"

_checkpoint_conn: sqlite3.Connection | None = None


def _build_checkpointer() -> SqliteSaver:
    global _checkpoint_conn
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_DB_PATH.exists() and HF_REPO_ID:
        if restore_checkpoint_from_hf(CHECKPOINT_DB_PATH):
            print("[graph] 이전 세션 체크포인트를 Hugging Face에서 복원했습니다.")

    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    _checkpoint_conn = conn
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return checkpointer


def backup_checkpoint() -> None:
    """현재 체크포인트 sqlite를 HF_REPO_ID의 checkpoints/ 경로에 백업한다.

    HF_REPO_ID가 없으면 조용히 건너뛴다. 업로드 실패는 대화를 막지 않도록 예외를 삼키고 로그만 남긴다
    (main.py가 PDF 처리 실패를 다루는 것과 동일한 방침).
    """
    if not HF_REPO_ID:
        return
    try:
        if _checkpoint_conn is not None:
            # WAL 모드에서 아직 -wal 사이드카에만 있는 커밋을 메인 파일로 합쳐, 업로드본이 최신 상태가 되게 한다.
            _checkpoint_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        upload_checkpoint_to_hf(CHECKPOINT_DB_PATH)
    except Exception as e:
        print(f"[체크포인트 백업 실패 - 계속 진행] {e}")


def build_graph():
    builder = StateGraph(WizardState)
    builder.add_node("coordinator", coordinator)
    builder.add_node("faq_agent", faq_agent)
    builder.add_node("wizard_supervisor", wizard_supervisor)
    builder.add_node("research_worker", research_worker)
    builder.add_node("presenter", presenter)
    builder.add_node("guardrail", guardrail)
    builder.add_edge(START, "coordinator")

    checkpointer = _build_checkpointer()
    return builder.compile(checkpointer=checkpointer)


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def make_run_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
        "callbacks": [TraceCallbackHandler(thread_id)],
        # LangSmith가 켜져 있으면(observability.LANGSMITH_ENABLED) 이 태그/메타데이터로 웹 대시보드에서
        # thread_id 기준 필터링이 가능해진다. 꺼져 있으면 LangChain이 그냥 무시하므로 조건 분기가 필요 없다.
        "tags": [f"thread:{thread_id}"],
        "metadata": {"thread_id": thread_id},
    }
