# 관찰가능성(observability) 모듈 — 외부 트레이싱 서비스 없이 파일 기반으로 "어떤 노드가 언제 무엇을 했는지" 재구성한다.
#
# 두 축으로 구성:
#   1) TraceCallbackHandler - LangChain/LangGraph의 체인·tool 호출을 자동으로 훅해 콘솔에 사람이 읽을 수 있게 출력
#   2) log_event()          - 콜백으로 안 잡히는 지점(라우팅 결정, react_loop 정지 사유, guardrail 판정,
#                              interrupt 진입/재개)까지 OUTPUT_DIR/traces/{thread_id}.jsonl에 기록
#
# 이 로그가 설계안 5번의 "이 선택지는 어떤 검색에서 나왔나 복원 가능"과
# "Agent 궤적 평가(어떤 tool을 몇 번 불렀나)"의 최소 단위 원자료가 된다.

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402

TRACE_DIR = Path(OUTPUT_DIR) / "traces"


def _trace_path(thread_id: str) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    return TRACE_DIR / f"{thread_id}.jsonl"


def log_event(thread_id: str, node: str, event: str, payload: Optional[dict] = None) -> None:
    """콜백이 못 잡는 지점을 JSONL에 append하고 콘솔에도 짧게 찍는다."""
    record = {
        "ts": time.time(),
        "thread_id": thread_id,
        "node": node,
        "event": event,
        "payload": payload or {},
    }
    with open(_trace_path(thread_id), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[trace] {node}.{event} {payload or ''}")


class TraceCallbackHandler(BaseCallbackHandler):
    """체인/tool 호출을 자동으로 콘솔에 출력하는 콜백 핸들러. graph.py의 config={"callbacks": [...]}로 연결한다."""

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self._start_times: dict[UUID, float] = {}

    def on_chain_start(self, serialized: dict, inputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        self._start_times[run_id] = time.time()
        name = (serialized or {}).get("name", "chain")
        print(f"[trace] -> node 진입: {name}")

    def on_chain_end(self, outputs: dict, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.time() - self._start_times.pop(run_id, time.time())
        print(f"[trace] <- node 종료 ({elapsed:.2f}s)")

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: UUID, **kwargs: Any) -> None:
        self._start_times[run_id] = time.time()
        name = (serialized or {}).get("name", "tool")
        print(f"[trace]   tool 호출: {name}({input_str[:120]!r})")

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.time() - self._start_times.pop(run_id, time.time())
        print(f"[trace]   tool 종료 ({elapsed:.2f}s)")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        print(f"[trace]   tool 에러: {error}")

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        log_event(self.thread_id, "llm", "error", {"error": str(error)})
