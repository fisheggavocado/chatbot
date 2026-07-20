# 관찰가능성(observability) 모듈 — 파일 기반 트레이스(항상 켜짐) + 선택적 LangSmith 외부 대시보드 연동.
#
# 세 축으로 구성:
#   1) TraceCallbackHandler - LangChain/LangGraph의 체인·tool 호출을 자동으로 훅해 콘솔에 사람이 읽을 수 있게 출력
#   2) log_event()          - 콜백으로 안 잡히는 지점(라우팅 결정, react_loop 정지 사유, guardrail 판정,
#                              interrupt 진입/재개)까지 OUTPUT_DIR/traces/{thread_id}.jsonl에 기록
#   3) configure_langsmith() - .env에 LANGCHAIN_API_KEY가 있으면 LangSmith 웹 대시보드 트레이싱을 켠다.
#                              LangChain/LangGraph는 이 두 환경변수만 있으면 추가 코드 없이 모든
#                              chain/tool 실행을 자동으로 LangSmith에 전송하므로(전역 트레이서), 이 모듈은
#                              활성화 여부만 판단해 환경변수를 세팅해줄 뿐 트레이싱 자체를 구현하지 않는다.
#                              키가 없으면 조용히 건너뛰고 1)/2)의 로컬 트레이스만 남는다(외부 계정 불필요).
#
# 이 로그(및 LangSmith가 있다면 그쪽 트레이스)가 설계안 5번의 "이 선택지는 어떤 검색에서 나왔나 복원 가능"과
# "Agent 궤적 평가(어떤 tool을 몇 번 불렀나)"의 최소 단위 원자료가 된다.

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402

TRACE_DIR = Path(OUTPUT_DIR) / "traces"
DEFAULT_LANGCHAIN_PROJECT = "consultant-bot"


def configure_langsmith() -> bool:
    """LANGCHAIN_API_KEY가 설정돼 있으면 LangSmith 트레이싱을 활성화하고 True를 반환한다.

    없으면 아무 것도 하지 않고 False를 반환한다(선택 기능 - LangSmith 계정이 없어도 로컬 JSONL
    트레이스만으로 정상 동작). config.py의 load_dotenv()가 이미 .env를 읽어 os.environ을 채워둔 상태를
    전제로 한다(이 모듈이 import하는 `config` 모듈의 부수효과).
    """
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return False
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", DEFAULT_LANGCHAIN_PROJECT)
    print(f"[observability] LangSmith 트레이싱 활성화 (project={os.environ['LANGCHAIN_PROJECT']})")
    return True


LANGSMITH_ENABLED = configure_langsmith()


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
