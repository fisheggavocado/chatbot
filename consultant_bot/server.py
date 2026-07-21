# HTTP API 진입점. app.py의 터미널 input() 루프 대신, gcube처럼 컨테이너에 대화형 stdin이
# 붙지 않는 배포 환경에서도 동작하도록 질문/응답을 HTTP 요청으로 주고받는다.
#
# 그래프의 interrupt()/Command(resume=...) 흐름은 요청 1건 = graph.invoke() 1회로 매핑한다.
# 즉 위저드 중간에 선택지가 나오면(interrupt) 그 자리에서 응답을 끝내고 클라이언트가 다음 선택을
# 담아 /chat/resume을 별도로 호출하는 식으로, app.py의 while 루프(터미널에서 그 자리에서 입력받기)를
# 요청-응답 두 엔드포인트로 나눈 것뿐이다.
#
# 사용법:
#   uvicorn consultant_bot.server:app --host 0.0.0.0 --port 8000
#   (Dockerfile 기본 명령이 이 서버를 실행한다)
#
# API:
#   GET  /            -> static/index.html — 브라우저로 여는 채팅 화면(바닐라 JS, 아래 API를 그대로 호출).
#                         gcube에 배포하면 그 워크로드의 공개 URL을 그대로 열면 된다.
#   POST /chat/message {message, thread_id?}  -> 새 대화면 thread_id 생략(서버가 발급), 이어가는
#                                                 대화면 이전 응답의 thread_id를 그대로 보낸다.
#   POST /chat/resume   {thread_id, resume}    -> status="interrupt" 응답을 받았을 때만 호출.
#                                                 resume 값의 형태는 stage에 따라 다르다:
#                                                   tech_select    -> ["기술1", "기술2", ...]
#                                                   pipeline_select -> {"pipeline": "이름", "confirm": true}
#                                                   compare        -> {"confirm": true}
#   두 엔드포인트 모두 {status: "interrupt", stage, payload} 또는 {status: "done", answer, download_url?} 를 반환한다.
#   GET  /download/{thread_id} -> compare 단계까지 끝낸 위저드 결과를 정리한 마크다운 문서(FileResponse).
#                                  document_export.save_document()가 OUTPUT_DIR/design_docs/에 저장해 둔 파일.

import os

# torch/BGE-M3/리랭커/kiwipiepy가 각자 자체 OpenMP 런타임(libiomp5md.dll 등)을 들고 있어, 같은 프로세스에서
# 전부 로드되면(Windows에서 특히) "OMP: Error #15" 세그멘테이션 폴트로 죽는다. graph.py를 import하기 전에
# 반드시 설정해야 해서 다른 import보다 앞에 둔다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Optional  # noqa: E402
from uuid import uuid4  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from document_export import doc_path_for_thread  # noqa: E402
from graph import backup_checkpoint, get_graph, make_run_config  # noqa: E402

app = FastAPI(title="consultant_bot")

_INDEX_HTML = (Path(__file__).resolve().parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def chat_ui() -> str:
    return _INDEX_HTML


class MessageRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ResumeRequest(BaseModel):
    thread_id: str
    resume: Any


def _respond(result: dict, thread_id: str, config: dict) -> dict:
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        stage = get_graph().get_state(config).values.get("stage")
        backup_checkpoint()  # 위저드 중간(interrupt로 멈춘) 상태도 컨테이너 재시작에 대비해 HF에 백업
        return {"thread_id": thread_id, "status": "interrupt", "stage": stage, "payload": interrupt_obj.value}

    final_messages = result.get("messages", [])
    answer = final_messages[-1].content if final_messages else ""
    response = {"thread_id": thread_id, "status": "done", "answer": answer}
    if final_messages and final_messages[-1].additional_kwargs.get("download_path"):
        response["download_url"] = f"/download/{thread_id}"
    backup_checkpoint()
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/download/{thread_id}")
def download_document(thread_id: str) -> FileResponse:
    path = doc_path_for_thread(thread_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="설계 상담 문서를 찾을 수 없습니다.")
    return FileResponse(path, media_type="text/markdown", filename=path.name)


@app.post("/chat/message")
def chat_message(req: MessageRequest) -> dict:
    thread_id = req.thread_id or str(uuid4())
    config = make_run_config(thread_id)
    result = get_graph().invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
    return _respond(result, thread_id, config)


@app.post("/chat/resume")
def chat_resume(req: ResumeRequest) -> dict:
    config = make_run_config(req.thread_id)
    snapshot = get_graph().get_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="이 thread_id에는 재개할 interrupt가 없습니다.")
    result = get_graph().invoke(Command(resume=req.resume), config=config)
    return _respond(result, req.thread_id, config)
