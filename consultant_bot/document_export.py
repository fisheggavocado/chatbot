# ⑥ Document Export — compare 단계가 끝났을 때, tech_select/pipeline_select/compare 3턴 결과를
# 마크다운 문서로 묶어 로컬 OUTPUT_DIR/design_docs/에 저장한다. wizard_supervisor가 done 전환 시 호출하고,
# server.py의 /download/{thread_id}가 같은 경로 규칙(thread_id -> 파일명)으로 이 파일을 서빙한다.

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import OUTPUT_DIR  # noqa: E402

DOC_DIR = Path(OUTPUT_DIR) / "design_docs"


def _safe_filename(thread_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", thread_id)


def doc_path_for_thread(thread_id: str) -> Path:
    return DOC_DIR / f"{_safe_filename(thread_id)}.md"


def _render_tech_select(entry: dict) -> str:
    output = entry.get("output") or {}
    selected = set(entry.get("resume") or [])
    lines = ["| 선택 | 기술 | 역할 | 출처 |", "| --- | --- | --- | --- |"]
    for row in output.get("rows", []):
        mark = "✅" if row.get("technology") in selected else ""
        lines.append(f"| {mark} | {row.get('technology', '')} | {row.get('role', '')} | {row.get('source', '')} |")
    return "\n".join(lines)


def _render_pipeline_select(entry: dict) -> str:
    output = entry.get("output") or {}
    resume = entry.get("resume") or {}
    picked = resume.get("pipeline") if isinstance(resume, dict) else None
    blocks = []
    for opt in output.get("options", []):
        mark = " (선택됨)" if opt.get("name") == picked else ""
        steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(opt.get("steps", [])))
        blocks.append(f"### {opt.get('name', '')}{mark}\n\n- 출처: {opt.get('source', '')}\n\n{steps}")
    return "\n\n".join(blocks)


def _render_compare(entry: dict) -> str:
    output = entry.get("output") or {}
    lines = ["| 항목 | 옵션 A | 옵션 B | 출처 |", "| --- | --- | --- | --- |"]
    for row in output.get("rows", []):
        lines.append(f"| {row.get('aspect', '')} | {row.get('option_a', '')} | {row.get('option_b', '')} | {row.get('source', '')} |")
    return "\n".join(lines)


_SECTIONS = {
    "tech_select": ("1. 기술/컴포넌트 선택", _render_tech_select),
    "pipeline_select": ("2. 파이프라인 선택", _render_pipeline_select),
    "compare": ("3. 비교", _render_compare),
}


def build_document(question: str, results: list[dict]) -> str:
    parts = ["# AI 파이프라인 설계 상담 결과", "", f"**요청**: {question}", ""]
    for entry in results:
        section = _SECTIONS.get(entry.get("stage"))
        if section is None:
            continue
        title, renderer = section
        parts.append(f"## {title}")
        parts.append("")
        parts.append(renderer(entry))
        parts.append("")
    return "\n".join(parts)


def save_document(thread_id: str, question: str, results: list[dict]) -> Path:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    path = doc_path_for_thread(thread_id)
    path.write_text(build_document(question, results), encoding="utf-8")
    return path
