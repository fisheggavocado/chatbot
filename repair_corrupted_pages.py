# 일회성 복구 스크립트: 폰트 cmap 오류로 깨진 텍스트가 인덱싱된 페이지만 골라
# 비전 API(gpt-5-mini)로 재추출하고, 기존 인덱스 노드를 교체한다.
# 전체 재임베딩(수 시간) 대신, 실제로 깨진 페이지 수(수십 개)만큼만 비용을 쓴다.
#
# 실행 전 output/*.json은 이미 output_backup_*/ 로 백업되어 있어야 한다.
#
# 사용법: python repair_corrupted_pages.py

from pathlib import Path

from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, OUTPUT_DIR, PDF_IMAGE_DPI
from embedder import encode_tokens
from llama_embedding import BGEM3Embedding
from pdf_reader import has_corrupted_encoding, image_to_text, page_to_png_bytes

import pypdfium2 as pdfium
from llama_index.core import Document

Settings.embed_model = BGEM3Embedding()

# docstore의 source 메타데이터 -> 실제 로컬 PDF 파일 경로.
# (인덱싱 당시 붙은 번호 프리픽스가 로컬 파일명과 다를 수 있어 수동으로 매핑한다.
#  page1 텍스트를 대조해 동일 문서/페이지 번호임을 이미 확인함.)
# source 문자열 자체를 dict 키로 타이핑하면 한글 유니코드 정규화 차이로 매칭이 실패할 수 있어
# (실제로 스캔 과정에서 한 번 겪음), 영문 마커 부분 문자열로 매칭한다.
MARKER_TO_LOCAL_PATH = [
    (
        "MCP",
        r"D:\누리\이어드림 AI 교육\수업 자료\6w\01 실무형 MCP서버 완성과 연계"
        r"\[수업자료] 실무형 MCP 서버 완성과 연계.pdf",
    ),
    (
        "Fine-tuning",
        r"D:\누리\이어드림 AI 교육\수업 자료\10강\[강의자료] Fine-tuning된 LLM 평가 및 최적화.pdf",
    ),
    (
        "RAG",
        r"D:\누리\이어드림 AI 교육\수업 자료\5w\RAG 검색 품질 전략 실습\[수업자료] RAG 검색 품질 전략.pdf",
    ),
]


def _resolve_local_path(source: str) -> str:
    for marker, path in MARKER_TO_LOCAL_PATH:
        if marker in source:
            return path
    raise RuntimeError(f"로컬 PDF 매핑을 찾지 못했습니다: {source!r}")


def main():
    output_dir = Path(OUTPUT_DIR)
    storage_context = StorageContext.from_defaults(persist_dir=str(output_dir))
    index = load_index_from_storage(storage_context)
    docstore = index.docstore

    # 1) 깨진 노드를 (source, page) 단위로 그룹핑
    corrupted_by_page: dict[tuple[str, int], list[str]] = {}
    for node_id, node in docstore.docs.items():
        text = node.get_content()
        if has_corrupted_encoding(text):
            meta = node.metadata
            key = (meta.get("source"), meta.get("page"))
            corrupted_by_page.setdefault(key, []).append(node_id)

    print(f"[스캔] 깨진 (source, page) {len(corrupted_by_page)}건, 노드 {sum(len(v) for v in corrupted_by_page.values())}개")

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, tokenizer=encode_tokens)

    # 2) source별로 PDF를 한 번만 열어서, 해당하는 페이지들만 비전 API로 재추출
    by_source: dict[str, list[int]] = {}
    for (source, page), _ in corrupted_by_page.items():
        by_source.setdefault(source, []).append(page)

    fixed_pages = 0
    still_bad = []

    for source, pages in by_source.items():
        pdf_path = _resolve_local_path(source)
        doc = pdfium.PdfDocument(pdf_path)
        for page_num in sorted(pages):
            page_index = page_num - 1  # metadata.page는 1-indexed
            page = doc[page_index]
            image_bytes = page_to_png_bytes(page, PDF_IMAGE_DPI)
            page.close()

            new_text = image_to_text(image_bytes)
            if has_corrupted_encoding(new_text):
                # 비전 API 결과까지 깨졌다면(거의 없겠지만) 자동 교체하지 않고 보고만 한다.
                still_bad.append((source, page_num))
                print(f"[경고] 비전 재추출 후에도 의심스러운 문자 포함: {source} p.{page_num}")
                continue

            node_ids = corrupted_by_page[(source, page_num)]
            index.delete_nodes(node_ids, delete_from_docstore=True)

            new_doc = Document(text=new_text, metadata={"source": source, "page": page_num})
            new_nodes = splitter.get_nodes_from_documents([new_doc])
            index.insert_nodes(new_nodes)

            fixed_pages += 1
            print(f"[교체] {source} p.{page_num} - 노드 {len(node_ids)}개 -> {len(new_nodes)}개 (텍스트 {len(new_text)}자)")
        doc.close()

    # 3) 디스크에 반영
    index.storage_context.persist(persist_dir=str(output_dir))
    print(f"[완료] {fixed_pages}페이지 교체, 미해결 {len(still_bad)}건")
    if still_bad:
        print("[미해결 목록]", still_bad)

    # 4) 캐시 무효화 (인덱스가 바뀌었으므로 과거 "모른다" 캐시가 남아있지 않도록 즉시 비운다).
    # cache.py는 consultant_bot/ 아래에 있어 임포트 경로가 꼬이기 쉬우므로, 같은 파일들을 직접 지운다
    # (cache.py의 invalidate_all()과 동일한 동작 - search/faq 캐시 + 지문 메타 삭제).
    for name in ("search_cache.json", "faq_cache.json", "cache_meta.json"):
        p = output_dir / name
        if p.exists():
            p.unlink()
            print(f"[캐시 무효화] {name} 삭제")


if __name__ == "__main__":
    main()
