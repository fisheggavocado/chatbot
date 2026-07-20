# 실행 진입점 (LlamaIndex 기반, 스트리밍 저장 + 재개 지원)
#
# PDF_DIR(또는 --use-hf 시 Hugging Face Dataset에서 동기화한 폴더) 내 PDF를 페이지 단위로 순차 처리하며,
# 페이지가 끝날 때마다 즉시 인덱스에 반영·저장하고 진행 상황을 checkpoint.json에 기록한다.
# 중간에 끊겨도 다시 실행하면 checkpoint를 보고 이어서 처리한다 (완료된 PDF/페이지는 건너뜀).
#
# 사용법:
#   python main.py                 # PDF_DIR 전체 처리
#   python main.py --limit 1       # 시험 삼아 PDF 1개만 처리
#   python main.py --use-hf        # PDF_DIR 대신 HF_REPO_ID에서 PDF를 동기화해 처리

import argparse
import time
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter

from checkpoint import get_resume_page, is_pdf_done, load_checkpoint, mark_page_done
from config import CHUNK_OVERLAP, CHUNK_SIZE, HF_REPO_ID, HF_REPO_ID_ORG, OUTPUT_DIR, PDF_DIR, PERSIST_EVERY_N_PAGES
from embedder import encode_tokens
from hf_storage import restore_output_from_hf, sync_pdfs_from_hf, upload_output_to_hf
from llama_embedding import BGEM3Embedding
from llama_pdf_reader import VisionPDFReader
from pdf_reader import count_pages
from time_estimate import print_measured_estimate, print_upfront_estimate

# LlamaIndex 전역 설정: 기본값인 OpenAI 임베딩 대신 로컬 BGE-M3를 쓰도록 고정한다.
Settings.embed_model = BGEM3Embedding()


def load_or_create_index(output_dir: Path) -> VectorStoreIndex:
    """기존에 저장된 인덱스가 있으면 이어서 쓰고, 없으면 빈 인덱스를 새로 만든다."""
    if (output_dir / "docstore.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=str(output_dir))
        return load_index_from_storage(storage_context)
    storage_context = StorageContext.from_defaults()
    return VectorStoreIndex([], storage_context=storage_context)


def main():
    parser = argparse.ArgumentParser(
        description="PDF -> gpt-5-mini 텍스트 추출 -> 청킹 -> BGE-M3 임베딩 (스트리밍 저장/재개 지원)"
    )
    parser.add_argument("--limit", type=int, default=None, help="처리할 PDF 개수 제한 (예: 1 = 시험 삼아 1개만)")
    parser.add_argument(
        "--use-hf", action="store_true", help="PDF_DIR 대신 Hugging Face Dataset에서 PDF를 동기화해 사용"
    )
    args = parser.parse_args()

    pdf_dir = Path(sync_pdfs_from_hf()) if args.use_hf else Path(PDF_DIR)
    if not str(pdf_dir) or not pdf_dir.exists():
        raise ValueError(f"PDF 폴더 경로를 확인해주세요: '{pdf_dir}'")

    # HF Dataset repo는 PDF를 하위 폴더에 둘 수 있으므로 재귀적으로 찾는다.
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if args.limit:
        pdf_files = pdf_files[: args.limit]
    if not pdf_files:
        print(f"'{pdf_dir}'에서 PDF 파일을 찾지 못했습니다.")
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 로컬에 진행 기록이 없으면(휘발성 컨테이너 재시작 등) HF의 백업에서 이어받는다.
    if HF_REPO_ID and not (output_dir / "checkpoint.json").exists():
        if restore_output_from_hf(str(output_dir)):
            print("[재개 준비] Hugging Face 백업에서 이전 진행분을 복원했습니다.")

    state = load_checkpoint()

    # 비전 API를 호출하지 않고 페이지 수만 세어, 처리 전 예상 시간을 먼저 안내한다.
    page_counts = {pdf_path: count_pages(pdf_path) for pdf_path in pdf_files}
    total_pages = sum(page_counts.values())
    pending_pages = sum(
        page_counts[pdf_path] - get_resume_page(state, pdf_path.name)
        for pdf_path in pdf_files
        if not is_pdf_done(state, pdf_path.name)
    )

    print(f"[대상] PDF {len(pdf_files)}개, 총 {total_pages}페이지 (처리 남은 페이지: {pending_pages}장)")
    print_upfront_estimate(pending_pages)

    reader = VisionPDFReader()
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, tokenizer=encode_tokens
    )
    index = load_or_create_index(output_dir)

    processed_count = 0
    measured = False  # 첫 페이지 실측 후 True로 전환

    for pdf_path in pdf_files:
        if is_pdf_done(state, pdf_path.name):
            print(f"[건너뜀] {pdf_path.name} (이미 완료됨)")
            continue

        resume_page = get_resume_page(state, pdf_path.name)
        if resume_page > 0:
            print(f"[재개] {pdf_path.name} - {resume_page + 1}페이지부터 이어서 처리")

        pdf_total_pages = page_counts[pdf_path]

        for page_index, document in reader.iter_pages(pdf_path, start_page=resume_page):
            t0 = time.time()

            # 페이지 하나를 청킹 -> 인덱스에 즉시 반영 -> (설정 주기마다) 디스크에 저장한다.
            nodes = splitter.get_nodes_from_documents([document])
            index.insert_nodes(nodes)
            if (page_index + 1) % PERSIST_EVERY_N_PAGES == 0:
                index.storage_context.persist(persist_dir=str(output_dir))

            # checkpoint도 페이지마다 즉시 저장해, 다음 실행 시 정확히 이 지점부터 재개할 수 있게 한다.
            mark_page_done(state, pdf_path.name, page_index, pdf_total_pages)

            elapsed = time.time() - t0
            processed_count += 1

            if not measured:
                remaining = pending_pages - processed_count
                print_measured_estimate(elapsed, remaining)
                measured = True

            print(
                f"[{processed_count}/{pending_pages}] {pdf_path.name} "
                f"p.{page_index + 1}/{pdf_total_pages} - {elapsed:.1f}초, {len(nodes)}개 노드"
            )

        # 페이지 루프 종료 시점의 최종 상태를 디스크에 확실히 반영한다.
        index.storage_context.persist(persist_dir=str(output_dir))

        # PDF 하나가 끝날 때마다 HF에 백업해, 컨테이너가 죽어도 완료된 PDF까지는 보존한다.
        # 업로드가 일시적으로 실패해도 처리는 계속하고, 다음 PDF 완료 시 백업에 함께 포함된다.
        # 업로드는 쓰기 권한이 있는 HF_REPO_ID_ORG로 간다 (HF_REPO_ID는 읽기 전용).
        if HF_REPO_ID_ORG:
            try:
                upload_output_to_hf(str(output_dir))
            except Exception as e:
                print(f"[백업 실패 - 계속 진행] {pdf_path.name}: {e}")

    print(f"[완료] {output_dir}")

    if HF_REPO_ID_ORG:
        print("[업로드] 인덱스를 Hugging Face(쓰기 권한 repo)에 업로드합니다...")
        upload_output_to_hf(output_dir)
    else:
        print("[건너뜀] HF_REPO_ID_ORG가 설정되지 않아 Hugging Face 업로드를 건너뜁니다.")


if __name__ == "__main__":
    main()
