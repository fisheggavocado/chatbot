# PDF를 LlamaIndex의 Document 형식으로 로드하는 커스텀 리더
# 하이브리드 추출: 먼저 PyMuPDF로 텍스트 레이어를 직접 뽑고,
# - 텍스트가 부족하거나(스캔 이미지) 폰트 cmap 오류로 깨져 있으면 페이지 전체를 gpt-5-mini 비전 API로 해석하고,
# - 텍스트는 충분해도 로고/아이콘보다 큰 삽입 이미지가 있으면 그림 설명만 비전 API로 보강해 덧붙인다.
# 관련 저수준 로직은 pdf_reader.py에 있다.
# main.py는 스트리밍 저장/재개를 위해 load_data() 대신 iter_pages()로 한 페이지씩 처리한다.

from pathlib import Path
from typing import Iterator, List, Tuple

import pypdfium2 as pdfium
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

from config import MIN_EXTRACTABLE_TEXT_LENGTH, PDF_IMAGE_DPI
from pdf_reader import (
    describe_page_images,
    extract_page_text,
    has_corrupted_encoding,
    has_significant_image,
    image_to_text,
    page_to_png_bytes,
)


class VisionPDFReader(BaseReader):
    """PDF 페이지에서 텍스트를 하이브리드로 추출해(텍스트 레이어 우선, 부족하면 비전 API) Document로 반환한다."""

    def load_data(self, pdf_path: Path, dpi: int = PDF_IMAGE_DPI) -> List[Document]:
        """BaseReader 인터페이스 호환용. 내부적으로 iter_pages()를 모두 소비해 리스트로 반환한다."""
        return [document for _, document in self.iter_pages(pdf_path, dpi)]

    def iter_pages(
        self, pdf_path: Path, dpi: int = PDF_IMAGE_DPI, start_page: int = 0
    ) -> Iterator[Tuple[int, Document]]:
        """PDF를 페이지 단위로 순차 처리하며 (0-indexed 페이지 번호, Document)를 하나씩 반환한다.

        start_page를 지정하면 이전에 처리한 페이지는 건너뛰고 그 다음부터 이어서 처리한다 (재개용).
        """
        pdf_path = Path(pdf_path)
        doc = pdfium.PdfDocument(pdf_path)

        for page_index in range(start_page, len(doc)):
            page = doc[page_index]

            extracted = extract_page_text(page)
            if len(extracted) < MIN_EXTRACTABLE_TEXT_LENGTH or has_corrupted_encoding(extracted):
                # 텍스트 레이어가 부족하거나(스캔 페이지) 폰트 cmap 오류로 깨진 페이지: 전체를 비전 API로 대체 추출한다.
                image_bytes = page_to_png_bytes(page, dpi)
                text = image_to_text(image_bytes)
            elif has_significant_image(page):
                # 텍스트는 충분하지만 차트/다이어그램 등 삽입 이미지가 있는 페이지: 그림 설명만 보강해 덧붙인다.
                image_bytes = page_to_png_bytes(page, dpi)
                image_description = describe_page_images(image_bytes)
                text = f"{extracted}\n\n{image_description}" if image_description else extracted
            else:
                text = extracted

            page.close()

            document = Document(
                text=text, metadata={"source": pdf_path.name, "page": page_index + 1}
            )
            yield page_index, document

        doc.close()
