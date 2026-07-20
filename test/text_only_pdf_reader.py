# 로컬 CPU 테스트 전용 리더: 비전 API를 전혀 호출하지 않고 pypdfium2의 텍스트 레이어만으로 PDF를 읽는다.
# (Windows Smart App Control 등 환경 제약이나 비전 API 비용/네트워크 이슈를 피해 순수 텍스트만으로
# 멀티에이전트 배관(plumbing)이 맞는지 확인하고 싶을 때 사용한다.)
#
# llama_pdf_reader.VisionPDFReader와 같은 iter_pages(pdf_path, start_page=0) 인터페이스를 제공하는
# drop-in 대체품이라, embed_one_pdf.py가 main.py의 리더만 이걸로 바꿔치기해서 쓴다.
#
# 주의: 스캔 이미지 위주 페이지는 텍스트 레이어가 거의/전혀 없어 빈 텍스트로 인덱싱된다.
# 멀티에이전트 배관 테스트가 목적이라 무방하지만, 실제 서비스용 인덱스는 항상 main.py(비전 하이브리드)로 만들어야 한다.

from pathlib import Path
from typing import Iterator, List, Tuple

import pypdfium2 as pdfium
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader

from pdf_reader import extract_page_text


class TextOnlyPDFReader(BaseReader):
    """비전 API 없이 텍스트 레이어만 추출하는 리더 (로컬 CPU 테스트 전용)."""

    def load_data(self, pdf_path: Path) -> List[Document]:
        return [document for _, document in self.iter_pages(pdf_path)]

    def iter_pages(self, pdf_path: Path, start_page: int = 0) -> Iterator[Tuple[int, Document]]:
        pdf_path = Path(pdf_path)
        doc = pdfium.PdfDocument(pdf_path)

        for page_index in range(start_page, len(doc)):
            page = doc[page_index]
            text = extract_page_text(page)
            page.close()

            document = Document(text=text, metadata={"source": pdf_path.name, "page": page_index + 1})
            yield page_index, document

        doc.close()
