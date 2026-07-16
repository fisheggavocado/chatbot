# PDF 페이지를 이미지로 렌더링하고 gpt-5-mini(비전 모델)로 텍스트를 추출하는 저수준 유틸리티
# LlamaIndex 연동은 llama_pdf_reader.py에서 이 함수들을 그대로 재사용한다.
#
# PyMuPDF(fitz)는 이 개발 머신의 Windows Smart App Control에 네이티브 DLL이 차단되어
# pypdfium2(Chromium의 PDFium 바인딩)로 교체했다.

import base64
import io

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from openai import OpenAI

from config import MIN_IMAGE_AREA_RATIO, OPENAI_API_KEY, OPENAI_BASE_URL, VISION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# 비전 모델에게 페이지 이미지를 해석시킬 때 사용할 지시문
PAGE_PROMPT = """다음은 PDF 문서의 한 페이지 이미지입니다.
페이지에 보이는 내용을 아래 규칙에 따라 텍스트로 변환해주세요.

- 본문은 읽는 순서대로 그대로 전사한다.
- 표는 마크다운 표 형식으로 재구성한다.
- 차트/그림/도표는 "[그림: 핵심 내용 요약]" 형태로 간단히 설명한다.
- 페이지 번호, 머리글/바닥글 등 반복되는 장식 요소는 생략한다.
- 원문 언어를 그대로 유지한다(번역하지 않는다).
- 다른 설명 없이 변환된 텍스트만 출력한다.
"""

# 텍스트 레이어는 이미 충분한 페이지에서, 삽입된 그림만 추가로 설명시킬 때 쓰는 지시문.
# (본문을 다시 전사하면 텍스트 레이어와 중복되므로 그림 설명만 뽑는다.)
IMAGE_ONLY_PROMPT = """다음은 PDF 문서의 한 페이지 이미지입니다.
이 페이지의 본문 텍스트는 이미 별도로 추출되어 있으니, 본문은 무시하고
차트/그림/도표/사진 등 시각 자료만 아래 규칙에 따라 설명해주세요.

- 시각 자료 하나당 "[그림: 핵심 내용 요약]" 형태로 한 줄씩 설명한다.
- 시각 자료가 여러 개면 각각 줄바꿈으로 구분해서 나열한다.
- 설명할 시각 자료가 전혀 없으면 아무 것도 출력하지 않는다.
- 다른 설명 없이 결과만 출력한다.
"""


def count_pages(pdf_path) -> int:
    """비전 모델 호출 없이 PDF의 전체 페이지 수만 빠르게 센다 (예상 시간 계산용)."""
    doc = pdfium.PdfDocument(pdf_path)
    n_pages = len(doc)
    doc.close()
    return n_pages


def extract_page_text(page: "pdfium.PdfPage") -> str:
    """PDFium으로 페이지에 내장된 텍스트 레이어를 직접 추출한다.

    스캔 이미지로만 이루어진 페이지는 텍스트 레이어가 없어 빈 문자열(또는 매우 짧은 문자열)을 반환한다.
    """
    textpage = page.get_textpage()
    text = textpage.get_text_range().strip()
    textpage.close()
    return text


def has_significant_image(page: "pdfium.PdfPage", min_area_ratio: float = MIN_IMAGE_AREA_RATIO) -> bool:
    """페이지에 로고/아이콘보다 큰, 의미 있는 삽입 이미지(차트·다이어그램 등)가 있는지 확인한다.

    텍스트 레이어가 충분해도 이런 이미지가 있으면 비전 API로 그림 설명을 보강해야 한다.
    """
    page_width, page_height = page.get_size()
    page_area = page_width * page_height
    if page_area <= 0:
        return False

    for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,)):
        x0, y0, x1, y1 = obj.get_bounds()
        image_area = max(x1 - x0, 0) * max(y1 - y0, 0)
        if image_area / page_area >= min_area_ratio:
            return True
    return False


def page_to_png_bytes(page: "pdfium.PdfPage", dpi: int) -> bytes:
    """PDF 한 페이지를 지정한 해상도(dpi)의 PNG 이미지 바이트로 렌더링한다."""
    scale = dpi / 72  # PDF 기본 단위(72dpi) 대비 확대 배율 계산
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def _call_vision(prompt: str, image_bytes: bytes) -> str:
    """페이지 이미지와 지시문을 gpt-5-mini 비전 API에 전달해 결과 텍스트를 받는다."""
    # OpenAI 비전 API는 이미지를 base64 data URL 형태로 전달해야 한다.
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()


def image_to_text(image_bytes: bytes) -> str:
    """페이지 이미지를 gpt-5-mini에 전달해 텍스트로 변환한다 (텍스트 레이어가 부족한 스캔 페이지용)."""
    return _call_vision(PAGE_PROMPT, image_bytes)


def describe_page_images(image_bytes: bytes) -> str:
    """텍스트 레이어는 충분하지만 삽입 이미지가 있는 페이지에서, 그림 설명만 뽑아낸다."""
    return _call_vision(IMAGE_ONLY_PROMPT, image_bytes)
