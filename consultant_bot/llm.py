# 공용 LLM 팩토리
# 설계안 5번 "역할별 설정 차이" 원칙: 모델은 gpt-5-mini 하나만 쓰고, 노드마다 프롬프트 길이/구조화 출력 스키마만 다르게 가져간다.
# 부모 프로젝트(first_project)의 config.py에 이미 있는 OPENAI_API_KEY/OPENAI_BASE_URL/VISION_MODEL을 그대로 재사용한다.

import sys
from pathlib import Path
from typing import Optional, Type, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from config import OPENAI_API_KEY, OPENAI_BASE_URL, VISION_MODEL  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

T = TypeVar("T", bound=BaseModel)

# --- 품질 튜닝 파라미터 (필요에 따라 조정) ---
CHAT_MODEL = VISION_MODEL  # 대화/라우팅에 쓸 모델. 정확도를 더 올리고 싶으면 더 큰 모델로 교체 가능
TEMPERATURE = 0  # gpt-5 계열이 아닌 모델로 교체했을 때만 적용됨 (아래 참고)
REASONING_EFFORT = "minimal"  # gpt-5 계열 전용. minimal|low|medium|high — 높일수록 정확도↑ 지연·비용↑

# gpt-5/o1/o3 등 "reasoning" 계열 모델은 temperature를 지원하지 않는다 — 기본값(1) 이외의 값을 보내면
# "Unsupported parameter: 'temperature' is not supported with this model" 400 에러가 난다.
# 대신 reasoning_effort로 답변의 결정성/속도를 조절한다. VISION_MODEL이 "openai/gpt-5-mini"처럼
# 게이트웨이 접두사가 붙어 있어도 감지되도록 부분 문자열로 판별한다.
_is_reasoning_model = any(tag in CHAT_MODEL.lower() for tag in ("gpt-5", "o1", "o3", "o4"))

_llm_kwargs: dict = dict(model=CHAT_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
if _is_reasoning_model:
    _llm_kwargs["reasoning_effort"] = REASONING_EFFORT
else:
    _llm_kwargs["temperature"] = TEMPERATURE

_base_llm = ChatOpenAI(**_llm_kwargs)


def get_llm() -> ChatOpenAI:
    """자유 텍스트 생성용 공용 LLM (FAQ 응답, Research Thought 등)."""
    return _base_llm


def get_structured_llm(schema: Type[T]):
    """schema(Pydantic 모델)로 강제 출력하는 LLM. Coordinator/Supervisor/Presenter/Guardrail이 사용."""
    return _base_llm.with_structured_output(schema)
