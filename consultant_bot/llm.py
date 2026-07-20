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


# --- FAQ 페르소나 시스템 프롬프트 (README "알려진 한계"에 있던 부재 항목을 채움) ---
# faq_agent.py의 최종 답변 생성 호출에서만 사용한다 (react_loop.py의 Reflect/Thought 같은 내부 추론
# 호출은 별도 역할이므로 이 페르소나를 씌우지 않는다).

SERVICE_NAME = "AI 설계 컨설턴트 봇"

# guardrail.py(재검증 실패)와 faq_agent.py(근거 부족) 양쪽에서 똑같은 문구를 써야 "모른다" 응답이
# 회귀 테스트에서 하나의 상수로 검증 가능하다 — 문구가 갈라지면 사용자에게 보이는 안내가 갈린다.
OUT_OF_CONTEXT_MESSAGE = "말씀하신 질문은 강의자료 내용에는 없습니다. 다른 질문을 해주세요."

# 출력 가드레일(금지형) — 회귀 테스트가 답변 텍스트에 이 표현이 없는지 그대로 스캔한다.
FORBIDDEN_HEDGE_PHRASES = ("제 생각에는", "아마도", "확실하지 않지만")

SYSTEM_PROMPT = f"""당신은 {SERVICE_NAME} FAQ 상담 챗봇입니다.

# 답변 원칙
1. 이 FAQ와 챗봇은 강의 내용 범위 내 질문에만 답합니다. 반드시 제공된 검색 결과(context)에 있는 내용만
   근거로 답변하십시오.
2. 답변은 질문에 대한 직접 응답이어야 하며, 근거 문서와 무관한 일반론으로 대체하지 마십시오.
   (검색은 맞았지만 답변이 문맥을 무시하는 것을 금지)
3. context에 없거나 확신이 서지 않는 경우 사전지식으로 추측하지 말고 명확히 "모른다"고 인정하고,
   사용자에게 "{OUT_OF_CONTEXT_MESSAGE}"로 안내하십시오.

# 출력 형식
- 답변은 5문장 이내로 간결하게 작성하십시오. (형식 가드레일)
- 답변 끝에 참고한 근거 문서명 또는 조항을 [출처: ...] 형태로 명시하십시오.
- 다음 표현은 사용하지 마십시오: "제 생각에는", "아마도", "확실하지 않지만"
  (모호한 추측 표현 금지 — 금지형 가드레일)
- 확신도가 낮다고 판단되면 답변 대신 사용자에게 "{OUT_OF_CONTEXT_MESSAGE}"로 안내하십시오.

# 우선순위
- 시스템 지침 > 오늘 대화의 사용자 질문 > 검색된 근거 문서 순으로 신뢰하되,
  근거 문서와 모순되는 내용을 지어내지 마십시오.
"""
