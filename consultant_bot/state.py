# 설계안(chat_log/AI 설계 컨설턴트 봇 — Multi Agent 구성안.py) 3번 State Contract를 그대로 옮긴 모듈.
# 그래프 전체가 공유하는 WizardState와, 노드 간에 주고받는 구조화 스키마를 여기 모아둔다.

import operator
from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Stage = Literal["tech_select", "pipeline_select", "compare", "detail", "done"]
Intent = Literal["faq", "design", "out_of_scope", "extract"]


class Evidence(TypedDict):
    """hybrid_search 결과 1건. source/page가 Presenter·FAQ 응답의 출처 컬럼 근거가 된다."""

    text: str
    source: str
    page: int | str
    score: float


class WizardState(TypedDict):
    messages: Annotated[list, add_messages]  # 대화 이력 (reducer 누적)
    intent: Intent
    stage: Stage  # 위저드 단계 커서
    selected_techs: list[str]  # 1턴 사용자 다중 선택 (interrupt 결과)
    selected_pipeline: Optional[str]  # 3턴 선택
    evidence: Annotated[list[Evidence], operator.add]  # 검색 근거 누적 (reducer 필수!)
    trace: Annotated[list[str], operator.add]  # 실행 경로 기록
    last_error: Optional[dict]  # error envelope (재시도 판단용)
    budget_remaining: int  # ReAct 반복 예산
    presenter_output: Optional[dict]  # Presenter가 만든 이번 턴 구조화 출력 (interrupt() payload로 사용)
    guardrail_retries: int  # guardrail 재시도 횟수 (1회 초과 재시도 방지용 카운터)
    extract_source: Optional[str]  # B형(lecture_agent)이 잠근 단일 문서 출처 (연속 턴 검색 범위 고정용)
    extract_turns: int  # B형 연속 턴 카운터 (coordinator의 sticky 라우팅 상한 체크용)


# --- Coordinator: 규칙 기반으로 애매할 때만 LLM 폴백에 쓰는 구조화 출력 ---
class IntentDecision(BaseModel):
    intent: Intent = Field(description="사용자 발화의 의도 분류")
    reason: str = Field(description="분류 근거 한 줄")


# --- Wizard Supervisor: 다음 노드를 구조화 출력으로 받는다 (LangManus Router 패턴) ---
class RouterDecision(BaseModel):
    next: Literal["research_worker", "presenter", "end"] = Field(description="다음에 실행할 노드")
    reason: str = Field(description="라우팅 근거 한 줄")


# --- react_loop.py의 Reflect 단계 구조화 출력 ---
class ReflectDecision(BaseModel):
    sufficient: bool = Field(description="지금까지 모은 근거로 답변하기에 충분한지 여부")
    next_query: Optional[str] = Field(default=None, description="부족하다면 다음에 검색할 쿼리")
    reason: str = Field(description="판단 근거 한 줄")


# --- Presenter: 턴별 structured output (스키마 자체가 출력 스키마 가드) ---
class TechRow(BaseModel):
    technology: str
    role: str
    source: str


class TechSelectOutput(BaseModel):
    """1턴: 기술 + 역할 체크박스 표."""

    rows: list[TechRow]
    prompt: str = Field(description="사용자에게 보여줄 안내 문구 (다중 선택 요청)")


class PipelineOption(BaseModel):
    name: str
    steps: list[str] = Field(
        description=(
            "사용자 요청 1건이 시스템에 입력되어 응답이 나가기까지 거치는 런타임 처리 순서"
            "(예: '질문 입력 -> 하이브리드 검색 -> 근거로 답변 생성'). 모델 준비·파인튜닝·배포 같은 "
            "구축/설치 절차가 아니라, 실행 중 요청이 통과하는 단계여야 한다."
        )
    )
    source: str


class PipelineSelectOutput(BaseModel):
    """2턴: 파이프라인 2개 + 예/아니오."""

    options: list[PipelineOption]
    prompt: str = Field(description="사용자에게 보여줄 안내 문구 (파이프라인 선택 + 예/아니오 요청)")


class CompareRow(BaseModel):
    aspect: str
    option_a: str
    option_b: str
    source: str


class CompareOutput(BaseModel):
    """3턴: 비교표."""

    rows: list[CompareRow]
    prompt: str = Field(description="사용자에게 보여줄 안내 문구")


# --- Guardrail 판정 ---
class GuardrailVerdict(BaseModel):
    passed: bool
    reason: str = Field(description="실패 시 어떤 표현이 근거 없이 확정적이었는지")
