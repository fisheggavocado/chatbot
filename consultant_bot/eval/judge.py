# LLM-as-Judge — faq_agent.py/guardrail.py와 같은 llm.py 팩토리를 재사용해, evidence 대비 답변이
# 충실한지(faithful)/질문에 실제로 답했는지(relevant)/전반적 품질(1~5)을 별도의 LLM 호출로 채점한다.
#
# guardrail.py(GuardrailVerdict)가 "근거 없는 확정 표현이 있는가"만 이진 판정해 실제 대화 흐름을 막는 것과
# 달리, 이 judge는 오프라인 평가 전용이다 - 대화에는 전혀 관여하지 않고 run_eval.py에서만 호출된다.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402

from llm import get_structured_llm  # noqa: E402

JUDGE_PROMPT = """당신은 RAG 챗봇 답변 품질을 채점하는 평가자입니다. 아래 질문/답변/근거를 보고 판정하세요.

질문: {question}

답변:
{answer}

근거(evidence, 챗봇이 실제로 검색한 자료):
{evidence}

판정 기준:
- faithful: 답변이 근거에 없는 내용을 확정적으로 지어내지 않았으면 true. 근거가 아예 없는데(위 근거란이
  "(근거 없음)") 답변이 모른다고 정직하게 인정하거나 범위 밖이라고 안내했다면 이것도 faithful=true로
  판정하세요.
- relevant: 답변이 실제로 질문에 대한 응답이면 true (엉뚱한 화제로 새면 false. 범위 밖 질문에 대한
  정중한 거절은 relevant=true로 판정).
- score: 1(매우 나쁨)~5(매우 좋음). 근거 기반이고 질문에 맞으며 출처가 명시돼 있으면 높은 점수.
- reason: 판정 근거 한두 문장.
"""


class JudgeVerdict(BaseModel):
    faithful: bool = Field(description="근거 밖 내용을 지어내지 않았는가 (근거 없을 때 정직한 인정도 faithful)")
    relevant: bool = Field(description="답변이 실제로 질문에 응답하는가")
    score: int = Field(description="1~5 전체 품질 점수", ge=1, le=5)
    reason: str = Field(description="판정 근거 한두 문장")


def _format_evidence(evidence: list[dict]) -> str:
    return "\n".join(f"- [{e['source']} p.{e['page']}] {e['text'][:300]}" for e in evidence) or "(근거 없음)"


def judge_answer(question: str, answer: str, evidence: list[dict]) -> JudgeVerdict:
    """실제 대화에 쓰는 것과 같은 gpt-5-mini로 답변 품질을 채점한다 (호출 시 비용 발생)."""
    judge_llm = get_structured_llm(JudgeVerdict)
    return judge_llm.invoke(
        JUDGE_PROMPT.format(question=question, answer=answer, evidence=_format_evidence(evidence))
    )
