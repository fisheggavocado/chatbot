# consultant_bot 평가용 E2E 시나리오 데이터셋.
# run_eval.py가 이 목록을 그대로 순회하며 실제 그래프를 실행하고 judge.py로 채점한다.
# 새 케이스를 추가하려면 이 파일에 항목만 추가하면 된다(하네스 코드 변경 불필요).

FAQ_CASES = [
    {"id": "faq_rag", "question": "RAG가 뭐야?", "expected_intent": "faq"},
    {"id": "faq_embedding", "question": "임베딩이 무엇인지 설명해줘", "expected_intent": "faq"},
]

OUT_OF_SCOPE_CASES = [
    {"id": "oos_dinner", "question": "오늘 저녁 메뉴 추천해줘", "expected_intent": "out_of_scope"},
]

# design 케이스는 위저드 3턴(interrupt) 흐름이라 run_eval.py가 각 턴에서 제시된 선택지 중 첫 번째를
# 자동으로 골라 끝까지 진행시킨다 (test/run_scenarios.py의 _pick_resume과 동일한 방식).
DESIGN_CASES = [
    {
        "id": "design_pipeline",
        "question": "AI 챗봇 파이프라인을 어떻게 설계하면 좋을지 추천해줘",
        "expected_intent": "design",
    },
]

ALL_CASES = FAQ_CASES + OUT_OF_SCOPE_CASES + DESIGN_CASES
