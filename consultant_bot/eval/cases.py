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

# extract(B형) 케이스는 interrupt() 없는 짧은 연속 대화라 run_eval.py가 questions 리스트를 순서대로
# graph.invoke한다(design의 위저드 재개 루프와 달리, coordinator의 sticky 라우팅이 턴을 이어준다).
EXTRACT_CASES = [
    {
        "id": "extract_hf_pipeline",
        "questions": [
            "Hugging Face Pipeline 강의 내용을 요약해줘",
            "관련된 예시 코드 중 하나를 작성해줘",
            "이 코드를 해석해줘",
        ],
        "expected_intent": "extract",
    },
]

ALL_CASES = FAQ_CASES + OUT_OF_SCOPE_CASES + DESIGN_CASES + EXTRACT_CASES
