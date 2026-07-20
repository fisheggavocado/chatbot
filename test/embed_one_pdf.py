# 멀티에이전트 테스트용: HF_REPO_ID의 PDF 중 1개만 격리된 test/output_test/에 임베딩한다.
# 루트의 test_hf_one_pdf.py와 같은 기법(모듈 임포트 전에 OUTPUT_DIR 등을 바꿔치기)을 쓰되,
# 결과를 test/ 폴더 안에 완전히 격리해 test/output_test/만 지우면 흔적이 안 남게 한다.
# 진짜 output/과 HF의 embedding/ 폴더는 전혀 건드리지 않는다 (HF 업로드도 생략).
#
# 텍스트 전용: 비전 API(gpt-5-mini)를 아예 호출하지 않는 text_only_pdf_reader.TextOnlyPDFReader로
# main.py의 리더를 바꿔치기한다 — 로컬에 GPU가 없고, 이미지를 LLM으로 처리하는 단계에서 문제가 생길 수
# 있는 환경(Windows Smart App Control 등)에서도 안전하게 순수 텍스트만으로 임베딩 배관을 확인하기 위함.
# 스캔 이미지 위주 페이지는 텍스트가 거의 없이 인덱싱되지만, 멀티에이전트 배관 테스트 목적이라 무방하다.
# 임베딩 자체(BGE-M3)는 DEVICE 자동 감지로 CPU에서 그대로 동작한다.
#
# 주의: PDF 자체는 HF_REPO_ID에서 전체 목록을 동기화(snapshot_download)한 뒤 그중 1개만 처리한다
# (root의 test_hf_one_pdf.py와 동일한 동작 — 첫 실행 시 PDF 전체 다운로드 시간이 걸릴 수 있음).
#
# 사용법 (first_project/ 에서 실행):
#   python test/embed_one_pdf.py            # PDF 1개만 처리 (텍스트만, 비전 API 호출 없음)
#   python test/embed_one_pdf.py --limit 2  # 2개 처리
#   python test/embed_one_pdf.py --clean    # test/output_test/ 삭제 (hf_pdfs 캐시는 공용이라 그대로 둠)

import argparse
import shutil
import sys
from pathlib import Path

# Windows 콘솔의 기본 코드페이지(cp949)는 em-dash 등 일부 유니코드 문자를 인코딩하지 못해 print()가
# 죽을 수 있다 (PDF 파일명·추출된 본문 등에 흔히 섞여 나옴). 출력 인코딩을 UTF-8로 강제한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TEST_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TEST_DIR.parent
TEST_OUTPUT_DIR = TEST_DIR / "output_test"

sys.path.insert(0, str(PROJECT_DIR))  # config/main 등 루트 모듈을 임포트하기 위함


def clean() -> None:
    """테스트 산출물을 삭제해 원래 상태로 되돌린다."""
    if TEST_OUTPUT_DIR.exists():
        shutil.rmtree(TEST_OUTPUT_DIR)
        print(f"[삭제 완료] {TEST_OUTPUT_DIR}")
    else:
        print(f"[이미 없음] {TEST_OUTPUT_DIR}")
    print("[원상복구 완료] 실제 output/ 폴더와 HF repo는 처음부터 건드리지 않았습니다.")
    print("(PDF 캐시 hf_pdfs/ 는 다른 테스트와 공유하는 폴더라 그대로 뒀습니다.)")


def run(limit: int) -> None:
    # 무거운 모듈(main -> BGE-M3 로딩)을 import 하기 전에 config의 출력 경로를 먼저 바꿔치기한다.
    import config

    config.OUTPUT_DIR = str(TEST_OUTPUT_DIR)

    import checkpoint

    # checkpoint.py는 import 시점에 경로를 계산하므로, 테스트 폴더를 보도록 다시 지정한다.
    checkpoint.CHECKPOINT_PATH = TEST_OUTPUT_DIR / "checkpoint.json"

    import main as pipeline

    pipeline.OUTPUT_DIR = str(TEST_OUTPUT_DIR)
    # main.py 마지막의 upload_output_to_hf() 호출을 막는다. (HF에서 PDF를 받아오는 sync는 정상 동작)
    pipeline.HF_REPO_ID = None

    # main.py는 `reader = VisionPDFReader()`처럼 모듈 전역 이름을 호출 시점에 조회하므로,
    # 그 이름 자체를 텍스트 전용 리더로 바꿔치기하면 비전 API를 전혀 타지 않는다.
    from text_only_pdf_reader import TextOnlyPDFReader

    pipeline.VisionPDFReader = TextOnlyPDFReader

    print(f"[테스트 모드] 결과 저장 위치: {TEST_OUTPUT_DIR} (실제 output/은 사용하지 않음)")
    print("[테스트 모드] 처리 완료 후 HF 업로드는 생략됩니다 (진짜 embedding/ 폴더는 안 건드림).")
    print("[테스트 모드] 비전 API 호출 없음 - 텍스트 레이어만 추출합니다 (스캔 페이지는 텍스트가 거의 없을 수 있음).")

    sys.argv = ["main.py", "--use-hf", "--limit", str(limit)]
    pipeline.main()

    print()
    print(f"[임베딩 완료] {TEST_OUTPUT_DIR}")
    print("다음: python test/run_scenarios.py 로 멀티에이전트 시나리오 테스트 실행")


def cli() -> None:
    parser = argparse.ArgumentParser(description="멀티에이전트 테스트용 PDF 1개 격리 임베딩")
    parser.add_argument("--limit", type=int, default=1, help="처리할 PDF 개수 (기본 1)")
    parser.add_argument("--clean", action="store_true", help="test/output_test/ 삭제")
    args = parser.parse_args()

    if args.clean:
        clean()
    else:
        run(args.limit)


if __name__ == "__main__":
    cli()
