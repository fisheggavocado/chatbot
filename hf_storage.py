# Hugging Face Hub 연동 모듈
# - upload_pdfs_to_hf(): 로컬 PDF 폴더를 HF Dataset repo에 업로드
# - sync_pdfs_from_hf(): HF Dataset repo의 PDF를 로컬로 내려받아 PDF_DIR처럼 사용
# - upload_output_to_hf(): 로컬 인덱스(OUTPUT_DIR)를 HF Dataset repo의 embedding/ 폴더에 업로드
# - upload_checkpoint_to_hf() / restore_checkpoint_from_hf(): consultant_bot의 대화 체크포인트(sqlite)를
#   HF Dataset repo의 checkpoints/ 폴더와 주고받음 (embedding/과는 별도 경로)
#
# 사용법:
#   python hf_storage.py --upload          # PDF_DIR -> HF_REPO_ID 업로드
#   python hf_storage.py --sync            # HF_REPO_ID -> HF_LOCAL_SYNC_DIR 로 동기화
#   python hf_storage.py --upload-output   # OUTPUT_DIR -> HF_REPO_ID/embedding 업로드

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, snapshot_download

from config import HF_LOCAL_SYNC_DIR, HF_REPO_ID, HF_TOKEN, OUTPUT_DIR, PDF_DIR

OUTPUT_PATH_IN_REPO = "embedding"
CHECKPOINT_PATH_IN_REPO = "checkpoints"


def upload_pdfs_to_hf(local_dir: str = PDF_DIR) -> None:
    """local_dir의 모든 PDF를 HF_REPO_ID(Dataset repo)에 업로드한다. repo가 없으면 새로 만든다."""
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID가 설정되지 않았습니다. .env에 HF_REPO_ID=사용자명/저장소명 을 추가해주세요.")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=["*.pdf"],
    )
    print(f"[업로드 완료] '{local_dir}' -> https://huggingface.co/datasets/{HF_REPO_ID}")


def upload_output_to_hf(output_dir: str = OUTPUT_DIR) -> None:
    """OUTPUT_DIR의 인덱스 파일(docstore/vector_store/checkpoint 등)을 HF_REPO_ID의
    embedding/ 폴더에 업로드한다. repo가 없으면 새로 만든다.
    """
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID가 설정되지 않았습니다. .env에 HF_REPO_ID=사용자명/저장소명 을 추가해주세요.")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True, private=True)
    api.upload_folder(
        folder_path=output_dir,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        path_in_repo=OUTPUT_PATH_IN_REPO,
    )
    print(
        f"[업로드 완료] '{output_dir}' -> "
        f"https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{OUTPUT_PATH_IN_REPO}"
    )


def restore_output_from_hf(output_dir: str = OUTPUT_DIR) -> bool:
    """HF repo의 embedding/ 폴더(이전 실행 백업)를 output_dir로 복원한다.

    백업이 없으면(첫 실행) False를 반환하고 아무것도 하지 않는다.
    컨테이너처럼 로컬 디스크가 휘발성인 환경에서, 재시작 시 이전 진행분을 이어받기 위해 사용한다.
    """
    if not HF_REPO_ID:
        return False
    try:
        files = HfApi(token=HF_TOKEN).list_repo_files(HF_REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"[복원 건너뜀] HF repo 조회 실패: {e}")
        return False
    if not any(f.startswith(f"{OUTPUT_PATH_IN_REPO}/") for f in files):
        return False

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=tmp,
            token=HF_TOKEN,
            allow_patterns=[f"{OUTPUT_PATH_IN_REPO}/*"],
        )
        src = Path(tmp) / OUTPUT_PATH_IN_REPO
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            shutil.copy2(item, Path(output_dir) / item.name)
    print(f"[복원 완료] {HF_REPO_ID}/{OUTPUT_PATH_IN_REPO} -> '{output_dir}'")
    return True


def _checkpoint_files(checkpoint_path: Path) -> List[Path]:
    """sqlite는 WAL 모드에서 -wal/-shm 사이드카 파일에 아직 메인 파일로 안 옮겨진 데이터가 남을 수 있어
    존재하는 파일을 모두 함께 다룬다 (본 파일 + -wal + -shm)."""
    candidates = [
        checkpoint_path,
        checkpoint_path.with_name(checkpoint_path.name + "-wal"),
        checkpoint_path.with_name(checkpoint_path.name + "-shm"),
    ]
    return [p for p in candidates if p.exists()]


def upload_checkpoint_to_hf(checkpoint_path) -> None:
    """consultant_bot의 체크포인트 sqlite 파일(및 WAL 사이드카)을 HF_REPO_ID의 checkpoints/ 폴더에 업로드한다.

    WizardState(대화 이력·evidence·stage 등)가 담긴 파일이므로, embedding/ 인덱스와는 별도 경로에 둔다.
    """
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID가 설정되지 않았습니다. .env에 HF_REPO_ID=사용자명/저장소명 을 추가해주세요.")
    checkpoint_path = Path(checkpoint_path)
    files = _checkpoint_files(checkpoint_path)
    if not files:
        return  # 아직 로컬에 체크포인트가 생성되지 않았으면 할 일 없음

    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True, private=True)
    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"{CHECKPOINT_PATH_IN_REPO}/{f.name}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
    print(
        f"[체크포인트 업로드 완료] '{checkpoint_path}' -> "
        f"https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{CHECKPOINT_PATH_IN_REPO}"
    )


def restore_checkpoint_from_hf(checkpoint_path) -> bool:
    """HF repo의 checkpoints/ 폴더(이전 세션 백업)를 checkpoint_path 자리로 복원한다.

    백업이 없으면(첫 실행) False를 반환하고 아무것도 하지 않는다. 휘발성 컨테이너에서 재시작해도
    이전 대화(WizardState)를 이어받기 위해 사용한다.
    """
    if not HF_REPO_ID:
        return False
    checkpoint_path = Path(checkpoint_path)
    try:
        files = HfApi(token=HF_TOKEN).list_repo_files(HF_REPO_ID, repo_type="dataset")
    except Exception as e:
        print(f"[체크포인트 복원 건너뜀] HF repo 조회 실패: {e}")
        return False
    if not any(f.startswith(f"{CHECKPOINT_PATH_IN_REPO}/") for f in files):
        return False

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=tmp,
            token=HF_TOKEN,
            allow_patterns=[f"{CHECKPOINT_PATH_IN_REPO}/*"],
        )
        src = Path(tmp) / CHECKPOINT_PATH_IN_REPO
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            shutil.copy2(item, checkpoint_path.parent / item.name)
    print(f"[체크포인트 복원 완료] {HF_REPO_ID}/{CHECKPOINT_PATH_IN_REPO} -> '{checkpoint_path.parent}'")
    return True


def sync_pdfs_from_hf() -> str:
    """HF_REPO_ID(Dataset repo)의 PDF를 HF_LOCAL_SYNC_DIR로 내려받고 로컬 경로를 반환한다.

    이미 내려받은 파일은 snapshot_download의 캐시 덕분에 다시 받지 않는다.
    """
    if not HF_REPO_ID:
        raise ValueError("HF_REPO_ID가 설정되지 않았습니다. .env에 HF_REPO_ID=사용자명/저장소명 을 추가해주세요.")
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=HF_LOCAL_SYNC_DIR,
        token=HF_TOKEN,
        allow_patterns=["*.pdf"],
    )
    print(f"[동기화 완료] {HF_REPO_ID} -> '{local_dir}'")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="PDF/인덱스를 Hugging Face Dataset repo와 주고받는다.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--upload", action="store_true", help="PDF_DIR의 PDF를 HF_REPO_ID에 업로드")
    group.add_argument("--sync", action="store_true", help="HF_REPO_ID의 PDF를 로컬로 동기화")
    group.add_argument(
        "--upload-output", action="store_true", help="OUTPUT_DIR의 인덱스를 HF_REPO_ID/embedding에 업로드"
    )
    args = parser.parse_args()

    if args.upload:
        upload_pdfs_to_hf()
    elif args.upload_output:
        upload_output_to_hf()
    else:
        sync_pdfs_from_hf()


if __name__ == "__main__":
    main()
