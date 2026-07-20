# 데이터 유실 방지 + 토큰 절약을 위한 두 종류의 디스크 캐시.
#
#   SearchCache - hybrid_search(query) 결과를 정확 일치(exact query) 기준으로 캐싱.
#                 재검색 compute를 아끼지만, 이것만으로는 LLM 토큰이 줄지 않는다.
#   FaqCache    - FAQ 질문/답변을 BGE-M3 임베딩 유사도로 캐싱. 비슷한 질문이 재입력되면
#                 검색(react_loop) + LLM 호출을 아예 건너뛰므로 토큰 절약 효과가 가장 크다.
#                 임베딩은 이미 로컬에 로드된 BGE-M3를 재사용하므로 캐시 조회 자체엔 추가 비용이 없다.
#
# 둘 다 OUTPUT_DIR 아래 JSON 파일에 즉시(write-through) 저장해, 프로세스가 죽어도 캐시가 남는다.
#
# 무효화(invalidation) — 두 겹으로 처리한다:
#   1) 인덱스 버전 감지 - OUTPUT_DIR의 인덱스 파일(docstore.json 등) mtime/size로 지문을 만들어
#      cache_meta.json에 저장해두고, 프로세스 시작 시 지문이 바뀌었으면(재임베딩 발생) 두 캐시 파일을
#      통째로 비운다. PDF 자료가 갱신됐는데 과거 캐시가 그대로 재사용되는 것을 막는다.
#   2) TTL 만료 - 지문이 그대로여도(재임베딩 없이도) 오래된 항목은 CACHE_TTL_SECONDS가 지나면
#      조회 시점에 즉시 폐기한다(lazy eviction). 두 메커니즘은 서로 다른 문제를 막는다: 1)은 "자료 자체가
#      바뀜"을, 2)는 "자료는 그대로지만 답변이 낡음(예: 프롬프트/가드레일 기준 변경)"을 커버한다.

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from config import OUTPUT_DIR  # noqa: E402
from embedder import embed_texts  # noqa: E402 - BGE-M3, 로컬 (API 비용 없음)

from state import Evidence  # noqa: E402

SEARCH_CACHE_PATH = Path(OUTPUT_DIR) / "search_cache.json"
FAQ_CACHE_PATH = Path(OUTPUT_DIR) / "faq_cache.json"
CACHE_META_PATH = Path(OUTPUT_DIR) / "cache_meta.json"
FAQ_SIMILARITY_THRESHOLD = 0.92  # 이 이상 코사인 유사도면 캐시 히트로 간주 (조정 가능한 품질 파라미터)
CACHE_TTL_SECONDS = 60 * 60 * 24 * 14  # 14일 - 이보다 오래된 캐시 항목은 조회 시 폐기(조정 가능)

# 인덱스 지문을 만들 때 참고하는 파일들 (llama_index persist 산출물 중 내용이 바뀌면 반드시 갱신되는 것들).
_INDEX_FINGERPRINT_FILES = ("docstore.json", "default__vector_store.json", "index_store.json")


def _index_fingerprint() -> str:
    """OUTPUT_DIR의 현재 인덱스 상태를 나타내는 문자열 지문. 재임베딩되면 mtime/size가 바뀌어 값이 달라진다."""
    parts = []
    for name in _INDEX_FINGERPRINT_FILES:
        p = Path(OUTPUT_DIR) / name
        if p.exists():
            st = p.stat()
            parts.append(f"{name}:{st.st_mtime_ns}:{st.st_size}")
        else:
            parts.append(f"{name}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _invalidate_stale_index_cache() -> None:
    """저장된 지문과 현재 인덱스 지문이 다르면(재임베딩 발생) 검색/FAQ 캐시 파일을 모두 비운다.

    모듈 임포트 시 캐시 싱글턴을 만들기 전에 한 번만 호출한다 - 그래야 SearchCache/FaqCache의
    __init__이 이미 비워진 상태에서 파일을 읽는다.
    """
    fingerprint = _index_fingerprint()
    previous = None
    if CACHE_META_PATH.exists():
        try:
            previous = json.loads(CACHE_META_PATH.read_text(encoding="utf-8")).get("index_fingerprint")
        except (json.JSONDecodeError, OSError):
            previous = None

    if previous == fingerprint:
        return

    for path in (SEARCH_CACHE_PATH, FAQ_CACHE_PATH):
        if path.exists():
            path.unlink()

    CACHE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_META_PATH.write_text(json.dumps({"index_fingerprint": fingerprint}), encoding="utf-8")

    if previous is not None:
        print("[cache] 인덱스 변경 감지 - 검색/FAQ 캐시를 초기화했습니다.")


class SearchCache:
    """hybrid_search 결과의 정확 일치 캐시. 항목은 CACHE_TTL_SECONDS가 지나면 조회 시 폐기된다."""

    def __init__(self, path: Path = SEARCH_CACHE_PATH):
        self._path = path
        self._data: dict = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, query: str) -> Optional[list]:
        key = self._key(query)
        hit = self._data.get(key)
        if hit is None:
            return None
        if time.time() - hit.get("ts", 0) > CACHE_TTL_SECONDS:
            del self._data[key]
            self._save()
            return None
        return [dict(e) for e in hit["evidence"]]

    def put(self, query: str, evidence: list) -> None:
        self._data[self._key(query)] = {"ts": time.time(), "evidence": [dict(e) for e in evidence]}
        self._save()

    def clear(self) -> None:
        """캐시를 전부 비운다(수동 무효화용 - 예: 자료 갱신 직후 CLI에서 호출)."""
        self._data = {}
        if self._path.exists():
            self._path.unlink()


class FaqCache:
    """FAQ 질문-답변의 임베딩 유사도 캐시. 항목은 CACHE_TTL_SECONDS가 지나면 조회 시 폐기된다."""

    def __init__(self, path: Path = FAQ_CACHE_PATH, threshold: float = FAQ_SIMILARITY_THRESHOLD):
        self._path = path
        self._threshold = threshold
        self._entries: list[dict] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)

    def _evict_expired(self) -> None:
        now = time.time()
        fresh = [e for e in self._entries if now - e.get("ts", now) <= CACHE_TTL_SECONDS]
        if len(fresh) != len(self._entries):
            self._entries = fresh
            self._save()

    def lookup(self, question: str) -> Optional[dict]:
        self._evict_expired()
        if not self._entries:
            return None
        query_vec = np.array(embed_texts([question])[0])
        query_norm = np.linalg.norm(query_vec) + 1e-8

        best_entry, best_score = None, -1.0
        for entry in self._entries:
            vec = np.array(entry["embedding"])
            score = float(np.dot(query_vec, vec) / (query_norm * (np.linalg.norm(vec) + 1e-8)))
            if score > best_score:
                best_score, best_entry = score, entry

        if best_entry is not None and best_score >= self._threshold:
            return {"answer": best_entry["answer"], "evidence": best_entry["evidence"], "score": best_score}
        return None

    def store(self, question: str, answer: str, evidence: list[Evidence]) -> None:
        vec = embed_texts([question])[0]
        self._entries.append(
            {
                "question": question,
                "embedding": vec,
                "answer": answer,
                "evidence": [dict(e) for e in evidence],
                "ts": time.time(),
            }
        )
        self._save()

    def clear(self) -> None:
        """캐시를 전부 비운다(수동 무효화용 - 예: 자료 갱신 직후 CLI에서 호출)."""
        self._entries = []
        if self._path.exists():
            self._path.unlink()


def invalidate_all() -> None:
    """검색/FAQ 캐시를 즉시 모두 비운다. 재임베딩 파이프라인(main.py 등)이나 운영 CLI에서 수동 호출용."""
    search_cache.clear()
    faq_cache.clear()
    if CACHE_META_PATH.exists():
        CACHE_META_PATH.unlink()
    print("[cache] 검색/FAQ 캐시를 수동으로 초기화했습니다.")


_invalidate_stale_index_cache()

search_cache = SearchCache()
faq_cache = FaqCache()
