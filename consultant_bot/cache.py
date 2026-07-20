# 데이터 유실 방지 + 토큰 절약을 위한 두 종류의 디스크 캐시.
#
#   SearchCache - hybrid_search(query) 결과를 정확 일치(exact query) 기준으로 캐싱.
#                 재검색 compute를 아끼지만, 이것만으로는 LLM 토큰이 줄지 않는다.
#   FaqCache    - FAQ 질문/답변을 BGE-M3 임베딩 유사도로 캐싱. 비슷한 질문이 재입력되면
#                 검색(react_loop) + LLM 호출을 아예 건너뛰므로 토큰 절약 효과가 가장 크다.
#                 임베딩은 이미 로컬에 로드된 BGE-M3를 재사용하므로 캐시 조회 자체엔 추가 비용이 없다.
#
# 둘 다 OUTPUT_DIR 아래 JSON 파일에 즉시(write-through) 저장해, 프로세스가 죽어도 캐시가 남는다.

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
FAQ_SIMILARITY_THRESHOLD = 0.92  # 이 이상 코사인 유사도면 캐시 히트로 간주 (조정 가능한 품질 파라미터)


class SearchCache:
    """hybrid_search 결과의 정확 일치 캐시."""

    def __init__(self, path: Path = SEARCH_CACHE_PATH):
        self._path = path
        self._data: dict = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    @staticmethod
    def _key(query: str) -> str:
        return " ".join(query.strip().lower().split())

    def get(self, query: str) -> Optional[list]:
        hit = self._data.get(self._key(query))
        return list(hit) if hit is not None else None

    def put(self, query: str, evidence: list) -> None:
        self._data[self._key(query)] = [dict(e) for e in evidence]
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)


class FaqCache:
    """FAQ 질문-답변의 임베딩 유사도 캐시."""

    def __init__(self, path: Path = FAQ_CACHE_PATH, threshold: float = FAQ_SIMILARITY_THRESHOLD):
        self._path = path
        self._threshold = threshold
        self._entries: list[dict] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self._entries = json.load(f)

    def lookup(self, question: str) -> Optional[dict]:
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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)


search_cache = SearchCache()
faq_cache = FaqCache()
