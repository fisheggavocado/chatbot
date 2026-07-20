# Research Worker / FAQ Agent가 공유하는 hybrid_search 도구.
# verify_embeddings.py의 build_retriever(mode="hybrid") 패턴(벡터+BM25 RRF 융합 -> cross-encoder rerank)을 그대로 재사용한다.
#
# "hugging face에 임베딩된 파일" 활용 지점: OUTPUT_DIR에 로컬 인덱스가 없으면,
# HF_REPO_ID(Dataset repo)의 embedding/ 폴더에서 hf_storage.restore_output_from_hf()로 자동 복원한다.

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool  # noqa: E402
from llama_index.core import Settings, StorageContext, load_index_from_storage  # noqa: E402
from llama_index.core.llms import MockLLM  # noqa: E402
from llama_index.core.retrievers import QueryFusionRetriever  # noqa: E402

from config import CANDIDATE_TOP_K, HF_REPO_ID, OUTPUT_DIR, RERANK_TOP_K  # noqa: E402
from hf_storage import restore_output_from_hf  # noqa: E402
from llama_bm25_retriever import KiwiBM25Retriever  # noqa: E402
from llama_embedding import BGEM3Embedding  # noqa: E402
from reranker import rerank as cross_encoder_rerank  # noqa: E402

from cache import search_cache  # noqa: E402
from state import Evidence  # noqa: E402

# 인덱스를 저장할 때와 동일한 임베딩 모델을 지정해야 질의 임베딩 차원이 맞는다.
Settings.embed_model = BGEM3Embedding()

# QueryFusionRetriever는 num_queries=1(질의 재생성 없음)이라도 생성자에서 Settings.llm을 즉시 참조한다.
# 명시적으로 지정 안 하면 llama_index가 기본 OpenAI LLM을 자동 임포트하려다 llama-index-llms-openai
# 패키지가 없어 ImportError로 죽는다. 실제로 호출되지는 않으므로(질의 재생성을 안 하니) 진짜 LLM 대신
# MockLLM로 채워 그 자동 해석 자체를 건너뛴다 (실제 답변 생성은 llm.py의 langchain ChatOpenAI가 담당).
Settings.llm = MockLLM()

_index = None
_retriever = None

# --- 품질 튜닝 파라미터 (필요에 따라 조정) ---
NUM_QUERY_EXPANSIONS = 1  # 1=원 질의만 사용. 2~3으로 올리면 LLM이 질의를 변형해 여러 각도로 검색(재현율↑, LLM 호출 비용↑)


def _ensure_index_available() -> None:
    """OUTPUT_DIR에 로컬 인덱스가 없으면 HF Dataset(embedding/ 폴더)에서 복원을 시도한다."""
    if (Path(OUTPUT_DIR) / "docstore.json").exists():
        return
    if not HF_REPO_ID:
        raise RuntimeError(
            f"'{OUTPUT_DIR}'에 인덱스가 없고 HF_REPO_ID도 설정되지 않았습니다. "
            "먼저 python main.py로 인덱스를 만들거나, .env에 HF_REPO_ID를 설정하세요."
        )
    print(f"[retrieval] 로컬 인덱스 없음 - Hugging Face Dataset({HF_REPO_ID})에서 복원 시도...")
    restored = restore_output_from_hf(OUTPUT_DIR)
    if not restored:
        raise RuntimeError(
            f"Hugging Face Dataset({HF_REPO_ID})에 embedding/ 백업이 없습니다. "
            "먼저 python main.py --use-hf 로 인덱스를 만들고 업로드하세요."
        )


def _get_retriever() -> QueryFusionRetriever:
    global _index, _retriever
    if _retriever is not None:
        return _retriever

    _ensure_index_available()
    storage_context = StorageContext.from_defaults(persist_dir=str(OUTPUT_DIR))
    _index = load_index_from_storage(storage_context)

    all_nodes = list(_index.docstore.docs.values())
    vector_retriever = _index.as_retriever(similarity_top_k=CANDIDATE_TOP_K)
    bm25_retriever = KiwiBM25Retriever(nodes=all_nodes, similarity_top_k=CANDIDATE_TOP_K)
    _retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=CANDIDATE_TOP_K,
        num_queries=NUM_QUERY_EXPANSIONS,
        mode="reciprocal_rerank",  # RRF 융합
        use_async=False,
    )
    return _retriever


def hybrid_search(query: str, top_k: int = RERANK_TOP_K) -> List[Evidence]:
    """bi-encoder(BGE-M3)+BM25를 RRF로 융합한 후보를 cross-encoder로 재정렬해 top_k개의 근거를 반환한다.

    동일 쿼리(정규화 기준)로 이전에 검색한 적이 있으면 디스크 캐시(cache.SearchCache)에서 즉시 반환한다
    — 세션이 끊겼다 재개되거나, 다른 노드가 같은 문구로 검색할 때 재계산을 피한다.
    """
    cached = search_cache.get(query)
    if cached is not None:
        print(f"[retrieval] 캐시 히트: '{query}' -> {len(cached)}건 (재검색 생략)")
        return [Evidence(**e) for e in cached[:top_k]]

    retriever = _get_retriever()
    candidates = retriever.retrieve(query)
    results = cross_encoder_rerank(query, candidates, top_k)

    evidence: List[Evidence] = []
    for node_with_score in results:
        node = node_with_score.node
        evidence.append(
            Evidence(
                text=node.get_content(),
                source=node.metadata.get("source", "unknown"),
                page=node.metadata.get("page", "?"),
                score=float(node_with_score.score),
            )
        )

    search_cache.put(query, evidence)
    return evidence


@tool
def hybrid_search_tool(query: str) -> List[dict]:
    """강의 PDF에서 질의와 관련된 근거를 하이브리드(벡터+BM25, RRF)로 검색하고 cross-encoder로 재정렬해 반환한다."""
    return [dict(e) for e in hybrid_search(query)]
