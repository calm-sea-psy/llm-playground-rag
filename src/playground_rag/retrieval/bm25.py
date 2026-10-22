"""BM25 키워드 검색기. 청크를 Kiwi 토큰으로 색인해 numpy 역색인으로 점수를 낸다."""

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from playground_corpus import Chunk

from playground_rag.core import Filters, Hit
from playground_rag.retrieval.analyzer import KiwiAnalyzer

K1 = 1.2  # Lucene 기본값. 단어 빈도가 점수에 미치는 포화 정도
B = 0.75  # Lucene 기본값. 청크 길이 보정 강도


class Bm25Retriever:
    """청크 목록을 메모리에 색인하는 BM25 검색기 (서브셋 규모용).

    rank_bm25 는 질의 단어마다 모든 문서를 파이썬으로 훑어 느리다. 단어 → (청크 번호, 미리 계산한 가중치)
    역색인을 두고 질의 단어가 나오는 청크만 더한다. idf 는 Lucene 식 ln(1 + (N − df + 0.5) / (df + 0.5)).

    예)
        retriever = Bm25Retriever(chunks, KiwiAnalyzer())
        hits = retriever.search("조선의 수도는?", k=5)
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        analyzer: KiwiAnalyzer,
        *,
        k1: float = K1,
        b: float = B,
        doc_categories: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._analyzer = analyzer
        self._doc_categories = doc_categories or {}
        self._postings = _build_postings(
            analyzer.tokens_batch([c.text for c in self._chunks]), k1, b
        )
        self._doc_ids = np.array([chunk.doc_id for chunk in self._chunks])

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        """질의와 BM25 점수가 높은 청크 k건을 점수 내림차순으로 돌려준다. 겹치는 단어가 없는 청크는 빼고."""
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k=}")
        scores = np.zeros(len(self._chunks), dtype=np.float32)
        for term, count in Counter(self._analyzer.tokens(query)).items():
            posting = self._postings.get(term)
            if posting is not None:
                scores[posting[0]] += count * posting[1]
        if filters is not None:
            scores[~self._allowed(filters)] = 0.0
        candidates = np.flatnonzero(scores > 0)
        if len(candidates) > k:
            candidates = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [Hit(chunk=self._chunks[i], score=float(scores[i])) for i in ordered]

    def _allowed(self, filters: Filters) -> NDArray[np.bool_]:
        mask = np.ones(len(self._chunks), dtype=bool)
        if filters.doc_ids:
            mask &= np.isin(self._doc_ids, list(filters.doc_ids))
        if filters.categories:
            allowed_docs = [
                doc for doc, cats in self._doc_categories.items() if cats & filters.categories
            ]
            mask &= np.isin(self._doc_ids, allowed_docs)
        return mask


def _build_postings(
    tokenized: Sequence[Sequence[str]], k1: float, b: float
) -> dict[str, tuple[NDArray[np.int32], NDArray[np.float32]]]:
    """단어 → (그 단어가 나오는 청크 번호, 청크별 BM25 가중치 = idf × tf 포화 · 길이 보정)."""
    lengths = np.array([len(tokens) for tokens in tokenized], dtype=np.float32)
    average = float(lengths.mean()) if len(lengths) else 0.0
    by_term: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, tokens in enumerate(tokenized):
        for term, tf in Counter(tokens).items():
            by_term[term].append((index, tf))
    total = len(tokenized)
    postings: dict[str, tuple[NDArray[np.int32], NDArray[np.float32]]] = {}
    for term, pairs in by_term.items():
        docs = np.array([doc for doc, _ in pairs], dtype=np.int32)
        tf = np.array([count for _, count in pairs], dtype=np.float32)
        idf = math.log(1 + (total - len(pairs) + 0.5) / (len(pairs) + 0.5))
        norm = k1 * (1 - b + b * lengths[docs] / average)
        postings[term] = (docs, (idf * tf * (k1 + 1) / (tf + norm)).astype(np.float32))
    return postings
