"""Regression tests for adding texts from one-shot iterables (generators)."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing_extensions import override

from langchain_qdrant import Qdrant, QdrantVectorStore

TEXTS = ["alpha", "beta", "gamma", "delta", "epsilon"]


class _FakeEmbeddings(Embeddings):
    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, float(i)] for i, _ in enumerate(texts)]

    @override
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _make_client() -> QdrantClient:
    client = QdrantClient(":memory:")
    client.create_collection(
        "test", vectors_config=VectorParams(size=2, distance=Distance.COSINE)
    )
    return client


def _stored_texts(client: QdrantClient) -> list[str]:
    points, _ = client.scroll("test", limit=100, with_payload=True)
    payloads = [point.payload for point in points]
    assert all(payload is not None for payload in payloads)
    return [str(payload["page_content"]) for payload in payloads if payload]


@pytest.mark.parametrize("batch_size", [2, 64])
def test_qdrant_vector_store_add_texts_from_generator(batch_size: int) -> None:
    client = _make_client()
    store = QdrantVectorStore(
        client=client, collection_name="test", embedding=_FakeEmbeddings()
    )

    ids = store.add_texts((t for t in TEXTS), batch_size=batch_size)

    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert sorted(_stored_texts(client)) == sorted(TEXTS)


def test_qdrant_vector_store_add_texts_from_generator_with_ids() -> None:
    client = _make_client()
    store = QdrantVectorStore(
        client=client, collection_name="test", embedding=_FakeEmbeddings()
    )

    ids = store.add_texts((t for t in TEXTS), ids=[1, 2, 3, 4, 5], batch_size=2)

    assert ids == [1, 2, 3, 4, 5]
    assert sorted(_stored_texts(client)) == sorted(TEXTS)


@pytest.mark.parametrize("batch_size", [2, 64])
def test_legacy_qdrant_add_texts_from_generator(batch_size: int) -> None:
    client = _make_client()
    store = Qdrant(client=client, collection_name="test", embeddings=_FakeEmbeddings())

    ids = store.add_texts((t for t in TEXTS), batch_size=batch_size)

    assert len(ids) == 5
    assert len(set(ids)) == 5
    assert sorted(_stored_texts(client)) == sorted(TEXTS)


async def test_legacy_qdrant_async_batches_from_generator() -> None:
    # Local mode can't share a collection between sync and async clients, so
    # only the batch generator is exercised here.
    store = Qdrant(
        client=QdrantClient(":memory:"),
        collection_name="test",
        embeddings=_FakeEmbeddings(),
        async_client=AsyncQdrantClient(":memory:"),
    )

    all_ids: list[str] = []
    payload_texts: list[str] = []
    async for batch_ids, points in store._agenerate_rest_batches(
        (t for t in TEXTS), batch_size=2
    ):
        all_ids.extend(batch_ids)
        for point in points:
            assert point.payload is not None
            payload_texts.append(str(point.payload["page_content"]))

    assert len(all_ids) == 5
    assert len(set(all_ids)) == 5
    assert payload_texts == TEXTS
