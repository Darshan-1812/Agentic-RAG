"""Live five-candidate confidence integration test with real BGE models.

This test is opt-in because it requires the live ParadeDB PostgreSQL service and
downloads/loads the real BGE-M3 and BGE reranker models:

    $env:RETRIEVAL_TEST_DATABASE_URL='postgresql://postgres:postgres@localhost:5432/calibrag'
    $env:RETRIEVAL_RUN_REAL_MODELS='1'
    python -m pytest tests/test_retrieval_confidence_live.py -q -s

It uses no mocked embedding or reranking components.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import numpy as np
import pytest

from core.config import DEFAULT_POLICY_PROFILES, SLAPolicyMode
from core.schemas import RetrievalTier
from retrieval.config import RetrievalSettings
from retrieval.db.connection import Database
from retrieval.db.migrate import apply_migrations
from retrieval.db.repository import IngestionRepository
from retrieval.embeddings import BgeM3Embedder
from retrieval.reranking import BgeReranker
from retrieval.schemas import DocumentChunk, DocumentMetadata, compute_document_id
from retrieval.service import RetrievalService
from retrieval.vector import PGVectorStore
from verifier.confidence_gate import (
    compute_entity_coverage,
    compute_evidence_confidence,
    sigmoid,
)


_TEST_DSN = os.getenv("RETRIEVAL_TEST_DATABASE_URL")
_REAL_MODELS = os.getenv("RETRIEVAL_RUN_REAL_MODELS") == "1"


def _document_with_chunks(workspace_id, suffix, contents):
    content_hash = uuid.uuid4().hex + uuid.uuid4().hex
    doc_id = compute_document_id(workspace_id, content_hash)
    document = DocumentMetadata(
        doc_id=doc_id,
        workspace_id=workspace_id,
        filename=f"confidence-{suffix}.md",
        doc_title=f"Confidence Policy {suffix}",
        mime_type="text/markdown",
        content_hash=content_hash,
        byte_size=sum(len(content) for content in contents),
        page_count=1,
        storage_path=f"test/{workspace_id}/confidence-{suffix}.md",
        metadata={"suffix": suffix},
    )
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}:c{index:05d}",
            doc_id=doc_id,
            doc_title=document.doc_title,
            content=content,
            page_number=1,
            section_hierarchy=["Confidence"],
            char_count=len(content),
            token_count_est=max(1, len(content.split())),
            workspace_id=workspace_id,
            chunk_index=index,
            source_type="markdown",
            metadata={"suffix": suffix, "index": index},
        )
        for index, content in enumerate(contents)
    ]
    return document, chunks


@pytest.mark.skipif(
    not _TEST_DSN,
    reason="RETRIEVAL_TEST_DATABASE_URL is not set",
)
@pytest.mark.skipif(
    not _REAL_MODELS,
    reason="RETRIEVAL_RUN_REAL_MODELS is not set to 1",
)
def test_live_five_candidate_confidence_integration_with_real_models():
    async def scenario():
        settings = RetrievalSettings(
            database_url=_TEST_DSN,
            embedding_device="cpu",
            reranker_device="cpu",
            embedding_batch_size=2,
            reranker_batch_size=2,
            embedding_max_length=512,
            embedding_query_max_length=512,
            reranker_max_length=512,
            retrieval_top_k=5,
        )
        database = Database(settings)
        workspace_a = f"p31_a_{uuid.uuid4().hex[:8]}"
        workspace_b = f"p31_b_{uuid.uuid4().hex[:8]}"
        await database.connect()
        try:
            await database.assert_healthy()
            await apply_migrations(database)
            repository = IngestionRepository(database)
            await repository.ensure_workspace(workspace_a)
            await repository.ensure_workspace(workspace_b)

            contents_a = [
                "EV-Pack-V2 maximum defect tolerance is 0.5 percent for lithium cell batches.",
                "EV-Pack-V2 inspection records are retained for five years.",
                "EV-Pack-V2 firmware version 4.2 requires a controller reset after installation.",
                "EV-Pack-V2 shipments use moisture-barrier packaging.",
                "EV-Pack-V2 supplier audits occur every quarter.",
            ]
            document_a, chunks_a = _document_with_chunks(
                workspace_a, "a", contents_a
            )
            document_b, chunks_b = _document_with_chunks(
                workspace_b,
                "b",
                [
                    "EV-Pack-V2 tenant B confidential note: defect tolerance is 9.9 percent."
                ],
            )
            await repository.save_document_with_chunks(document_a, chunks_a)
            await repository.save_document_with_chunks(document_b, chunks_b)

            embedder = BgeM3Embedder(settings)
            reranker = BgeReranker(settings)
            store = PGVectorStore(database, embedder, settings)
            assert await store.upsert_embeddings(workspace_a, chunks_a) == 5
            assert await store.upsert_embeddings(workspace_b, chunks_b) == 1

            service = RetrievalService(
                database,
                settings=settings,
                embedder=embedder,
                reranker=reranker,
                vector_store=store,
            )
            query = "What is the EV-Pack-V2 maximum defect tolerance?"
            evidence_a = await service.retrieve(
                query, top_k=5, workspace_id=workspace_a
            )
            evidence_b = await service.retrieve(
                query, top_k=5, workspace_id=workspace_b
            )

            assert len(evidence_a) == 5
            assert len(evidence_b) == 1
            assert {item.doc_id for item in evidence_a} == {document_a.doc_id}
            assert {item.doc_id for item in evidence_b} == {document_b.doc_id}
            assert chunks_b[0].chunk_id not in {
                item.chunk_id for item in evidence_a
            }

            logits = [item.cross_encoder_logit for item in evidence_a]
            assert all(np.isfinite(logit) for logit in logits)
            assert all(
                item.metadata["retrieval_cross_encoder_logit"]
                == pytest.approx(item.cross_encoder_logit)
                for item in evidence_a
            )

            raw_scores = reranker.score_pairs_sync(
                query, [item.content for item in evidence_a]
            )
            assert np.allclose(
                raw_scores,
                logits,
                rtol=1e-4,
                atol=1e-5,
            )
            assert not np.allclose(
                raw_scores,
                [sigmoid(float(score)) for score in raw_scores],
                rtol=1e-5,
                atol=1e-6,
            )

            sorted_logits = sorted(logits, reverse=True)
            assert logits == pytest.approx(sorted_logits)
            assert evidence_a[0].cross_encoder_logit == pytest.approx(
                sorted_logits[0]
            )
            assert sorted_logits[4] == pytest.approx(min(logits))
            expected_delta = sorted_logits[0] - sorted_logits[4]

            policy = DEFAULT_POLICY_PROFILES[SLAPolicyMode.BALANCED]
            result = compute_evidence_confidence(
                query=query,
                chunks=evidence_a,
                p_jev_noul_sufficient=0.9,
                policy=policy,
                current_tier=RetrievalTier.TIER2_HYBRID,
            )
            shuffled_result = compute_evidence_confidence(
                query=query,
                chunks=list(reversed(evidence_a)),
                p_jev_noul_sufficient=0.9,
                policy=policy,
                current_tier=RetrievalTier.TIER2_HYBRID,
            )

            expected_sigma_unrounded = sigmoid(expected_delta)
            combined_text = " ".join(
                item.content
                for item in sorted(
                    evidence_a,
                    key=lambda item: item.cross_encoder_logit,
                    reverse=True,
                )[:5]
            )
            expected_cov = compute_entity_coverage(query, combined_text)
            expected_c_unrounded = min(
                1.0,
                max(
                    0.0,
                    0.50 * 0.9
                    + 0.30 * expected_sigma_unrounded
                    + 0.20 * expected_cov,
                ),
            )
            expected_c_evidence = round(expected_c_unrounded, 4)

            assert result.sigma_delta_rerank == pytest.approx(
                round(expected_sigma_unrounded, 4)
            )
            assert result.cov_entities == pytest.approx(round(expected_cov, 4))
            assert result.c_evidence == pytest.approx(expected_c_evidence)
            assert result.p_jev_noul_sufficient == pytest.approx(0.9)
            assert result.escalate_triggered == (
                expected_c_unrounded < policy.tau_escalate
            )
            assert shuffled_result.sigma_delta_rerank == pytest.approx(
                result.sigma_delta_rerank
            )
            assert shuffled_result.c_evidence == pytest.approx(
                result.c_evidence
            )
            assert shuffled_result.cov_entities == pytest.approx(
                result.cov_entities
            )

            print("five_candidate_logits=", [round(float(x), 6) for x in logits])
            print(
                "top1=",
                round(float(sorted_logits[0]), 6),
                "fifth=",
                round(float(sorted_logits[4]), 6),
                "delta=",
                round(float(expected_delta), 6),
            )
            print(
                "sigma_delta_rerank=",
                result.sigma_delta_rerank,
                "c_evidence=",
                result.c_evidence,
            )
            print(
                "workspace_a_ids=",
                [item.chunk_id for item in evidence_a],
            )
            print(
                "workspace_b_ids=",
                [item.chunk_id for item in evidence_b],
            )
            print("real_model_confidence_integration=ok")
        finally:
            async with database.acquire() as connection:
                await connection.execute(
                    "DELETE FROM workspaces WHERE workspace_id = ANY($1::text[])",
                    [workspace_a, workspace_b],
                )
            await database.close()

    asyncio.run(scenario())
