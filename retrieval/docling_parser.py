"""Compatibility wrapper for the original Member 2 Docling parser.

New code should use ``retrieval.ingestion``. This module keeps the historical
``DoclingDocumentParser`` and ``DocumentChunk`` import paths working while
delegating real parsing to the Docling adapter and structure-aware chunker.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from retrieval.config import RetrievalSettings
from retrieval.ingestion.chunker import chunk_blocks
from retrieval.ingestion.docling_parser import DoclingParser
from retrieval.schemas import DocumentChunk, validate_workspace_id


class DoclingDocumentParser:
    """Backward-compatible parser facade.

    The old class accepted a token-sized ``target_chunk_size`` and ``overlap``.
    Those values are converted to approximate character budgets before being
    passed to the structure-aware chunker.
    """

    def __init__(
        self,
        target_chunk_size: int = 512,
        overlap: int = 64,
        *,
        settings: Optional[RetrievalSettings] = None,
        converter: object = None,
    ):
        self.chunk_size = target_chunk_size
        self.overlap = overlap
        target_chars = max(200, int(target_chunk_size) * 4)
        overlap_chars = max(0, int(overlap) * 4)
        if overlap_chars >= target_chars:
            overlap_chars = max(0, target_chars // 4)
        max_chars = max(target_chars, int(target_chars * 1.5))
        self._settings = settings or RetrievalSettings.from_env(
            env_file=False,
            overrides={
                "chunk_target_chars": target_chars,
                "chunk_max_chars": max_chars,
                "chunk_overlap_chars": overlap_chars,
            },
        )
        self._parser = DoclingParser(settings=self._settings, converter=converter)

    def parse_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        workspace_id: str = "default_workspace",
    ) -> List[DocumentChunk]:
        """Parse a document through Docling and return legacy-compatible chunks."""
        tenant = validate_workspace_id(workspace_id)
        base_id = doc_id or Path(file_path).stem
        parsed = self._parser.parse(file_path, tenant)
        source_type = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".md": "markdown",
            ".markdown": "markdown",
        }.get(Path(file_path).suffix.lower(), "document")
        return chunk_blocks(
            parsed.blocks,
            doc_id=base_id,
            workspace_id=tenant,
            doc_title=parsed.doc_title,
            target_chars=self._settings.chunk_target_chars,
            max_chars=self._settings.chunk_max_chars,
            overlap_chars=self._settings.chunk_overlap_chars,
            source_type=source_type,
        )

    def _generate_simulated_chunks(self, file_path: str, doc_id: str) -> List[DocumentChunk]:
        """Deprecated deterministic fixtures retained for old tests/tooling.

        This helper is never called by ``parse_document`` and must not be used in
        production ingestion.
        """
        if "qa_sop" in file_path.lower():
            return [
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_001",
                    doc_id=doc_id,
                    doc_title="QA_SOP_2026.pdf",
                    content=(
                        "Section 4.2 Quality Standards: The maximum defect tolerance "
                        "percentage allowed for lithium cell batches is strictly 0.5%. "
                        "Any batch exceeding 0.5% defect rate must be quarantined immediately."
                    ),
                    page_number=12,
                    char_count=210,
                    token_count_est=52,
                )
            ]
        if "sla" in file_path.lower() or "voltcore" in file_path.lower():
            return [
                DocumentChunk(
                    chunk_id=f"{doc_id}_chunk_001",
                    doc_id=doc_id,
                    doc_title="VoltCore_SLA.pdf",
                    content=(
                        "Clause 8.2 Supplier Delivery Penalties: Standard lead time is "
                        "14 days. Any delivery delay exceeding 7 business days triggers a "
                        "mandatory 5% late penalty fee per week capped at 25% of total "
                        "contract value."
                    ),
                    page_number=6,
                    char_count=218,
                    token_count_est=55,
                )
            ]
        return [
            DocumentChunk(
                chunk_id=f"{doc_id}_chunk_001",
                doc_id=doc_id,
                doc_title=doc_id,
                content=f"Standard policy documentation for {doc_id}.",
                page_number=1,
                char_count=50,
                token_count_est=12,
            )
        ]


__all__ = ["DoclingDocumentParser", "DocumentChunk"]
