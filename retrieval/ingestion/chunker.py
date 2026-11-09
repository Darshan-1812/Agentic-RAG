"""Structure-aware chunking for normalized Docling blocks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

from retrieval.ingestion.normalizer import (
    normalize_table_markdown,
    normalize_text,
    unique_preserving_order,
)
from retrieval.schemas import (
    DocumentChunk,
    ParsedBlock,
    compute_chunk_id,
    estimate_tokens,
    validate_workspace_id,
)


@dataclass
class _Draft:
    parts: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    hierarchy: List[str] = field(default_factory=list)
    block_types: List[str] = field(default_factory=list)
    has_table: bool = False
    table_markdown: Optional[str] = None

    @property
    def length(self) -> int:
        if not self.parts:
            return 0
        return sum(len(part) for part in self.parts) + (2 * (len(self.parts) - 1))

    def add(
        self,
        text: str,
        page_number: int,
        block_type: str,
        *,
        has_table: bool = False,
        table_markdown: Optional[str] = None,
    ) -> None:
        self.parts.append(text)
        self.pages.append(page_number)
        self.block_types.append(block_type)
        self.has_table = self.has_table or has_table
        if table_markdown:
            self.table_markdown = table_markdown


def _split_long_piece(text: str, max_chars: int) -> List[str]:
    """Split an oversized paragraph without losing content."""
    remaining = text.strip()
    pieces: List[str] = []
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        candidates = [
            window.rfind("\n\n"),
            window.rfind("\n"),
            window.rfind(". "),
            window.rfind("; "),
            window.rfind(", "),
            window.rfind(" "),
        ]
        split_at = max(candidates)
        if split_at < max_chars // 2:
            split_at = max_chars
        else:
            split_at += 1

        piece = remaining[:split_at].strip()
        if not piece:
            piece = remaining[:max_chars].strip()
            split_at = max_chars
        if piece:
            pieces.append(piece)
        remaining = remaining[split_at:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _overlap_text(parts: Sequence[str], overlap_chars: int) -> str:
    if overlap_chars <= 0 or not parts:
        return ""
    content = "\n\n".join(parts).strip()
    if len(content) <= overlap_chars:
        return content
    tail = content[-overlap_chars:]
    cut = tail.find(" ")
    if 0 <= cut < len(tail) - 1:
        tail = tail[cut + 1 :]
    return tail.strip()


def chunk_blocks(
    blocks: Iterable[ParsedBlock],
    *,
    doc_id: str,
    workspace_id: str,
    doc_title: str,
    target_chars: int = 1800,
    max_chars: int = 2600,
    overlap_chars: int = 200,
    source_type: str = "pdf",
) -> List[DocumentChunk]:
    """Chunk blocks at section boundaries without splitting tables.

    Chunk IDs are deterministic: ``<doc_id>:c00000``, ``<doc_id>:c00001``, ...
    Table blocks are emitted as standalone chunks because splitting a table
    across chunks destroys row/header context.
    """
    tenant = validate_workspace_id(workspace_id)
    if max_chars < 1 or target_chars < 1:
        raise ValueError("target_chars and max_chars must be >= 1")
    if target_chars > max_chars:
        raise ValueError("target_chars cannot exceed max_chars")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than target_chars")

    chunks: List[DocumentChunk] = []
    current = _Draft()

    def emit(draft: _Draft) -> None:
        content = "\n\n".join(part for part in draft.parts if part.strip()).strip()
        if not content:
            return
        if not draft.pages:
            draft.pages = [1]
        index = len(chunks)
        page_start = min(draft.pages)
        page_end = max(draft.pages)
        chunks.append(
            DocumentChunk(
                chunk_id=compute_chunk_id(doc_id, index),
                doc_id=doc_id,
                doc_title=doc_title,
                content=content,
                page_number=page_start,
                section_hierarchy=list(draft.hierarchy),
                has_table=draft.has_table,
                table_markdown=draft.table_markdown,
                char_count=len(content),
                token_count_est=estimate_tokens(content),
                workspace_id=tenant,
                chunk_index=index,
                source_type=source_type,
                metadata={
                    "page_start": page_start,
                    "page_end": page_end,
                    "block_types": unique_preserving_order(draft.block_types),
                },
            )
        )

    def flush() -> None:
        nonlocal current
        emit(current)
        current = _Draft()

    for block in blocks:
        block_type = block.block_type
        page_number = max(1, int(block.page_number or 1))
        hierarchy = list(block.section_hierarchy or [])

        if block_type == "heading":
            flush()
            heading = normalize_text(block.text)
            if heading:
                current = _Draft(
                    parts=[heading],
                    pages=[page_number],
                    hierarchy=hierarchy,
                    block_types=["heading"],
                )
            continue

        if block_type == "table":
            flush()
            table = normalize_table_markdown(
                block.table_markdown or block.text or ""
            )
            if not table:
                continue
            table_draft = _Draft(
                parts=[table],
                pages=[page_number],
                hierarchy=hierarchy,
                block_types=["table"],
                has_table=True,
                table_markdown=table,
            )
            emit(table_draft)
            continue

        text = normalize_text(block.text)
        if not text:
            continue

        if current.parts and current.hierarchy != hierarchy:
            flush()

        for piece in _split_long_piece(text, max_chars):
            if not piece:
                continue

            proposed_length = current.length + (2 if current.parts else 0) + len(piece)
            if current.parts and proposed_length > max_chars:
                overlap = _overlap_text(current.parts, overlap_chars)
                last_page = current.pages[-1]
                previous_hierarchy = list(current.hierarchy)
                previous_types = list(current.block_types)
                flush()
                if overlap and len(overlap) + 2 + len(piece) <= max_chars:
                    current = _Draft(
                        parts=[overlap],
                        pages=[last_page],
                        hierarchy=previous_hierarchy,
                        block_types=["overlap", *previous_types[-1:]],
                    )

            elif current.parts and proposed_length > target_chars:
                overlap = _overlap_text(current.parts, overlap_chars)
                last_page = current.pages[-1]
                previous_hierarchy = list(current.hierarchy)
                previous_types = list(current.block_types)
                flush()
                if overlap and len(overlap) + 2 + len(piece) <= max_chars:
                    current = _Draft(
                        parts=[overlap],
                        pages=[last_page],
                        hierarchy=previous_hierarchy,
                        block_types=["overlap", *previous_types[-1:]],
                    )

            if not current.parts:
                current = _Draft(
                    parts=[],
                    pages=[],
                    hierarchy=hierarchy,
                    block_types=[],
                )
            current.add(piece, page_number, block_type)

    flush()
    return chunks
