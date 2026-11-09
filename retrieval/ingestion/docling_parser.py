"""IBM Docling adapter for PDF, DOCX, and Markdown ingestion.

Docling is imported lazily. Tests and downstream code can inject a converter
object exposing ``convert(path)`` to exercise the normalization/chunking path
without installing the full ML stack.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from retrieval.config import RetrievalSettings, get_settings
from retrieval.ingestion.metadata import UnsupportedDocumentError, detect_mime_type
from retrieval.ingestion.normalizer import (
    normalize_heading,
    normalize_table_markdown,
    normalize_text,
    split_paragraphs,
)
from retrieval.schemas import ParsedBlock, ParsedDocument, validate_workspace_id


class DoclingUnavailableError(RuntimeError):
    """Raised when Docling is not installed and no converter was injected."""


class DocumentParseError(RuntimeError):
    """Raised when Docling cannot convert a document."""


_HEADING_LABELS = {
    "title",
    "section_header",
    "section-header",
    "heading",
    "subtitle",
}
_TABLE_LABELS = {"table"}
_LIST_LABELS = {"list_item", "list-item", "list"}
_CAPTION_LABELS = {"caption"}
_CODE_LABELS = {"code"}
_FORMULA_LABELS = {"formula"}
_PAGE_FURNITURE_LABELS = {"page_header", "page_footer", "page-header", "page-footer"}
_SKIP_LABELS = {"picture", "image", "document_index", "reference"}


def _label_name(item: Any) -> str:
    label = getattr(item, "label", None)
    if label is None and isinstance(item, dict):
        label = item.get("label")
    value = getattr(label, "value", None)
    if value is None:
        value = label
    return str(value or "").strip().lower().replace(" ", "_")


def _item_text(item: Any) -> str:
    for attribute in ("text", "content"):
        value = getattr(item, attribute, None)
        if value is None and isinstance(item, dict):
            value = item.get(attribute)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _item_metadata(item: Any) -> Dict[str, Any]:
    metadata = getattr(item, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _first_provenance(item: Any) -> Any:
    provenance = getattr(item, "prov", None)
    if provenance is None and isinstance(item, dict):
        provenance = item.get("prov")
    if not provenance:
        return None
    if isinstance(provenance, (list, tuple)):
        return provenance[0]
    return provenance


def _page_number(item: Any) -> int:
    provenance = _first_provenance(item)
    if provenance is None:
        return 1
    page: Any
    if isinstance(provenance, dict):
        page = provenance.get("page_no", provenance.get("page"))
    else:
        page = getattr(provenance, "page_no", None) or getattr(provenance, "page", None)
    try:
        return max(1, int(page))
    except (TypeError, ValueError):
        return 1


def _bbox(item: Any) -> Optional[List[float]]:
    provenance = _first_provenance(item)
    if provenance is None:
        return None
    bbox = provenance.get("bbox") if isinstance(provenance, dict) else getattr(provenance, "bbox", None)
    if bbox is None:
        return None
    if hasattr(bbox, "as_tuple"):
        bbox = bbox.as_tuple()
    if isinstance(bbox, dict):
        bbox = [bbox.get(key) for key in ("l", "t", "r", "b")]
    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None


def _document_origin_filename(document: Any) -> Optional[str]:
    origin = getattr(document, "origin", None)
    if origin is None:
        return None
    if isinstance(origin, dict):
        value = origin.get("filename")
    else:
        value = getattr(origin, "filename", None)
    return str(value) if value else None


def _document_page_count(document: Any, blocks: Sequence[ParsedBlock]) -> int:
    pages = getattr(document, "pages", None)
    if pages is not None:
        try:
            count = len(pages)
            if count:
                return int(count)
        except TypeError:
            pass
    return max((block.page_number for block in blocks), default=0)


def _conversion_failed(result: Any) -> Optional[str]:
    status = getattr(result, "status", None)
    status_name = getattr(status, "value", None) or str(status or "")
    lowered = status_name.lower()
    if "failure" in lowered or "error" in lowered:
        errors = getattr(result, "errors", None)
        return f"Docling conversion status={status_name}, errors={errors!r}"
    errors = getattr(result, "errors", None)
    if errors:
        return f"Docling reported conversion errors: {errors!r}"
    return None


def _table_markdown(item: Any, document: Any) -> str:
    exporter = getattr(item, "export_to_markdown", None)
    if exporter is None and isinstance(item, dict):
        exporter = item.get("export_to_markdown")
    if not callable(exporter):
        return ""

    for args in ((document,), ()):
        try:
            value = exporter(*args)
        except TypeError:
            continue
        except Exception:
            break
        if value:
            return normalize_table_markdown(str(value))
    return ""


def _fallback_markdown_blocks(markdown: str) -> List[ParsedBlock]:
    """Build coarse blocks when a converter exposes only markdown output."""
    blocks: List[ParsedBlock] = []
    heading_stack: List[Tuple[int, str]] = []
    buffer: List[str] = []

    def flush_text() -> None:
        nonlocal buffer
        if not buffer:
            return
        hierarchy = [heading for _, heading in heading_stack]
        for paragraph in split_paragraphs("\n\n".join(buffer)):
            blocks.append(
                ParsedBlock(
                    block_type="text",
                    text=paragraph,
                    page_number=1,
                    section_hierarchy=hierarchy,
                )
            )
        buffer = []

    for raw_line in normalize_text(markdown).splitlines():
        line = raw_line.rstrip()
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_text()
            level = len(heading.group(1))
            text = normalize_heading(heading.group(2))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            blocks.append(
                ParsedBlock(
                    block_type="heading",
                    text=text,
                    page_number=1,
                    section_hierarchy=[value for _, value in heading_stack],
                    level=level,
                )
            )
        elif line.strip().startswith("|") and "|" in line:
            flush_text()
            blocks.append(
                ParsedBlock(
                    block_type="table",
                    text=normalize_table_markdown(line),
                    page_number=1,
                    section_hierarchy=[value for _, value in heading_stack],
                    table_markdown=normalize_table_markdown(line),
                )
            )
        else:
            buffer.append(line)
    flush_text()
    return blocks


class DoclingParser:
    """Convert one source file into normalized structural blocks."""

    def __init__(
        self,
        settings: Optional[RetrievalSettings] = None,
        converter: Any = None,
    ):
        self.settings = settings or get_settings()
        self._converter = converter

    def _get_converter(self) -> Any:
        if self._converter is not None:
            return self._converter
        try:
            from docling.document_converter import DocumentConverter  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise DoclingUnavailableError(
                "Docling is not installed. Install retrieval/requirements-ingestion.txt "
                "or inject a converter for tests."
            ) from exc

        try:  # Configure PDF OCR without depending on one exact Docling minor API.
            from docling.datamodel.base_models import InputFormat  # type: ignore
            from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
            from docling.document_converter import PdfFormatOption  # type: ignore

            pdf_options = PdfPipelineOptions()
            pdf_options.do_ocr = self.settings.docling_ocr
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
                }
            )
        except Exception:  # pragma: no cover - version-specific constructor
            self._converter = DocumentConverter()
        return self._converter

    def parse(
        self,
        file_path: Path | str,
        workspace_id: str,
    ) -> ParsedDocument:
        """Parse a document and return normalized blocks and provenance."""
        tenant = validate_workspace_id(workspace_id)
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document does not exist: {path}")
        try:
            detect_mime_type(path)
        except UnsupportedDocumentError:
            raise

        converter = self._get_converter()
        try:
            result = converter.convert(str(path))
        except Exception as exc:
            raise DocumentParseError(
                f"Docling failed to convert {path.name}: {type(exc).__name__}: {exc}"
            ) from exc

        failure = _conversion_failed(result)
        if failure:
            raise DocumentParseError(failure)

        document = getattr(result, "document", None)
        if document is None:
            document = result

        blocks = self._extract_blocks(document)
        if not blocks:
            markdown = self._export_markdown(document)
            if markdown:
                blocks = _fallback_markdown_blocks(markdown)
        if not blocks:
            raise DocumentParseError(f"Docling produced no readable content for {path.name}")

        title = self._document_title(document, blocks, path)
        page_count = _document_page_count(document, blocks)
        extension = path.suffix.lower()
        metadata = {
            "docling_used": True,
            "source_extension": extension,
            "source_filename": path.name,
            "workspace_id": tenant,
            "block_count": len(blocks),
            "conversion_status": str(getattr(result, "status", "") or ""),
        }
        return ParsedDocument(
            doc_title=title,
            page_count=page_count,
            blocks=blocks,
            metadata=metadata,
        )

    def _extract_blocks(self, document: Any) -> List[ParsedBlock]:
        iterator = getattr(document, "iterate_items", None)
        if not callable(iterator):
            return []

        blocks: List[ParsedBlock] = []
        heading_stack: List[Tuple[int, str]] = []

        try:
            items = iterator()
        except TypeError:
            items = iterator  # pragma: no cover - defensive for older adapters

        for item, raw_level in items:
            label = _label_name(item)
            if label in _SKIP_LABELS:
                continue
            if label in _PAGE_FURNITURE_LABELS and not self.settings.docling_include_page_furniture:
                continue

            if label in _HEADING_LABELS:
                text = normalize_heading(_item_text(item))
                if not text:
                    continue
                level = max(1, int(raw_level) if raw_level is not None else 1)
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, text))
                blocks.append(
                    ParsedBlock(
                        block_type="heading",
                        text=text,
                        page_number=_page_number(item),
                        section_hierarchy=[value for _, value in heading_stack],
                        level=level,
                        metadata={"label": label, "bbox": _bbox(item)},
                    )
                )
                continue

            hierarchy = [value for _, value in heading_stack]
            page_number = _page_number(item)
            metadata = _item_metadata(item)
            metadata.update({"label": label, "bbox": _bbox(item)})

            if label in _TABLE_LABELS:
                markdown = _table_markdown(item, document)
                text = markdown or normalize_text(_item_text(item))
                if not text:
                    continue
                blocks.append(
                    ParsedBlock(
                        block_type="table",
                        text=text,
                        page_number=page_number,
                        section_hierarchy=hierarchy,
                        table_markdown=markdown or None,
                        metadata=metadata,
                    )
                )
                continue

            text = normalize_text(_item_text(item))
            if not text:
                continue

            if label in _LIST_LABELS:
                block_type = "list"
            elif label in _CAPTION_LABELS:
                block_type = "caption"
            elif label in _CODE_LABELS:
                block_type = "code"
            elif label in _FORMULA_LABELS:
                block_type = "formula"
            else:
                block_type = "text"

            blocks.append(
                ParsedBlock(
                    block_type=block_type,
                    text=text,
                    page_number=page_number,
                    section_hierarchy=hierarchy,
                    metadata=metadata,
                )
            )

        return blocks

    @staticmethod
    def _export_markdown(document: Any) -> str:
        exporter = getattr(document, "export_to_markdown", None)
        if not callable(exporter):
            return ""
        try:
            value = exporter()
        except Exception:
            return ""
        return normalize_text(str(value)) if value else ""

    @staticmethod
    def _document_title(
        document: Any, blocks: Sequence[ParsedBlock], path: Path
    ) -> str:
        for block in blocks:
            if block.metadata.get("label") == "title":
                return block.text
        for block in blocks:
            if block.block_type == "heading":
                return block.text
        origin_name = _document_origin_filename(document)
        if origin_name:
            return Path(origin_name).stem
        return path.stem
