"""Reusable retrieval evaluation harness.

The harness evaluates a retrieval callable against labeled query cases and
reports Recall@K, Precision@K, Hit@K, nDCG@K, and MRR@K. It never invents
scores: every metric is computed from the actual ranked chunk identifiers
returned by the retriever.
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from retrieval.evaluation.metrics import evaluate_ranking

Retriever = Callable[..., Any]


def _safe_error(message: object) -> str:
    value = " ".join(str(message).split())
    value = re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***", value)
    value = re.sub(r"Bearer\s+[A-Za-z0-9._\-]+", "Bearer ***", value)
    return value[:500]


def _chunk_id(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        chunk_id = value.get("chunk_id")
    else:
        chunk_id = getattr(value, "chunk_id", None)
    if chunk_id is None:
        raise TypeError(
            "retriever results must be chunk ids, mappings with 'chunk_id', "
            "or objects exposing 'chunk_id'"
        )
    return str(chunk_id)


def _ranked_ids(results: Iterable[object]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for result in results:
        chunk_id = _chunk_id(result)
        if chunk_id and chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def _call_retriever(
    retriever: Retriever,
    query: str,
    *,
    workspace_id: str,
    top_k: int,
) -> Any:
    try:
        signature = inspect.signature(retriever)
    except (TypeError, ValueError):
        return retriever(query, workspace_id=workspace_id, top_k=top_k)

    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: Dict[str, Any] = {}
    if accepts_kwargs or "workspace_id" in parameters:
        kwargs["workspace_id"] = workspace_id
    if accepts_kwargs or "top_k" in parameters:
        kwargs["top_k"] = top_k
    return retriever(query, **kwargs)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


@dataclass(frozen=True)
class RetrievalEvalCase:
    """One labeled retrieval query."""

    query: str
    workspace_id: str = "default"
    relevant_chunk_ids: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query", str(self.query))
        object.__setattr__(self, "workspace_id", str(self.workspace_id or "default"))
        object.__setattr__(
            self,
            "relevant_chunk_ids",
            tuple(str(value) for value in self.relevant_chunk_ids if str(value)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass
class RetrievalEvalReport:
    """Aggregate and per-case metrics for a labeled retrieval evaluation."""

    case_count: int
    successful_cases: int
    failed_cases: int
    k_values: Tuple[int, ...]
    metrics: Dict[str, float] = field(default_factory=dict)
    per_case: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.failed_cases == 0 and self.successful_cases > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_count": self.case_count,
            "successful_cases": self.successful_cases,
            "failed_cases": self.failed_cases,
            "k_values": list(self.k_values),
            "metrics": dict(self.metrics),
            "per_case": list(self.per_case),
            "failures": list(self.failures),
            "ok": self.ok,
        }


async def evaluate_retrieval(
    retriever: Retriever,
    cases: Iterable[RetrievalEvalCase],
    *,
    top_k: Optional[int] = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalEvalReport:
    """Evaluate ``retriever`` against labeled cases and aggregate the metrics."""
    case_list = list(cases)
    ks = tuple(sorted({int(k) for k in k_values if int(k) >= 1}))
    if not ks:
        raise ValueError("k_values must contain at least one value >= 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    retrieval_depth = top_k if top_k is not None else max(ks)

    metric_sums: Dict[str, float] = {}
    per_case: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for case in case_list:
        try:
            if not case.query.strip():
                raise ValueError("query must be a non-empty string")
            if not case.relevant_chunk_ids:
                raise ValueError("relevant_chunk_ids must not be empty")
            raw_results = _call_retriever(
                retriever,
                case.query,
                workspace_id=case.workspace_id,
                top_k=retrieval_depth,
            )
            results = await _maybe_await(raw_results)
            ranked = _ranked_ids(results or [])
            metrics, _ = evaluate_ranking(ranked, case.relevant_chunk_ids, ks)
            for name, value in metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + value
            per_case.append(
                {
                    "query": case.query,
                    "workspace_id": case.workspace_id,
                    "relevant_chunk_ids": list(case.relevant_chunk_ids),
                    "ranked_chunk_ids": ranked,
                    "metrics": metrics,
                    "metadata": dict(case.metadata),
                }
            )
        except Exception as exc:  # noqa: BLE001 - report per-case failures
            failures.append(
                {
                    "query": case.query,
                    "workspace_id": case.workspace_id,
                    "error": _safe_error(f"{type(exc).__name__}: {exc}"),
                }
            )

    successful = len(per_case)
    aggregate = {
        name: metric_sums[name] / successful for name in metric_sums
    } if successful else {}
    return RetrievalEvalReport(
        case_count=len(case_list),
        successful_cases=successful,
        failed_cases=len(failures),
        k_values=ks,
        metrics=aggregate,
        per_case=per_case,
        failures=failures,
    )


async def evaluate_service(
    service: Any,
    cases: Iterable[RetrievalEvalCase],
    *,
    top_k: Optional[int] = None,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalEvalReport:
    """Evaluate a ``RetrievalService`` using its ``retrieve`` method."""
    return await evaluate_retrieval(
        service.retrieve,
        cases,
        top_k=top_k,
        k_values=k_values,
    )


def _case_from_mapping(data: Mapping[str, Any]) -> RetrievalEvalCase:
    relevant = (
        data.get("relevant_chunk_ids")
        or data.get("relevant_ids")
        or data.get("relevant")
        or ()
    )
    if isinstance(relevant, str):
        relevant = [relevant]
    return RetrievalEvalCase(
        query=str(data.get("query", "")),
        workspace_id=str(data.get("workspace_id", "default")),
        relevant_chunk_ids=tuple(relevant),
        metadata=dict(data.get("metadata") or {}),
    )


def load_eval_cases(path: Path | str) -> List[RetrievalEvalCase]:
    """Load JSON array or JSONL evaluation cases from ``path``."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError("JSON evaluation file must contain an array")
        return [_case_from_mapping(item) for item in payload]

    cases: List[RetrievalEvalCase] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        cases.append(_case_from_mapping(payload))
    return cases


def write_eval_report(report: RetrievalEvalReport, path: Path | str) -> Path:
    """Write a sanitized JSON evaluation report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return destination
