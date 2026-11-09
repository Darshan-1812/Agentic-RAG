"""Resolve Hugging Face models to immutable local snapshot paths.

FlagEmbedding does not expose a revision argument in its public constructors.
Resolving the snapshot first lets retrieval pin and persist the exact commit
revision used for stored embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ModelResolutionError(RuntimeError):
    """Raised when an embedding or reranker model cannot be resolved."""


@dataclass(frozen=True)
class ResolvedModel:
    path: str
    model_name: str
    revision: str
    local_path: bool = False


def resolve_model_source(
    model_name_or_path: str,
    revision: str = "main",
    *,
    cache_dir: Optional[Path] = None,
    local_files_only: bool = False,
) -> ResolvedModel:
    """Return a local model path and the concrete revision to persist."""
    name = model_name_or_path.strip()
    if not name:
        raise ModelResolutionError("model name must not be empty")

    candidate = Path(name).expanduser()
    if candidate.exists():
        return ResolvedModel(
            path=str(candidate.resolve()),
            model_name=name,
            revision=revision or "local",
            local_path=True,
        )

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is present in retrieval env
        raise ModelResolutionError(
            "huggingface_hub is required to download retrieval models"
        ) from exc

    try:
        snapshot = snapshot_download(
            repo_id=name,
            revision=revision or "main",
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        mode = "from the local cache" if local_files_only else "from Hugging Face"
        raise ModelResolutionError(
            f"Unable to resolve model {name!r} revision {revision!r} {mode}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    path = Path(snapshot)
    commit = path.name if "snapshots" in path.parts else (revision or "main")
    return ResolvedModel(
        path=str(path),
        model_name=name,
        revision=commit,
        local_path=False,
    )
