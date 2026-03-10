from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import torch

from .shakespeare import CharTokenizer

DEFAULT_PG19_DATASET_ID = "deepmind/pg19"
DEFAULT_PG19_STREAMING = True
PARQUET_FALLBACK_REVISION = "refs/convert/parquet"


def _cache_key(
    *,
    dataset_id: str,
    train_docs: int | None,
    val_docs: int | None,
    train_chars: int | None,
    val_chars: int | None,
) -> str:
    key_payload = {
        "dataset_id": dataset_id,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "train_chars": train_chars,
        "val_chars": val_chars,
    }
    digest = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest[:16]


def _normalize_dataset_id(dataset_id: str) -> str:
    value = dataset_id.strip()
    lowered = value.lower()
    if lowered in {"pg19", "pg-19", "deepmind/pg19.py", "pg19.py"}:
        return DEFAULT_PG19_DATASET_ID
    if value.endswith(".py"):
        # Avoid dataset script loading paths with modern datasets versions.
        stem = Path(value).stem.lower()
        if stem in {"pg19", "deepmind/pg19"}:
            return DEFAULT_PG19_DATASET_ID
    return value or DEFAULT_PG19_DATASET_ID


def _is_script_compat_error(exc: BaseException) -> bool:
    return "dataset scripts are no longer supported" in str(exc).lower()


def _load_hf_pg19_dataset(datasets_mod, dataset_id: str, *, streaming: bool):
    attempts = [
        (dataset_id, None),
        (dataset_id, PARQUET_FALLBACK_REVISION),
        (DEFAULT_PG19_DATASET_ID, PARQUET_FALLBACK_REVISION),
        (DEFAULT_PG19_DATASET_ID, None),
    ]

    seen: set[tuple[str, str | None]] = set()
    last_script_exc: BaseException | None = None
    for resolved_id, revision in attempts:
        key = (resolved_id, revision)
        if key in seen:
            continue
        seen.add(key)

        kwargs: dict[str, Any] = {"streaming": streaming}
        if revision is not None:
            kwargs["revision"] = revision

        try:
            return datasets_mod.load_dataset(resolved_id, **kwargs), revision
        except Exception as exc:
            if _is_script_compat_error(exc):
                last_script_exc = exc
                continue
            raise

    raise RuntimeError(
        "Failed to load PG-19 after script-compatibility fallbacks "
        f"(dataset_id={dataset_id!r}, fallback_revision={PARQUET_FALLBACK_REVISION!r}). "
        "Please ensure you are on latest main and have a recent 'datasets' package."
    ) from last_script_exc


def _import_hf_datasets():
    try:
        import datasets  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by message assertion in tests
        raise RuntimeError(
            "Loading PG-19 requires the optional 'datasets' package. "
            "Install it with: pip install datasets"
        ) from exc
    return datasets


def _infer_text_field(example: dict[str, Any]) -> str:
    for candidate in ("text", "document", "content", "book_text"):
        value = example.get(candidate)
        if isinstance(value, str):
            return candidate

    for key, value in example.items():
        if isinstance(value, str):
            return key

    raise ValueError("Could not find a string text field in PG-19 examples")


def _take_limited_text(
    split_iterable,
    *,
    text_field: str,
    max_docs: int | None,
    max_chars: int | None,
) -> str:
    if max_docs is not None and max_docs <= 0:
        raise ValueError(f"max_docs must be > 0 when provided, got {max_docs}")
    if max_chars is not None and max_chars <= 0:
        raise ValueError(f"max_chars must be > 0 when provided, got {max_chars}")

    docs_seen = 0
    chars_seen = 0
    chunks: list[str] = []

    for row in split_iterable:
        text = row.get(text_field)
        if not isinstance(text, str) or not text:
            continue

        chunks.append(text)
        chunks.append("\n\n")
        docs_seen += 1
        chars_seen += len(text) + 2

        if max_docs is not None and docs_seen >= max_docs:
            break
        if max_chars is not None and chars_seen >= max_chars:
            break

    if not chunks:
        raise ValueError(
            f"No usable text rows found for split (field={text_field!r}, "
            f"max_docs={max_docs}, max_chars={max_chars})"
        )

    if max_chars is not None and chars_seen > max_chars:
        joined = "".join(chunks)
        return joined[:max_chars]

    return "".join(chunks)


def _resolve_validation_split_name(dataset_dict) -> str:
    for name in ("validation", "val", "test"):
        if name in dataset_dict:
            return name
    raise ValueError("PG-19 dataset is missing a validation/test split")


def load_pg19(
    root: str | Path = "data",
    *,
    dataset_id: str = DEFAULT_PG19_DATASET_ID,
    train_docs: int | None = None,
    val_docs: int | None = None,
    train_chars: int | None = None,
    val_chars: int | None = None,
    streaming: bool = DEFAULT_PG19_STREAMING,
    force_rebuild: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """Load PG-19 and return tokenized train/val ids using a char-level tokenizer.

    Data is downloaded at runtime via Hugging Face datasets and cached under:
      <root>/pg19/cache_<hash>/

    The cache hash captures dataset id and truncation options so runs remain reproducible.
    """
    root = Path(root)
    dataset_id = _normalize_dataset_id(dataset_id)
    cache_dir = (
        root
        / "pg19"
        / (
            "cache_"
            + _cache_key(
                dataset_id=dataset_id,
                train_docs=train_docs,
                val_docs=val_docs,
                train_chars=train_chars,
                val_chars=val_chars,
            )
        )
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_ids_path = cache_dir / "train_ids.pt"
    val_ids_path = cache_dir / "val_ids.pt"
    meta_path = cache_dir / "meta.json"

    if (
        not force_rebuild
        and train_ids_path.exists()
        and val_ids_path.exists()
        and meta_path.exists()
    ):
        train_ids = torch.load(train_ids_path, map_location="cpu")
        val_ids = torch.load(val_ids_path, map_location="cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tok = CharTokenizer(stoi={ch: i for i, ch in enumerate(meta["itos"])}, itos=meta["itos"])
        return train_ids.contiguous(), val_ids.contiguous(), tok

    datasets = _import_hf_datasets()
    dataset_dict, hf_revision = _load_hf_pg19_dataset(
        datasets,
        dataset_id,
        streaming=streaming,
    )
    if "train" not in dataset_dict:
        raise ValueError(f"PG-19 dataset {dataset_id!r} has no 'train' split")

    val_split_name = _resolve_validation_split_name(dataset_dict)

    train_split = dataset_dict["train"]
    val_split = dataset_dict[val_split_name]

    train_iter = iter(train_split)
    try:
        first_example = next(train_iter)
    except StopIteration as exc:
        raise ValueError(f"PG-19 dataset {dataset_id!r} train split is empty") from exc
    text_field = _infer_text_field(first_example)

    # Include the first example consumed for text field inference.
    train_text = _take_limited_text(
        itertools.chain([first_example], train_iter),
        text_field=text_field,
        max_docs=train_docs,
        max_chars=train_chars,
    )
    val_text = _take_limited_text(
        val_split,
        text_field=text_field,
        max_docs=val_docs,
        max_chars=val_chars,
    )

    tok = CharTokenizer.from_text(train_text + val_text)
    train_ids = tok.encode(train_text).contiguous()
    val_ids = tok.encode(val_text).contiguous()

    torch.save(train_ids, train_ids_path)
    torch.save(val_ids, val_ids_path)
    meta = {
        "dataset_id": dataset_id,
        "hf_revision": hf_revision,
        "text_field": text_field,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "train_chars": train_chars,
        "val_chars": val_chars,
        "streaming": streaming,
        "itos": tok.itos,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return train_ids, val_ids, tok
