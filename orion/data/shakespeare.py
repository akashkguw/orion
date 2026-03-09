# orion/data/shakespeare.py
from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
MIN_SHAKESPEARE_BYTES = 100_000
DOWNLOAD_RETRIES = 5
DOWNLOAD_TIMEOUT_SECONDS = 30


def _download_if_needed(path: Path) -> None:
    """Download tinyshakespeare with retries and atomic file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= MIN_SHAKESPEARE_BYTES:
        return
    if path.exists():
        path.unlink()

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with urllib.request.urlopen(SHAKESPEARE_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp:
                content_length = resp.headers.get("Content-Length")
                expected_size = (
                    int(content_length) if content_length and content_length.isdigit() else None
                )
                with tmp_path.open("wb") as f:
                    shutil.copyfileobj(resp, f)

            actual_size = tmp_path.stat().st_size
            if expected_size is not None and actual_size < expected_size:
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got {actual_size} out of {expected_size} bytes",
                    None,
                )
            if actual_size < MIN_SHAKESPEARE_BYTES:
                raise urllib.error.ContentTooShortError(
                    f"downloaded file too small: got {actual_size} bytes",
                    None,
                )

            tmp_path.replace(path)
            return
        except (OSError, urllib.error.URLError, urllib.error.ContentTooShortError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise RuntimeError(
        f"Failed to download tinyshakespeare after {DOWNLOAD_RETRIES} attempts: {last_error}"
    ) from last_error


@dataclass
class CharTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        chars = sorted(set(text))
        itos = chars
        stoi = {ch: i for i, ch in enumerate(chars)}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        return "".join(self.itos[i] for i in ids.tolist())


def load_tiny_shakespeare(
    root: str | Path = "data",
) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """
    Returns: (train_ids, val_ids, tokenizer)
    """
    root = Path(root)
    path = root / "tinyshakespeare.txt"
    _download_if_needed(path)

    text = path.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = tok.encode(text)

    # 90/10 split
    n = int(0.9 * ids.numel())
    train_ids = ids[:n].contiguous()
    val_ids = ids[n:].contiguous()
    return train_ids, val_ids, tok


def sample_batch(
    ids: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random contiguous sequences for next-token prediction.
    x: [B, T], y: [B, T]
    """
    n = ids.numel()
    max_start = n - (seq_len + 1)
    if max_start < 0:
        raise ValueError(
            f"Not enough tokens to sample seq_len={seq_len}: need at least {seq_len + 1}, got {n}"
        )
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts], dim=0).to(device)
    y = torch.stack([ids[s + 1 : s + 1 + seq_len] for s in starts], dim=0).to(device)
    return x, y
