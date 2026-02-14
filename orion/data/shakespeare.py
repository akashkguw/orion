# orion/data/shakespeare.py
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def _download_if_needed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    urllib.request.urlretrieve(SHAKESPEARE_URL, path)


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
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([ids[s : s + seq_len] for s in starts], dim=0).to(device)
    y = torch.stack([ids[s + 1 : s + 1 + seq_len] for s in starts], dim=0).to(device)
    return x, y
