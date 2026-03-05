import io

import pytest
import torch

from orion.data import shakespeare
from orion.data.shakespeare import CharTokenizer


def test_char_tokenizer_from_text():
    """Test tokenizer creation from text."""
    text = "hello world"
    tok = CharTokenizer.from_text(text)

    assert tok.vocab_size > 0
    assert "h" in tok.stoi
    assert "w" in tok.stoi


def test_char_tokenizer_encode():
    """Test encoding text to tokens."""
    text = "hello"
    tok = CharTokenizer.from_text(text)

    encoded = tok.encode("hello")
    assert isinstance(encoded, torch.Tensor)
    assert encoded.shape[0] == 5


def test_char_tokenizer_roundtrip():
    """Test encode/decode roundtrip."""
    text = "the quick brown fox"
    tok = CharTokenizer.from_text(text)

    encoded = tok.encode("the quick")
    decoded = tok.decode(encoded)

    assert decoded == "the quick"


class _DummyResponse(io.BytesIO):
    def __init__(self, payload: bytes, content_length: int | None = None):
        super().__init__(payload)
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def test_download_if_needed_skips_existing_valid_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "tinyshakespeare.txt"
    path.write_bytes(b"a" * shakespeare.MIN_SHAKESPEARE_BYTES)
    original = path.read_bytes()

    called = {"value": False}

    def fake_urlopen(*args, **kwargs):
        called["value"] = True
        return _DummyResponse(b"unused")

    monkeypatch.setattr(shakespeare.urllib.request, "urlopen", fake_urlopen)
    shakespeare._download_if_needed(path)

    assert called["value"] is False
    assert path.read_bytes() == original


def test_download_if_needed_replaces_small_existing_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "tinyshakespeare.txt"
    path.write_bytes(b"partial")

    payload = b"x" * (shakespeare.MIN_SHAKESPEARE_BYTES + 123)

    def fake_urlopen(*args, **kwargs):
        return _DummyResponse(payload, content_length=len(payload))

    monkeypatch.setattr(shakespeare.urllib.request, "urlopen", fake_urlopen)
    shakespeare._download_if_needed(path)

    assert path.read_bytes() == payload


def test_download_if_needed_retries_on_partial_content(tmp_path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "tinyshakespeare.txt"

    attempts = {"n": 0}
    good_payload = b"z" * (shakespeare.MIN_SHAKESPEARE_BYTES + 99)

    def fake_urlopen(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            # First attempt: payload smaller than declared content length.
            return _DummyResponse(b"a" * 32768, content_length=1115394)
        return _DummyResponse(good_payload, content_length=len(good_payload))

    monkeypatch.setattr(shakespeare.urllib.request, "urlopen", fake_urlopen)
    # Remove sleep in retry path for fast tests.
    monkeypatch.setattr(shakespeare.time, "sleep", lambda *_: None)

    shakespeare._download_if_needed(path)

    assert attempts["n"] == 2
    assert path.read_bytes() == good_payload
