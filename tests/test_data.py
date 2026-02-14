import torch

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
