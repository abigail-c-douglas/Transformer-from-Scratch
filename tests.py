import torch
import pytest
from dataclasses import dataclass
from model import Config, Tokenizer, MLP, AttentionHead, TransformerBlock, Transformer

# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_TEXT = "the quick brown fox jumps over the lazy dog. is it fast? yes it is!"

@pytest.fixture
def config():
    tok = Tokenizer(SAMPLE_TEXT)
    return Config(d_model=16, d_vocab=tok.vocab_size, d_hidden=32, d_embedding=16)

@pytest.fixture
def tokenizer():
    return Tokenizer(SAMPLE_TEXT)

@pytest.fixture
def model(config, tokenizer):
    return Transformer(config, tokenizer=tokenizer, max_seq_length=32, num_blocks=2)


# ── Tokenizer tests ────────────────────────────────────────────────────────────

def test_tokenizer_encode_decode_roundtrip(tokenizer):
    """Encoding then decoding should return the original cleaned text."""
    original = "the fox"
    tokens = tokenizer.tokenize(original)
    result = tokenizer.detokenize(tokens)
    assert result == original, f"Expected '{original}', got '{result}'"

def test_tokenizer_output_is_list_of_ints(tokenizer):
    """tokenize() should return a list of integers."""
    tokens = tokenizer.tokenize("hello")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

def test_tokenizer_vocab_size(tokenizer):
    """Vocab size should be positive and match number of unique chars."""
    assert tokenizer.vocab_size > 0
    assert tokenizer.vocab_size == len(tokenizer.chars)

def test_tokenizer_unknown_chars_ignored(tokenizer):
    """Characters not in vocab should be silently ignored."""
    tokens = tokenizer.tokenize("hello@@@###")
    result = tokenizer.detokenize(tokens)
    assert "@" not in result
    assert "#" not in result


# ── MLP tests ─────────────────────────────────────────────────────────────────

def test_mlp_output_shape(config):
    """MLP should preserve input shape."""
    mlp = MLP(config)
    x = torch.rand(5, config.d_model)
    out = mlp(x)
    assert out.shape == (5, config.d_model), f"Expected (5, {config.d_model}), got {out.shape}"

def test_mlp_batched_output_shape(config):
    """MLP should work with a batch dimension."""
    mlp = MLP(config)
    x = torch.rand(4, 10, config.d_model)
    out = mlp(x)
    assert out.shape == (4, 10, config.d_model)


# ── AttentionHead tests ────────────────────────────────────────────────────────

def test_attention_output_shape(config):
    """Attention should return same shape as input."""
    attn = AttentionHead(config)
    x = torch.rand(10, config.d_model)
    out = attn(x)
    assert out.shape == (10, config.d_model), f"Expected (10, {config.d_model}), got {out.shape}"

def test_attention_causal_mask(config):
    """Future tokens should not influence past token outputs."""
    attn = AttentionHead(config)
    attn.eval()
    x = torch.rand(5, config.d_model)
    out1 = attn(x).detach().clone()

    # modify future token and check past tokens are unchanged
    x2 = x.clone()
    x2[4] = torch.rand(config.d_model)
    out2 = attn(x2).detach().clone()

    # position 0 should be unaffected by change at position 4
    assert torch.allclose(out1[0], out2[0], atol=1e-5), "Causal mask not working — past token changed"


# ── TransformerBlock tests ─────────────────────────────────────────────────────

def test_transformer_block_output_shape(config):
    """TransformerBlock should preserve input shape."""
    block = TransformerBlock(config)
    x = torch.rand(10, config.d_model)
    out = block(x)
    assert out.shape == (10, config.d_model)


# ── Transformer tests ──────────────────────────────────────────────────────────

def test_transformer_output_shape(config, tokenizer):
    """Transformer forward pass should output (batch, seq_len, d_vocab)."""
    model = Transformer(config, tokenizer=tokenizer, max_seq_length=32, num_blocks=2)
    x = torch.randint(0, config.d_vocab, (1, 10))
    out = model(x)
    assert out.shape == (1, 10, config.d_vocab), f"Expected (1, 10, {config.d_vocab}), got {out.shape}"

def test_transformer_different_inputs_different_outputs(config, tokenizer):
    """Different inputs should produce different outputs."""
    model = Transformer(config, tokenizer=tokenizer, max_seq_length=32, num_blocks=2)
    x1 = torch.randint(0, config.d_vocab, (1, 10))
    x2 = torch.randint(0, config.d_vocab, (1, 10))
    out1 = model(x1)
    out2 = model(x2)
    assert not torch.allclose(out1, out2), "Different inputs produced identical outputs"


# ── Generation tests ───────────────────────────────────────────────────────────

def test_generate_returns_string(model):
    """generate() should return a string."""
    result = model.generate("the fox", max_length=20)
    assert isinstance(result, str)

def test_generate_contains_prompt(model, tokenizer):
    """Generated text should contain the original prompt."""
    prompt = "the fox"
    result = model.generate(prompt, max_length=20)
    assert result.startswith(tokenizer.clean_text(prompt)) or prompt in result

def test_generate_length(model):
    """Generated text should be longer than the prompt."""
    prompt = "the"
    result = model.generate(prompt, max_length=50)
    assert len(result) > len(prompt)