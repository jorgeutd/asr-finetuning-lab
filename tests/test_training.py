import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from asr_lab.modeling import add_lora  # noqa: E402


@pytest.mark.parametrize("method", ["full", "lora"])
def test_real_optimizer_step_and_checkpoint_roundtrip(tmp_path, method):
    torch.manual_seed(42)
    config = transformers.WhisperConfig(
        vocab_size=32,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_source_positions=20,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    model = transformers.WhisperForConditionalGeneration(config)
    base = tmp_path / "base"
    model.save_pretrained(base)
    if method == "lora":
        model = add_lora(model, rank=2, alpha=4, dropout=0.0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    inputs = {"input_features": torch.randn(1, 80, 40), "labels": torch.tensor([[3, 4, 2]])}
    model.train()
    loss = model(**inputs).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed
    if method == "lora":
        assert all("lora_" in n for n in changed)
        assert (
            sum(p.numel() for p in model.parameters() if p.requires_grad) < model.num_parameters()
        )
    output = tmp_path / "trained"
    model.save_pretrained(output)
    if method == "lora":
        restored = transformers.WhisperForConditionalGeneration.from_pretrained(base)
        restored.load_adapter(output)
    else:
        restored = transformers.WhisperForConditionalGeneration.from_pretrained(output)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(**inputs).logits, restored(**inputs).logits)
