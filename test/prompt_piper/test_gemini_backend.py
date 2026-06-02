"""Tests for the Gemini backend.

Covers:
  - _load_model('gemini', ...) returns a LocalChatCompletion (chat model detection)
  - api_key reads from GEMINI_API_KEY and raises ValueError when unset
  - _create_payload uses max_tokens (not max_completion_tokens) and drops seed
  - gemini-1.5-flash is not mis-detected as an OpenAI o-series model
    (stop sequences must be preserved, temperature must not be forced to 1)
  - loglikelihood raises NotImplementedError (Gemini is generation-only)
  - LMEvalTokenizer falls back to whitespace counting for Gemini
  - Unknown model-type error message mentions 'gemini'
"""
import os
import unittest.mock as mock
import pytest

from prompt_piper.backends.lm_eval import _load_model, _is_chat_model, LMEvalTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gemini_model(model_name="gemini-2.0-flash", **kwargs):
    """Instantiate a Gemini backend with a dummy key in the environment."""
    with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key-xyz"}):
        return _load_model("gemini", model_name, **kwargs)


def _payload(model, until=None, max_gen_toks=256):
    """Call _create_payload with representative gen_kwargs."""
    return model._create_payload(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        generate=True,
        gen_kwargs={"max_gen_toks": max_gen_toks, "until": until or ["\n"]},
    )


# ---------------------------------------------------------------------------
# Model type and detection
# ---------------------------------------------------------------------------

class TestGeminiModelType:

    def test_is_chat_model(self):
        assert _is_chat_model(_make_gemini_model())

    def test_model_name_stored(self):
        assert _make_gemini_model("gemini-2.0-flash").model == "gemini-2.0-flash"

    def test_model_name_flash_15(self):
        assert _make_gemini_model("gemini-1.5-flash").model == "gemini-1.5-flash"

    def test_base_url_points_to_gemini(self):
        assert "generativelanguage.googleapis.com" in _make_gemini_model().base_url

    def test_loglikelihood_not_supported(self):
        with pytest.raises(NotImplementedError):
            _make_gemini_model().loglikelihood([])


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

class TestGeminiApiKey:

    def test_reads_gemini_env_var(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "my-key-123"}):
            model = _load_model("gemini", "gemini-2.0-flash")
            assert model.api_key == "my-key-123"

    def test_raises_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            model = _load_model("gemini", "gemini-2.0-flash")
            with pytest.raises(ValueError, match="GEMINI_API_KEY"):
                _ = model.api_key

    def test_does_not_use_openai_key(self):
        """OPENAI_API_KEY must be ignored; GEMINI_API_KEY takes precedence."""
        overrides = {"GEMINI_API_KEY": "gemini-key", "OPENAI_API_KEY": "openai-key"}
        with mock.patch.dict(os.environ, overrides):
            model = _load_model("gemini", "gemini-2.0-flash")
            assert model.api_key == "gemini-key"


# ---------------------------------------------------------------------------
# _create_payload — Gemini API compatibility
# ---------------------------------------------------------------------------

class TestGeminiCreatePayload:

    def test_uses_max_tokens_not_max_completion_tokens(self):
        p = _payload(_make_gemini_model())
        assert "max_tokens" in p
        assert "max_completion_tokens" not in p

    def test_seed_absent(self):
        assert "seed" not in _payload(_make_gemini_model())

    def test_max_tokens_bumped_for_thinking_model(self):
        # Gemini Flash 3+ is a thinking model: max_tokens covers both the
        # internal chain-of-thought (~1000-1500 tokens) and the visible output.
        # Values below 4096 are raised to 4096 so the model has enough budget
        # for a complete response; larger values are left unchanged.
        assert _payload(_make_gemini_model(), max_gen_toks=256)["max_tokens"] == 4096
        assert _payload(_make_gemini_model(), max_gen_toks=1024)["max_tokens"] == 4096
        assert _payload(_make_gemini_model(), max_gen_toks=8192)["max_tokens"] == 8192

    def test_model_field_set(self):
        assert _payload(_make_gemini_model("gemini-2.0-flash"))["model"] == "gemini-2.0-flash"

    def test_stop_sequences_preserved(self):
        p = _payload(_make_gemini_model(), until=["Question:", "\n\n"])
        assert "stop" in p
        assert "Question:" in p["stop"]

    def test_gemini_15_stop_sequences_not_dropped(self):
        """gemini-1.5-flash contains '5' in its name — must not trigger the
        OpenAI o-series heuristic in lm-eval that drops stop sequences."""
        p = _payload(_make_gemini_model("gemini-1.5-flash"), until=["Question:"])
        assert "stop" in p
        assert "Question:" in p["stop"]

    def test_gemini_15_temperature_not_forced_to_one(self):
        """Same: OpenAI o-series heuristic must not force temperature=1."""
        p = _payload(_make_gemini_model("gemini-1.5-flash"))
        assert p.get("temperature") != 1

    def test_messages_passed_through(self):
        model = _make_gemini_model()
        msgs = [{"role": "user", "content": "hello"}]
        p = model._create_payload(
            messages=msgs, generate=True,
            gen_kwargs={"max_gen_toks": 64, "until": []},
        )
        assert p["messages"] is msgs


# ---------------------------------------------------------------------------
# Tokenizer fallback
# ---------------------------------------------------------------------------

class TestGeminiTokenizer:

    def test_counter_is_callable(self):
        tok = LMEvalTokenizer(_make_gemini_model())
        assert callable(tok._counter)

    def test_positive_count_for_nonempty_text(self):
        tok = LMEvalTokenizer(_make_gemini_model())
        count = tok.count("Hello world, this is a test prompt.")
        assert isinstance(count, int) and count > 0

    def test_empty_string_is_zero(self):
        assert LMEvalTokenizer(_make_gemini_model()).count("") == 0

    def test_longer_text_has_higher_count(self):
        tok = LMEvalTokenizer(_make_gemini_model())
        assert tok.count("one two three four five") > tok.count("one")


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------

class TestLoadModelErrors:

    def test_unknown_model_type_mentions_gemini(self):
        with pytest.raises(ValueError) as exc_info:
            _load_model("unknown_xyz", "some-model")
        assert "gemini" in str(exc_info.value)