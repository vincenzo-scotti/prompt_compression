"""llama.cpp server backend: payload, generation and token counting (HTTP mocked)."""
import json

import pytest

requests = pytest.importorskip("requests")

from prompt_piper.backends import lm_eval as backend
from prompt_piper.backends.lm_eval import (
    LMEvalTokenizer, _is_chat_model, _load_model, _make_instance, _wrap_chat_prompt,
)


class _Resp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.ok = True
        self.text = json.dumps(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


@pytest.fixture
def model():
    return _load_model("llama-cpp", "local-gguf", base_url="http://localhost:9999/v1/chat/completions")


def test_is_chat_model_and_urls(model):
    assert _is_chat_model(model)
    assert model.server_url == "http://localhost:9999"


def test_payload_strips_hf_only_keys(model):
    payload = model._create_payload(
        [{"role": "user", "content": "hi"}], generate=True,
        gen_kwargs={"until": [], "max_gen_toks": 32, "do_sample": False, "num_beams": 1},
    )
    assert payload["max_tokens"] == 32
    assert "num_beams" not in payload and "do_sample" not in payload and "stop" not in payload


def test_tokenize_endpoint_used(model, monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        seen["url"], seen["json"] = url, json
        return _Resp({"tokens": [1, 2, 3, 4]})

    monkeypatch.setattr(requests, "post", fake_post)
    assert LMEvalTokenizer(model).count("four token text") == 4
    assert seen["url"] == "http://localhost:9999/tokenize"
    assert seen["json"] == {"content": "four token text", "add_special": False}


def test_generate_until_round_trip(model, monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, **kw):
        sent["url"], sent["payload"] = url, json
        return _Resp({"choices": [{"message": {"content": "The answer is 42."}, "index": 0}]})

    monkeypatch.setattr(requests, "post", fake_post)
    req = _make_instance("generate_until", {}, (_wrap_chat_prompt("Q?"), {"max_gen_toks": 16}), 0)
    out = model.generate_until([req], disable_tqdm=True)
    assert out == ["The answer is 42."]
    assert sent["url"].endswith("/v1/chat/completions")
    assert sent["payload"]["messages"][-1]["content"] == "Q?"


def test_provider_usage_reaches_evaluation(model, monkeypatch):
    usage = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}

    def fake_post(url, json=None, headers=None, **kw):
        return _Resp({"choices": [{"message": {"content": "42"}, "index": 0}],
                      "usage": usage, "timings": {"predicted_ms": 3.0}})

    monkeypatch.setattr(requests, "post", fake_post)
    req = _make_instance("generate_until", {}, (_wrap_chat_prompt("Q?"), {"max_gen_toks": 8}), 0)
    out = model.generate_until([req], disable_tqdm=True)[0]
    assert out == "42" and out.usage["prompt_tokens"] == 11 and out.usage["timings"]["predicted_ms"] == 3.0

    import types
    from prompt_piper.backends.lm_eval import LMEvalEvaluator
    ev = LMEvalEvaluator.__new__(LMEvalEvaluator)
    ev._model, ev._metric_key, ev._gen_kwargs = model, None, {"max_gen_toks": 8}
    ev._task = types.SimpleNamespace(
        OUTPUT_TYPE="generate_until", config=types.SimpleNamespace(filter_list=None),
        process_results=lambda doc, f: {"acc": float(f[0] == "42")},
    )
    ev._tasks = None
    detail = ev.evaluate_batch_detailed(["Q?"], ["Q?"], [{"_doc": {}}])[0]
    assert detail["score"] == 1.0 and detail["provider_usage"]["completion_tokens"] == 5
    assert type(detail["output"]) is str


def test_llama_swap_routing_auth_and_thinking(monkeypatch):
    monkeypatch.setenv("LLAMA_CPP_API_KEY", "secret")
    m = _load_model("llama-cpp", "qwen3.5-9b", base_url="http://swap:9292/v1/chat/completions",
                    llama_swap=True, disable_thinking=True)
    assert m.tokenize_url == "http://swap:9292/upstream/qwen3.5-9b/tokenize"
    assert m.header["Authorization"] == "Bearer secret"
    payload = m._create_payload([{"role": "user", "content": "hi"}], generate=True,
                                gen_kwargs={"max_gen_toks": 8})
    assert payload["model"] == "qwen3.5-9b"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kw):
        seen.update(url=url, headers=headers)
        return _Resp({"tokens": [1, 2]})

    monkeypatch.setattr(requests, "post", fake_post)
    assert m.count_tokens("x") == 2
    assert seen["url"].endswith("/upstream/qwen3.5-9b/tokenize")
    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_reasoning_content_is_captured(model, monkeypatch):
    def fake_post(url, json=None, headers=None, **kw):
        return _Resp({"choices": [{"message": {"content": "42", "reasoning_content": "hmm"}, "index": 0}]})

    monkeypatch.setattr(requests, "post", fake_post)
    req = _make_instance("generate_until", {}, (_wrap_chat_prompt("Q?"), {"max_gen_toks": 8}), 0)
    out = model.generate_until([req], disable_tqdm=True)[0]
    assert out == "42" and out.reasoning == "hmm"


def test_gpt_and_gemini_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    msgs = [{"role": "user", "content": "hi"}]
    gpt = _load_model("openai-chat", "gpt-5.4-nano-2026-03-17", reasoning_effort="none",
                      drop_stop_sequences=True)
    assert gpt._create_payload(msgs, generate=True, gen_kwargs={"max_gen_toks": 8})["reasoning_effort"] == "none"
    gem = _load_model("gemini", "gemini-3.8-flash", reasoning_effort="minimal")
    assert gem._create_payload(msgs, generate=True, gen_kwargs={"max_gen_toks": 8})["reasoning_effort"] == "minimal"
    with pytest.raises(ValueError, match="reasoning_effort"):
        _load_model("gemini", "gemini-3.8-flash", disable_thinking=True)


def test_base_url_resolution(monkeypatch, caplog):
    from prompt_piper.backends.lm_eval import _resolve_llama_cpp_url

    monkeypatch.delenv("LLAMA_CPP_BASE_URL", raising=False)
    with caplog.at_level("WARNING"):
        assert _resolve_llama_cpp_url(None) == "http://localhost:8080/v1/chat/completions"
    assert "LLAMA_CPP_BASE_URL" in caplog.text
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://swap:9292/")
    assert _resolve_llama_cpp_url(None) == "http://swap:9292/v1/chat/completions"
    monkeypatch.setenv("LLAMA_CPP_BASE_URL", "http://swap:9292/v1")
    assert _resolve_llama_cpp_url(None) == "http://swap:9292/v1/chat/completions"
    # an explicit config value wins over the environment
    assert _resolve_llama_cpp_url("http://cfg:1/v1/chat/completions") == "http://cfg:1/v1/chat/completions"
    m = _load_model("llama-cpp", "qwen3.5-9b", llama_swap=True)
    assert m.base_url == "http://swap:9292/v1/chat/completions"
    assert m.tokenize_url == "http://swap:9292/upstream/qwen3.5-9b/tokenize"


def test_chat_model_drops_task_fewshot_stops(model):
    import types
    from prompt_piper.backends.lm_eval import LMEvalEvaluator

    task = types.SimpleNamespace(
        OUTPUT_TYPE="generate_until",
        config=types.SimpleNamespace(generation_kwargs={"max_gen_toks": 1024, "until": ["</s>", "Q"]}),
    )
    ev = LMEvalEvaluator(model, task)
    assert "until" not in ev._gen_kwargs          # BBH's "Q" would cut "Quail"
    ev2 = LMEvalEvaluator(model, task, extra_stop_sequences=["<|im_end|>"])
    assert ev2._gen_kwargs["until"] == ["<|im_end|>"]  # explicit stops are kept
