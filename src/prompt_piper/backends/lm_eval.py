"""LM Evaluation Harness backend.

This module wires lm-eval into prompt_piper in three ways:

1. **Dataset** (:class:`LMEvalDataset`) — extracts prompts and targets from
   any lm-eval task, producing ``{"prompt": ..., "target": ..., "_doc": ...}``
   dicts compatible with :class:`~prompt_piper.optimization.runner.Experiment`.

2. **Evaluator** (:class:`LMEvalEvaluator`) — scores a compressed prompt by
   running the model through the task's native metric (loglikelihood for
   multiple-choice tasks, generation + task metric for open-ended tasks).

3. **Semantic backend** (:class:`LMEvalSemanticBackend`) — exposes the model
   as a ``(text, instruction) -> str`` callable so that semantic
   transformations O23–O28 can delegate to a real LLM instead of the default
   heuristic fallbacks.

The :class:`LMEvalBackend` facade bundles all three and provides a
``build_experiment()`` convenience method.

Requirements
------------
``pip install lm-eval``   (tested against lm-eval >= 0.4)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from prompt_piper.optimization.evaluator import Evaluator
from prompt_piper.optimization.tokenizer import Tokenizer, WhitespaceTokenizer
from prompt_piper.transformations.base import Transformation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------

def _require_lm_eval():
    try:
        import lm_eval
        return lm_eval
    except ImportError as exc:
        raise ImportError(
            "LM Evaluation Harness is required: pip install lm-eval"
        ) from exc


class GenerationText(str):
    """Generated text that also carries the provider's ``usage`` record.

    lm-eval passes generated strings through unchanged, so attaching the
    response's ``usage`` (billed prompt / completion / reasoning tokens) to the
    string lets the evaluator log exact provider token counts per request,
    also when requests run concurrently.
    """

    usage: dict | None = None
    reasoning: str | None = None


def _parse_generations_with_usage(outputs, **kwargs) -> list:
    """Chat-completions ``parse_generations`` that keeps each response's usage."""
    from lm_eval.models.openai_completions import LocalChatCompletion

    res = []
    for out in outputs if isinstance(outputs, list) else [outputs]:
        usage = out.get("usage") if isinstance(out, dict) else None
        if isinstance(out, dict) and "timings" in out:  # llama.cpp server
            usage = {**(usage or {}), "timings": out["timings"]}
        reasoning = {}
        if isinstance(out, dict):
            for choice in out.get("choices", []) or []:
                msg = choice.get("message") or {}
                r = msg.get("reasoning_content") or msg.get("reasoning")
                if r:
                    reasoning[choice.get("index", 0)] = r
        for i, text in enumerate(LocalChatCompletion.parse_generations(out)):
            if text is None:
                res.append(None)
            else:
                t = GenerationText(text)
                t.usage = usage
                # Visible reasoning returned by the provider (should be empty
                # when thinking is disabled); logged to verify the setting.
                t.reasoning = reasoning.get(i)
                res.append(t)
    return res


def _load_model(model_type: str, model_name: str, **kwargs):
    """Load an lm-eval model by type and name."""
    _require_lm_eval()
    if model_type in ("hf", "huggingface"):
        from lm_eval.models.huggingface import HFLM
        quantization_config = kwargs.pop("quantization_config", None)
        if quantization_config is not None:
            # lm-eval bug: HFLM.__init__ reads quantization_config from the HF
            # model config and passes it explicitly to _create_model(**kwargs),
            # so a user-supplied quantization_config in **kwargs ends up
            # duplicated. Workaround: pre-load the model and hand the object to
            # HFLM, which skips _create_model when pretrained is not a string.
            from transformers import AutoModelForCausalLM
            pretrained = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=kwargs.pop("device_map", "auto"),
                trust_remote_code=kwargs.get("trust_remote_code", False),
                revision=kwargs.get("revision", "main"),
                torch_dtype=kwargs.get("dtype", "auto"),
            )
        else:
            pretrained = model_name
        return HFLM(pretrained=pretrained, **kwargs)
    if model_type in ("openai", "openai-completions"):
        from lm_eval.models.openai_completions import OpenAICompletionsAPI
        return OpenAICompletionsAPI(model=model_name, **kwargs)
    if model_type == "openai-chat-completions":
        from lm_eval.models.openai_completions import OpenAIChatCompletion
        return OpenAIChatCompletion(model=model_name, **kwargs)
    if model_type == "openai-chat":
        # Like openai-chat-completions but without the lm-eval heuristic that
        # inspects model names for "o1"/"5"/"o3"/"o4" and unconditionally drops
        # stop sequences + forces temperature=1.  Use this type for any OpenAI
        # chat model whose name happens to match those substrings (e.g. gpt-5.*).
        from lm_eval.models.openai_completions import LocalChatCompletion, OpenAIChatCompletion

        class _OpenAIChatModel(OpenAIChatCompletion):
            parse_generations = staticmethod(_parse_generations_with_usage)

            def __init__(self, drop_stop_sequences: bool = False,
                         reasoning_effort: str | None = None, **kw):
                self._drop_stop = drop_stop_sequences
                # Sent explicitly on every request (e.g. "none" for gpt-5.4-nano,
                # whose documented default is also "none").
                self._reasoning_effort = reasoning_effort
                super().__init__(**kw)

            def _create_payload(
                self, messages, generate=False, gen_kwargs=None,
                seed=1234, eos="<|endoftext|>", **kw
            ):
                # Use LocalChatCompletion's payload logic (preserves stop sequences
                # and task-specified temperature), then upgrade max_tokens →
                # max_completion_tokens for the newer OpenAI API.
                payload = LocalChatCompletion._create_payload(
                    self, messages, generate=generate, gen_kwargs=gen_kwargs,
                    seed=seed, eos=eos, **kw
                )
                if "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                if self._drop_stop:
                    payload.pop("stop", None)
                if self._reasoning_effort is not None:
                    payload["reasoning_effort"] = self._reasoning_effort
                # Strip HuggingFace-only generation parameters not accepted by
                # the OpenAI API (e.g. num_beams from task default gen_kwargs).
                _HF_ONLY = {
                    "num_beams", "do_sample", "num_return_sequences",
                    "repetition_penalty", "length_penalty", "no_repeat_ngram_size",
                    "early_stopping", "min_length", "min_new_tokens",
                }
                for key in _HF_ONLY:
                    payload.pop(key, None)
                return payload

        return _OpenAIChatModel(model=model_name, **kwargs)
    if model_type == "vllm":
        from lm_eval.models.vllm_causallms import VLLM
        return VLLM(pretrained=model_name, **kwargs)
    if model_type == "gemini":
        import os as _os
        from lm_eval.models.openai_completions import LocalChatCompletion, OpenAIChatCompletion

        _GEMINI_BASE_URL = (
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )

        class _GeminiChatCompletion(OpenAIChatCompletion):
            parse_generations = staticmethod(_parse_generations_with_usage)

            def __init__(self, base_url=_GEMINI_BASE_URL,
                         disable_thinking: bool = False,
                         reasoning_effort: str | None = None, **kw):
                # Gemini 3.x models cannot turn thinking off; the minimum is
                # reasoning_effort="minimal".  "none" only works on Gemini 2.5
                # non-Pro models.  The old disable_thinking flag was stored but
                # never sent, so refuse it instead of silently ignoring it.
                if disable_thinking:
                    raise ValueError(
                        "disable_thinking is not supported for Gemini; set reasoning_effort "
                        "('none' for Gemini 2.5 non-Pro, 'minimal' for Gemini 3.x)."
                    )
                self._reasoning_effort = reasoning_effort
                super().__init__(base_url=base_url, **kw)

            @property
            def api_key(self):
                key = _os.environ.get("GEMINI_API_KEY")
                if key is None:
                    raise ValueError(
                        "GEMINI_API_KEY environment variable is not set."
                    )
                return key

            def _create_payload(
                self, messages, generate=False, gen_kwargs=None,
                seed=1234, eos=None, **kw
            ):
                # Call LocalChatCompletion._create_payload directly (uses max_tokens,
                # no OpenAI-specific model-name heuristics that break gemini-1.5-flash)
                payload = LocalChatCompletion._create_payload(
                    self, messages, generate=generate, gen_kwargs=gen_kwargs,
                    seed=seed, eos=eos, **kw
                )
                payload.pop("seed", None)
                if self._reasoning_effort is not None:
                    payload["reasoning_effort"] = self._reasoning_effort
                # Gemini rejects stop=[] with a 400 error ("Value is not a string").
                # Remove the key when the stop list is empty so the model generates
                # until max_tokens without explicit stop sequences.
                if "stop" in payload and not payload["stop"]:
                    payload.pop("stop", None)
                if "max_tokens" in payload:
                    # For thinking-capable Gemini models, max_tokens covers both
                    # internal CoT tokens (~1000-1500) and visible output.
                    # Raise the budget so the model has enough room for a
                    # complete response even when thinking is active.
                    payload["max_tokens"] = max(payload["max_tokens"], 4096)
                logger.debug(
                    "Gemini payload keys=%r max_tokens=%r stop=%r",
                    [k for k in payload if k != "messages"],
                    payload.get("max_tokens"),
                    payload.get("stop"),
                )
                return payload

        return _GeminiChatCompletion(model=model_name, **kwargs)
    if model_type in ("llama-cpp", "llamacpp", "llama.cpp"):
        return _make_llama_cpp_model(model_name, **kwargs)
    raise ValueError(
        f"Unknown model type {model_type!r}. "
        "Supported: 'hf'/'huggingface', 'openai', 'openai-chat-completions', 'openai-chat', "
        "'vllm', 'gemini', 'llama-cpp'."
    )


_LLAMA_CPP_DEFAULT_URL = "http://localhost:8080/v1/chat/completions"
_LLAMA_CPP_URL_ENV = "LLAMA_CPP_BASE_URL"


def _resolve_llama_cpp_url(base_url: str | None) -> str:
    """Endpoint URL: config ``base_url`` > ``$LLAMA_CPP_BASE_URL`` > localhost:8080.

    A server root (``http://host:port``) is completed with
    ``/v1/chat/completions``.  Falling back to the default is logged as a
    warning, so a missing ``.env`` entry is noticed.
    """
    import os as _os

    url = base_url or _os.environ.get(_LLAMA_CPP_URL_ENV)
    if not url:
        logger.warning(
            "No base_url in the config and %s is not set; using %s",
            _LLAMA_CPP_URL_ENV, _LLAMA_CPP_DEFAULT_URL,
        )
        return _LLAMA_CPP_DEFAULT_URL
    url = url.rstrip("/")
    if "/v1/" not in url + "/":
        url += "/v1/chat/completions"
    elif url.endswith("/v1"):
        url += "/chat/completions"
    return url


def _make_llama_cpp_model(model_name: str, **kwargs):
    """Chat model served by llama.cpp ``llama-server``, directly or via llama-swap.

    YAML (direct llama-server, started with ``--jinja``)::

        model:
          type: llama-cpp
          name: qwen3.5-9b                 # label; a single server serves one model
          kwargs:
            base_url: http://localhost:8080/v1/chat/completions
            num_concurrent: 1              # match the server's --parallel slots
            disable_thinking: true

    YAML (llama-swap, which routes requests to models by the ``model`` field)::

        model:
          type: llama-cpp
          name: qwen3.5-9b                 # must be the model id configured in llama-swap
          kwargs:
            llama_swap: true               # URL from $LLAMA_CPP_BASE_URL (e.g. in .env)
            disable_thinking: true

    * **API key**: read from the environment variable named by ``api_key_env``
      (default ``LLAMA_CPP_API_KEY``, e.g. set in ``.env``), sent as a Bearer token.
    * **Token counts**: the served model's own ``/tokenize`` endpoint, so counts
      match its vocabulary exactly.  Through llama-swap this is
      ``/upstream/<model>/tokenize``; override with ``tokenize_url``.
    * **Thinking**: ``disable_thinking: true`` sends
      ``chat_template_kwargs: {"enable_thinking": false}`` on every request
      (Qwen3/Gemma chat templates); extra ``chat_template_kwargs`` can be given.
      Reasoning returned by the server is logged per call, so the setting can
      be verified.
    * Log-likelihood tasks are not supported (no prompt log-probs over this API).
    """
    import os as _os

    import requests
    from lm_eval.models.openai_completions import LocalChatCompletion

    class _LlamaCppChatCompletion(LocalChatCompletion):
        parse_generations = staticmethod(_parse_generations_with_usage)

        def __init__(self, base_url: str | None = None, llama_swap: bool = False,
                     api_key_env: str = "LLAMA_CPP_API_KEY", tokenize_url: str | None = None,
                     disable_thinking: bool = False, chat_template_kwargs: dict | None = None,
                     **kw):
            kw.setdefault("tokenizer_backend", None)
            kw.setdefault("tokenized_requests", False)
            base_url = _resolve_llama_cpp_url(base_url)
            super().__init__(base_url=base_url, **kw)
            self.server_url = base_url.split("/v1/")[0].rstrip("/")
            self._api_key_env = api_key_env
            self._chat_template_kwargs = dict(chat_template_kwargs or {})
            if disable_thinking:
                self._chat_template_kwargs["enable_thinking"] = False
            if tokenize_url is None:
                tokenize_url = (
                    f"{self.server_url}/upstream/{self.model}/tokenize" if llama_swap
                    else f"{self.server_url}/tokenize"
                )
            self.tokenize_url = tokenize_url

        @property
        def api_key(self):
            return _os.environ.get(self._api_key_env, "")

        def _create_payload(self, messages, generate=False, gen_kwargs=None,
                            seed=1234, eos=None, **kw):
            payload = super()._create_payload(
                messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos, **kw
            )
            for key in ("num_beams", "num_return_sequences", "length_penalty",
                        "no_repeat_ngram_size", "early_stopping", "min_length",
                        "min_new_tokens", "do_sample"):
                payload.pop(key, None)
            if "stop" in payload and not payload["stop"]:
                payload.pop("stop", None)
            if self._chat_template_kwargs:
                payload["chat_template_kwargs"] = dict(self._chat_template_kwargs)
            return payload

        def count_tokens(self, text: str) -> int:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.post(
                self.tokenize_url,
                json={"content": text, "add_special": False},
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            return len(response.json()["tokens"])

    return _LlamaCppChatCompletion(model=model_name, **kwargs)


def _collect_tasks(node, out: dict) -> None:
    """Recursively collect lm-eval task objects from an arbitrary nested structure.

    lm-eval returns ``{ConfigurableGroup: nested_value}`` dicts for groups.
    Any object exposing ``test_docs`` is treated as a leaf task.
    """
    if isinstance(node, dict):
        for key, val in node.items():
            if val is None:
                continue
            if hasattr(val, "test_docs"):
                name = key if isinstance(key, str) else (
                    getattr(key, "task", None) or str(key)
                )
                out[name] = val
            else:
                _collect_tasks(val, out)
    elif isinstance(node, (list, tuple)):
        for item in node:
            if hasattr(item, "test_docs"):
                out.setdefault(str(len(out)), item)
            else:
                _collect_tasks(item, out)


def _resolve_group_members(tm, group_name: str) -> list[str]:
    """Return individual task names belonging to *group_name* via ``task_index``."""
    task_index = getattr(tm, "task_index", None) or {}
    members: list[str] = []
    for name, config in task_index.items():
        if not isinstance(config, dict):
            continue
        groups = config.get("group", [])
        if isinstance(groups, str):
            groups = [groups]
        if group_name in (groups or []):
            members.append(name)
    return members


def _load_task(task_name: str, num_fewshot: int = 0):
    """Return an lm-eval task object, or a ``dict[str, task]`` for groups.

    Single tasks (e.g. ``"hellaswag"``) return the task object directly.
    Group names (e.g. ``"mmlu"``) return a dict mapping each subtask name to
    its task object.  Both forms are accepted by :class:`LMEvalDataset` and
    :class:`LMEvalEvaluator`.
    """
    _require_lm_eval()
    from lm_eval.tasks import TaskManager

    tm = TaskManager()
    task_dict = tm.load_task_or_group([task_name])

    # Strategy 1: single task — direct string key lookup
    if task_name in task_dict:
        task = task_dict[task_name]
        if hasattr(task, "set_fewshot_seed"):
            task.set_fewshot_seed(1234)
        return task

    # Strategy 2: group — collect task objects from nested lm-eval structure
    collected: dict = {}
    _collect_tasks(task_dict, collected)
    if collected:
        logger.info("Loaded task group %r with %d subtasks.", task_name, len(collected))
        for t in collected.values():
            if hasattr(t, "set_fewshot_seed"):
                t.set_fewshot_seed(1234)
        return collected

    # Strategy 3: group — load each subtask individually by scanning task_index
    member_names = _resolve_group_members(tm, task_name)
    if member_names:
        logger.info("Loading group %r: %d subtasks.", task_name, len(member_names))
        tasks: dict = {}
        for name in member_names:
            try:
                sub = tm.load_task_or_group([name])
                if name in sub and hasattr(sub[name], "test_docs"):
                    t = sub[name]
                    if hasattr(t, "set_fewshot_seed"):
                        t.set_fewshot_seed(1234)
                    tasks[name] = t
                else:
                    logger.warning("Subtask %r could not be loaded; skipping.", name)
            except Exception as exc:
                logger.warning("Error loading subtask %r: %s; skipping.", name, exc)
        if tasks:
            return tasks

    raise ValueError(
        f"Could not load {task_name!r} as a single task or group. "
        f"Keys returned by lm-eval: {[str(k) for k in task_dict]!r}. "
        f"Use 'lm-eval --tasks list' to browse available task names."
    )


def _make_instance(request_type: str, doc, arguments, idx: int = 0):
    """Create an lm-eval Instance, handling API differences across versions."""
    try:
        from lm_eval.api.instance import Instance
        return Instance(
            request_type=request_type,
            doc=doc,
            arguments=arguments,
            idx=idx,
        )
    except ImportError:
        # Older lm-eval versions that don't have Instance; return raw tuple.
        return arguments


def _call_model(model, method: str, requests: list):
    """Call ``model.loglikelihood`` or ``model.generate_until``, handling
    both Instance-based (lm-eval >= 0.4) and tuple-based (older) APIs."""
    fn = getattr(model, method)
    return fn(requests)


def _is_chat_model(model) -> bool:
    """Return True when *model* is an OpenAI-style chat-completions model.

    Chat models require prompts to be delivered as ``JsonChatStr`` objects
    (JSON-encoded lists of ``{"role": ..., "content": ...}`` dicts) rather
    than plain strings.  They also do not support ``loglikelihood``.
    """
    try:
        from lm_eval.models.openai_completions import LocalChatCompletion
        return isinstance(model, LocalChatCompletion)
    except ImportError:
        return False


def _hf_model_has_chat_template(model) -> bool:
    """Return True when *model*'s tokenizer has a configured chat template.

    ``apply_chat_template`` is defined on the base ``PreTrainedTokenizer``
    class and therefore exists on *all* HF tokenizers — including base models
    like GPT-2.  The correct signal is ``tokenizer.chat_template is not None``,
    which is only set for instruct/chat checkpoints (Qwen, Llama-Instruct, …).
    A string tokenizer (self-hosted OpenAI-compatible servers) is excluded.
    """
    tokenizer = getattr(model, "tokenizer", None)
    return (
        tokenizer is not None
        and not isinstance(tokenizer, str)
        and getattr(tokenizer, "chat_template", None) is not None
    )


def _wrap_chat_prompt(prompt: str, system_prompt: str | None = None):
    """Wrap *prompt* in a ``JsonChatStr`` for OpenAI chat-completions models.

    Returns a ``JsonChatStr`` whose ``.prompt`` field is a JSON array of
    ``{"role": ..., "content": ...}`` dicts as expected by lm-eval's
    ``OpenAIChatCompletion._create_payload``.
    """
    import json
    try:
        from lm_eval.models.api_models import JsonChatStr
    except ImportError:
        raise ImportError("lm-eval >= 0.4 is required for chat model support")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return JsonChatStr(json.dumps(messages))


# ---------------------------------------------------------------------------
# 0. Model-backed tokenizer
# ---------------------------------------------------------------------------

class LMEvalTokenizer(Tokenizer):
    """Count tokens using the same tokenizer the model uses.

    Wraps the underlying tokenizer exposed by the lm-eval model object so that
    token-count objectives reflect the model's actual vocabulary rather than a
    whitespace approximation.

    Detection order
    ---------------
    1. **``model.tokenizer`` is a tokenizer object** — covers HuggingFace
       (``HFLM``), vLLM, OpenAI via lm-eval (uses tiktoken internally), and
       Anthropic Claude via lm-eval (uses ``client.get_tokenizer()``).
       Anything whose ``.encode(text)`` returns a sequence of tokens is handled
       here.  Plain Python ``str`` objects (which also have ``.encode()``) are
       excluded explicitly.
    2. **``model.encoding_name`` / ``model.tokenizer_name``** — tiktoken
       encoding name exposed by some lm-eval OpenAI wrappers.  ``tiktoken``
       must be installed (``pip install tiktoken``).
    3. **``model.tokenizer`` is a string** (model or encoding name) — covers
       self-hosted OpenAI-compatible servers (``LocalCompletionsAPI``).
       Tries tiktoken first, then falls back to loading a HuggingFace
       ``AutoTokenizer`` by that name (requires ``transformers``).
    4. **Fallback** — :class:`~prompt_piper.optimization.tokenizer.WhitespaceTokenizer`
       with a logged warning.

    Parameters
    ----------
    model:
        An lm-eval LM object (``HFLM``, ``OpenAICompletionsAPI``,
        ``LocalCompletionsAPI``, ``AnthropicLM``, ``VLLM``, …).

    Notes
    -----
    *Anthropic SDK deprecation*: ``client.get_tokenizer()`` was removed in
    Anthropic SDK ≥ 0.20.  If ``AnthropicLM`` fails to initialise because of
    this, lm-eval itself will error before ``LMEvalTokenizer`` is ever called.
    When using a recent SDK and a Claude model, pass an explicit ``tokenizer``
    to :meth:`LMEvalBackend.build_experiment` — e.g. a
    :class:`~prompt_piper.optimization.tokenizer.TiktokenTokenizer` with
    ``"cl100k_base"`` as a reasonable approximation.

    *Self-hosted models*: for models accessed via an OpenAI-compatible API
    whose name cannot be resolved to a tiktoken encoding (e.g. ``"llama3"``),
    detection falls through to :class:`WhitespaceTokenizer`.  Pass an explicit
    tokenizer or load the HuggingFace tokenizer manually in that case.
    """

    def __init__(self, model) -> None:
        self._counter = self._build_counter(model)

    # ------------------------------------------------------------------
    # Tokenizer interface
    # ------------------------------------------------------------------

    def count(self, text: str) -> int:
        return self._counter(text)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_counter(model):
        """Return a callable ``(text: str) -> int`` for the given model."""
        tok_attr = getattr(model, "tokenizer", None)

        # ---- 0. model-provided exact counter (e.g. llama.cpp /tokenize) --
        count_fn = getattr(model, "count_tokens", None)
        if callable(count_fn):
            return count_fn

        # ---- 1. tokenizer object with .encode() -------------------------
        # Explicitly exclude plain strings — str.encode() returns bytes, not
        # tokens, so len(str.encode(text)) would silently be a byte count.
        if tok_attr is not None and not isinstance(tok_attr, str):
            encode_fn = getattr(tok_attr, "encode", None)
            if callable(encode_fn):
                def _obj_count(text: str, _enc=encode_fn) -> int:
                    return len(_enc(text))
                return _obj_count

        # ---- 2. tiktoken via encoding_name / tokenizer_name -------------
        # Some lm-eval OpenAI wrappers set model.encoding_name or
        # model.tokenizer_name to a tiktoken-resolvable string.
        try:
            import tiktoken  # noqa: PLC0415
            enc_name = (
                getattr(model, "encoding_name", None)
                or getattr(model, "tokenizer_name", None)
            )
            if enc_name and isinstance(enc_name, str):
                try:
                    enc = tiktoken.encoding_for_model(enc_name)
                except KeyError:
                    enc = tiktoken.get_encoding(enc_name)
                def _tiktoken_count(text: str, _enc=enc) -> int:
                    return len(_enc.encode(text))
                return _tiktoken_count
        except ImportError:
            pass

        # ---- 3. model.tokenizer is a string (e.g. LocalCompletionsAPI) --
        # Try to resolve the model/encoding name to a real tokenizer.
        if isinstance(tok_attr, str) and tok_attr:
            # 3a. tiktoken — works for OpenAI model names and encoding names.
            try:
                import tiktoken  # noqa: PLC0415
                try:
                    enc = tiktoken.encoding_for_model(tok_attr)
                except KeyError:
                    try:
                        enc = tiktoken.get_encoding(tok_attr)
                    except ValueError:
                        enc = None
                if enc is not None:
                    def _name_tiktoken_count(text: str, _enc=enc) -> int:
                        return len(_enc.encode(text))
                    return _name_tiktoken_count
            except ImportError:
                pass

            # 3b. HuggingFace AutoTokenizer — works for HF model names used
            # as the tokenizer string (e.g. "mistralai/Mistral-7B-v0.1").
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415
                hf_tok = AutoTokenizer.from_pretrained(tok_attr)
                logger.info(
                    "LMEvalTokenizer: loaded HuggingFace tokenizer %r from model name.",
                    tok_attr,
                )
                def _hf_name_count(text: str, _tok=hf_tok) -> int:
                    return len(_tok.encode(text))
                return _hf_name_count
            except Exception:
                pass

        # ---- 4. OpenAI chat models: resolve via model name attribute ----
        # OpenAIChatCompletion stores the model name in ``model.model``
        # (e.g. "gpt-4o").  Try tiktoken before falling back to whitespace.
        # For unknown model names (e.g. gpt-5.x-mini-*), fall back to
        # o200k_base (GPT-4o / GPT-5 family) then cl100k_base (GPT-4).
        model_name_attr = getattr(model, "model", None)
        if isinstance(model_name_attr, str) and model_name_attr:
            try:
                import tiktoken  # noqa: PLC0415
                enc = None
                try:
                    enc = tiktoken.encoding_for_model(model_name_attr)
                except KeyError:
                    for _fallback in ("o200k_base", "cl100k_base"):
                        try:
                            enc = tiktoken.get_encoding(_fallback)
                            logger.info(
                                "LMEvalTokenizer: model %r not in tiktoken registry; "
                                "using fallback encoding %r.",
                                model_name_attr, _fallback,
                            )
                            break
                        except Exception:
                            pass
                if enc is not None:
                    def _openai_name_tiktoken(text: str, _enc=enc) -> int:
                        return len(_enc.encode(text))
                    logger.info(
                        "LMEvalTokenizer: using tiktoken %r for model %r.",
                        enc.name, model_name_attr,
                    )
                    return _openai_name_tiktoken
            except ImportError:
                pass

        # ---- 5. Fallback: whitespace ------------------------------------
        logger.warning(
            "LMEvalTokenizer: could not resolve a tokenizer for model %r; "
            "falling back to whitespace splitting.  For accurate token counts "
            "pass an explicit tokenizer= to build_experiment().",
            type(model).__name__,
        )
        return WhitespaceTokenizer().count


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class LMEvalDataset:
    """Extract samples from an lm-eval task or task group.

    Each sample is a dict with:

    * ``"prompt"`` — the fully formatted context string (with few-shot
      examples if ``num_fewshot > 0``).
    * ``"target"`` — the expected output / label.
    * ``"_doc"`` — the raw lm-eval document (needed by
      :class:`LMEvalEvaluator`).
    * ``"_task_name"`` — *(groups only)* subtask name used by
      :class:`LMEvalEvaluator` to route evaluation to the correct task.

    Parameters
    ----------
    task:
        An initialised lm-eval task object **or** a ``dict[str, task]``
        returned by :func:`_load_task` for groups.
    split:
        Which split to use: ``"test"``, ``"validation"``/``"val"``,
        or ``"train"``.
    num_fewshot:
        Number of few-shot examples to prepend.
    limit:
        Cap the number of samples returned (``None`` = all).
    prompt_field:
        Key used for the full (few-shot) prompt in the output dict.
    template_field:
        When set, each sample also gets an entry under this key containing
        the 0-shot prompt (i.e. ``fewshot_context(doc, 0)`` — the prompt
        with data already substituted but no few-shot examples prepended).
        Use ``"prompt_0shot"`` as the conventional name and pass it as
        ``compress_field`` in :class:`ExperimentConfig` to compress that
        field instead of the full formatted prompt.
    shots_field:
        When set (and ``template_field`` is also set), each sample gets an
        entry under this key containing the plain-text few-shot block — the
        text that precedes the 0-shot prompt in the full plain-text context.
        Defaults to ``"_shots"``.  The value is ``""`` when ``num_fewshot``
        is 0 or when the boundary cannot be detected.
    """

    def __init__(
        self,
        task,
        split: str = "test",
        num_fewshot: int = 0,
        limit: int | None = None,
        prompt_field: str = "prompt",
        template_field: str | None = None,
        shots_field: str | None = "_shots",
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
    ) -> None:
        if isinstance(task, dict):
            self._task = None
            self._tasks: dict | None = task   # {task_name: task_obj}
        else:
            self._task = task
            self._tasks = None
        self._split = split
        self._num_fewshot = num_fewshot
        self._limit = limit
        self._prompt_field = prompt_field
        self._template_field = template_field
        self._shots_field = shots_field
        self._apply_chat_template = apply_chat_template
        self._fewshot_as_multiturn = fewshot_as_multiturn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        if self._tasks is not None:
            raw: list[tuple] = []
            for task_name, task_obj in self._tasks.items():
                for doc in self._get_docs_for_task(task_obj):
                    raw.append((doc, task_name, task_obj))
        else:
            raw = [(doc, None, self._task) for doc in self._get_docs()]

        if self._limit is not None:
            raw = raw[: self._limit]
        for doc, task_name, task_obj in raw:
            yield self._doc_to_sample(doc, task_name=task_name, task_obj=task_obj)

    def __len__(self) -> int:
        if self._tasks is not None:
            total = sum(
                sum(1 for _ in self._get_docs_for_task(t))
                for t in self._tasks.values()
            )
        else:
            total = sum(1 for _ in self._get_docs())
        if self._limit is not None:
            return min(total, self._limit)
        return total

    def as_list(self) -> list[dict]:
        return list(self)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_task_name(
        cls,
        task_name: str,
        split: str = "test",
        num_fewshot: int = 0,
        limit: int | None = None,
        prompt_field: str = "prompt",
        template_field: str | None = None,
        shots_field: str | None = "_shots",
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
    ) -> "LMEvalDataset":
        """Convenience constructor — loads the task by name."""
        task = _load_task(task_name, num_fewshot=num_fewshot)
        return cls(task, split=split, num_fewshot=num_fewshot, limit=limit,
                   prompt_field=prompt_field, template_field=template_field,
                   shots_field=shots_field,
                   apply_chat_template=apply_chat_template,
                   fewshot_as_multiturn=fewshot_as_multiturn)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_docs_for_task(self, task_obj):
        split = self._split.lower()
        if split == "test":
            return task_obj.test_docs()
        if split in ("validation", "val"):
            return task_obj.validation_docs()
        if split == "train":
            return task_obj.training_docs()
        raise ValueError(f"Unknown split {self._split!r}")

    def _get_docs(self):
        return self._get_docs_for_task(self._task)

    def _doc_to_sample(self, doc: dict, task_name=None, task_obj=None) -> dict:
        task = task_obj if task_obj is not None else self._task
        chat_kwargs = {}
        if self._apply_chat_template or self._fewshot_as_multiturn:
            chat_kwargs = {
                "apply_chat_template": self._apply_chat_template,
                "fewshot_as_multiturn": self._fewshot_as_multiturn,
            }
        try:
            prompt = task.fewshot_context(doc, self._num_fewshot, **chat_kwargs)
        except TypeError:
            prompt = task.fewshot_context(
                doc, self._num_fewshot, rnd=None, description=None
            )
        target = task.doc_to_target(doc)
        sample = {
            self._prompt_field: prompt,
            "target": target,
            "_doc": doc,
        }
        if task_name is not None:
            sample["_task_name"] = task_name
        if self._template_field is not None:
            # 0-shot template: always raw text (no chat template) so transformations
            # operate on plain instruction strings, not pre-formatted chat turns.
            try:
                template = task.fewshot_context(doc, 0)
            except TypeError:
                template = task.fewshot_context(doc, 0, rnd=None, description=None)
            sample[self._template_field] = template

            if self._shots_field is not None:
                sample.update(self._structure(task, doc, template))

        return sample


    def _structure(self, task, doc, template: str) -> dict:
        """Cut lm-eval's exact plain-text prompt into its parts.

        lm-eval renders a plain-text prompt as ``description + shots +
        doc_to_text(doc) + tail`` (``tail`` = delimiter + ``gen_prefix`` when the
        task defines one, e.g. MBPP's code-block opener).  The parts are taken
        from lm-eval's own output, so re-assembling them reproduces its prompt
        exactly:

        * ``_description`` — static task description (goes to the instruction section);
        * ``_shots`` — the few-shot block, including its trailing delimiter;
        * ``_input_text`` — the per-sample text (question + answer cue);
        * ``_tail`` — text lm-eval appends after the sample text (never compressed);
        * ``_reference_prompt`` — lm-eval's full prompt, for verification.

        Raises ``ValueError`` when the prompt cannot be cut consistently, instead
        of silently dropping the shots.
        """
        text = task.doc_to_text(doc)
        if not isinstance(text, str):
            raise ValueError(f"doc_to_text returned {type(text).__name__}; cannot locate the sample text")
        gen_prefix = None
        if hasattr(task, "doc_to_prefix"):
            try:
                gen_prefix = task.doc_to_prefix(doc)
            except Exception:
                gen_prefix = None
        try:
            reference = task.fewshot_context(doc, self._num_fewshot, gen_prefix=gen_prefix)
        except TypeError:
            reference = task.fewshot_context(doc, self._num_fewshot)
        pos_zero = template.rfind(text)
        pos_full = reference.rfind(text)
        if pos_zero < 0 or pos_full < 0:
            raise ValueError("sample text not found in lm-eval's prompt")
        description = template[:pos_zero]
        if not reference.startswith(description):
            raise ValueError("few-shot prompt does not start with the task description")
        shots = reference[len(description):pos_full]
        if self._num_fewshot > 0 and not shots.strip():
            raise ValueError(f"expected {self._num_fewshot} shots but found none")
        return {
            "_description": description,
            self._shots_field: shots,
            "_input_text": text,
            "_tail": reference[pos_full + len(text):],
            "_reference_prompt": reference,
        }


def _first_fenced_block(text: str) -> str:
    """Content of the first complete ```...``` block that starts on its own line."""
    m = re.search(r"(?:^|\n)```[\w+-]*\n(.*?)\n?```", text, re.DOTALL)
    return m.group(1) if m and m.group(1).strip() else ""


def _is_python(code: str) -> bool:
    import ast

    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return bool(code.strip())


# ---------------------------------------------------------------------------
# 2. Evaluator
# ---------------------------------------------------------------------------

class LMEvalEvaluator(Evaluator):
    """Score a compressed prompt using an lm-eval model and task.

    Dispatches to the correct evaluation strategy based on the task's
    ``OUTPUT_TYPE``:

    * ``"multiple_choice"`` / ``"loglikelihood"`` — compare per-choice
      log-probabilities; return 1.0 if the predicted choice matches the gold
      label, 0.0 otherwise.
    * ``"generate_until"`` — generate text, then delegate to the task's
i      ``process_results`` metric.

    Parameters
    ----------
    model:
        An lm-eval LM object (e.g. ``HFLM``).
    task:
        The lm-eval task object corresponding to the benchmark.
    generation_kwargs:
        Extra kwargs for generation tasks (``until``, ``max_gen_toks``, …).
        Defaults to a reasonable stopping condition.
    metric_key:
        Which key from ``process_results`` to use as the scalar score.
        ``None`` (default) takes the first value in the returned dict,
        which matches the first metric lm-eval would report for the task.
        Pass an explicit key — e.g. ``"exact_match,flexible-extract"`` —
        to pin the optimisation to a specific lm-eval metric.
    """

    def __init__(
        self,
        model,
        task,
        generation_kwargs: dict | None = None,
        extra_stop_sequences: list[str] | None = None,
        metric_key: str | None = None,
        apply_chat_template: bool = False,
    ) -> None:
        self._model = model
        self._apply_chat = apply_chat_template
        if isinstance(task, dict):
            self._task = None
            self._tasks: dict | None = task   # {task_name: task_obj}
            first_task = next(iter(task.values()))
        else:
            self._task = task
            self._tasks = None
            first_task = task

        # Always start from the task's own generation_kwargs so that task-specific
        # until sequences (e.g. GSM8K's ["Question:", "</s>"]) are preserved.
        # User-supplied generation_kwargs are merged on top (user wins on conflicts).
        cfg = getattr(first_task, "config", None)
        task_kwargs = dict(getattr(cfg, "generation_kwargs", None) or {})
        task_kwargs.setdefault("max_gen_toks", 256)
        # Only add a default stop-sequence when the task provided none at all.
        # An explicit until=[] from the task config means "no stop sequences"
        # (e.g. MBPP needs multi-line code generation) — preserve it.
        if "until" not in task_kwargs:
            task_kwargs["until"] = ["\n", "\n\n"]
        if generation_kwargs is not None:
            self._gen_kwargs = {**task_kwargs, **generation_kwargs}
        else:
            self._gen_kwargs = task_kwargs

        # Merge model-specific stop tokens (e.g. "<|eot_id|>" for Llama-3)
        # into the until list without disturbing other generation kwargs.
        if extra_stop_sequences:
            until = list(self._gen_kwargs.get("until", []))
            for seq in extra_stop_sequences:
                if seq not in until:
                    until.append(seq)
            self._gen_kwargs = {**self._gen_kwargs, "until": until}

        # Some API backends (e.g. Gemini) reject stop=[] with a 400 error.
        # When the final until list is empty, remove the key entirely so the
        # model generates up to max_gen_toks without explicit stop sequences.
        if "until" in self._gen_kwargs and not self._gen_kwargs["until"]:
            del self._gen_kwargs["until"]

        # Strip newline-based stop sequences for chat/instruct models.
        # lm-eval tasks like BBH set until=['\n\n'] as a completion-style
        # separator between few-shot examples. Chat API models (Gemini, GPT,
        # OpenAI-compatible) and HF instruct models with chat templates use
        # structured messages, not raw text separators, so these stops cause
        # premature truncation before the model completes its chain-of-thought.
        # Only strip when the caller did not explicitly supply an 'until' key —
        # an explicit override is honoured as-is.
        _user_set_until = generation_kwargs is not None and "until" in generation_kwargs
        if (
            not _user_set_until
            and "until" in self._gen_kwargs
            and (_is_chat_model(self._model) or apply_chat_template)
        ):
            _stripped = [s for s in self._gen_kwargs["until"] if s not in ("\n", "\n\n")]
            if len(_stripped) != len(self._gen_kwargs["until"]):
                logger.info(
                    "LMEvalEvaluator: stripped newline stop sequences for chat model "
                    "(remaining: %r). Pass generation_kwargs={'until': [...]} to override.",
                    _stripped,
                )
            self._gen_kwargs = {**self._gen_kwargs, "until": _stripped}
            if not self._gen_kwargs["until"]:
                del self._gen_kwargs["until"]

        self._metric_key = metric_key

    def evaluate(self, original: str, compressed: str, sample: dict) -> float:
        doc = sample.get("_doc")
        if doc is None:
            logger.warning(
                "LMEvalEvaluator: sample missing '_doc' key; "
                "returning 0.0.  Use LMEvalDataset to build samples."
            )
            return 0.0

        task = self._get_task(sample)
        output_type = getattr(task, "OUTPUT_TYPE", "generate_until")

        if output_type in ("multiple_choice", "loglikelihood"):
            return self._eval_multiple_choice(compressed, doc, task)
        return self._eval_generation(compressed, doc, task)

    def evaluate_batch(
        self,
        originals: list[str],
        compresseds: list[str],
        samples: list[dict],
    ) -> list[float]:
        """Score a batch of compressed prompts in a single model call."""
        return [d["score"] for d in self.evaluate_batch_detailed(originals, compresseds, samples)]

    def evaluate_batch_detailed(
        self,
        originals: list[str],
        compresseds: list[str],
        samples: list[dict],
    ) -> list[dict]:
        """Score a batch and return per-sample details (output, metrics, …)."""
        if not samples:
            return []

        output_type = getattr(self._get_task(samples[0]), "OUTPUT_TYPE", "generate_until")
        if output_type in ("multiple_choice", "loglikelihood"):
            return self._eval_multiple_choice_batch(compresseds, samples, detailed=True)
        return self._eval_generation_batch(compresseds, samples, detailed=True)

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _eval_multiple_choice(self, prompt: str, doc, task) -> float:
        """Loglikelihood-based scoring for MC tasks."""
        try:
            choices = task.doc_to_choice(doc)
        except AttributeError:
            # Task doesn't expose choices; fall back to generation.
            return self._eval_generation(prompt, doc, task)

        requests = [
            _make_instance("loglikelihood", doc, (prompt, choice), i)
            for i, choice in enumerate(choices)
        ]
        results = _call_model(self._model, "loglikelihood", requests)
        log_probs = [r[0] if isinstance(r, (tuple, list)) else r for r in results]
        predicted = int(np.argmax(log_probs))

        gold = task.doc_to_target(doc)
        # Gold may be an int (index) or a string (choice text).
        if isinstance(gold, int):
            return 1.0 if predicted == gold else 0.0
        # String: compare to the predicted choice text.
        return 1.0 if choices[predicted] == gold else 0.0

    def _eval_multiple_choice_batch(
        self, prompts: list[str], samples: list[dict], detailed: bool = False
    ) -> list:
        """Batched loglikelihood scoring: one model call for all samples.

        With ``detailed=True`` returns one dict per sample (score, per-choice
        log-probabilities, predicted index, gold) instead of bare scores.
        """
        first_task = self._get_task(samples[0])
        first_doc = samples[0].get("_doc")
        try:
            first_task.doc_to_choice(first_doc)
        except AttributeError:
            return self._eval_generation_batch(prompts, samples, detailed=detailed)

        # Empty prompts (fully compressed away) cannot be tokenised; score 0.
        scores: list[float] = [0.0] * len(prompts)
        details: list[dict] = [
            {"score": 0.0, "error": "empty prompt"} for _ in prompts
        ]
        valid = [(i, p, s) for i, (p, s) in enumerate(zip(prompts, samples)) if p.strip()]
        if not valid:
            return details if detailed else scores

        requests = []
        choices_counts: list[int] = []
        valid_indices: list[int] = []
        for orig_i, prompt, sample in valid:
            task = self._get_task(sample)
            doc = sample.get("_doc")
            choices = task.doc_to_choice(doc)
            choices_counts.append(len(choices))
            valid_indices.append(orig_i)
            for i, choice in enumerate(choices):
                requests.append(_make_instance("loglikelihood", doc, (prompt, choice), i))

        results = _call_model(self._model, "loglikelihood", requests)

        idx = 0
        for orig_i, n_choices, (_, _, sample) in zip(
            valid_indices, choices_counts, valid
        ):
            chunk = results[idx : idx + n_choices]
            idx += n_choices
            log_probs = [r[0] if isinstance(r, (tuple, list)) else r for r in chunk]
            predicted = int(np.argmax(log_probs))
            task = self._get_task(sample)
            doc = sample.get("_doc")
            gold = task.doc_to_target(doc)
            choices = task.doc_to_choice(doc)
            if isinstance(gold, int):
                scores[orig_i] = 1.0 if predicted == gold else 0.0
            else:
                scores[orig_i] = 1.0 if choices[predicted] == gold else 0.0
            details[orig_i] = {
                "score": scores[orig_i],
                "loglikelihoods": [float(lp) for lp in log_probs],
                "choices": [str(c) for c in choices],
                "predicted": predicted,
                "gold": gold,
            }
        return details if detailed else scores

    def _eval_generation(self, prompt: str, doc, task) -> float:
        """Generation-based scoring for open-ended tasks."""
        if _is_chat_model(self._model):
            actual_prompt = _wrap_chat_prompt(prompt)
        else:
            actual_prompt = self._apply_chat_template(prompt)
        instance = _make_instance("generate_until", doc, (actual_prompt, self._gen_kwargs), 0)
        generated = _call_model(self._model, "generate_until", [instance])[0]
        filtered = self._apply_filter_list(generated, task, doc)

        try:
            metrics: dict = task.process_results(doc, filtered)
            value = float(self._pick_metric(metrics))
            return max(0.0, min(1.0, value))
        except Exception as exc:
            logger.warning("LMEvalEvaluator: process_results failed (%s); returning 0.0", exc)
            return 0.0

    def _eval_generation_batch(
        self, prompts: list[str], samples: list[dict], detailed: bool = False
    ) -> list:
        """Batched generation scoring: one model call for all samples.

        With ``detailed=True`` returns one dict per sample (score, raw output,
        the exact prompt sent, filtered answer, full metrics, error) instead
        of bare scores.
        """
        # Empty prompts (fully compressed away) produce a zero-length token
        # tensor, causing an IndexError in HFLM._model_generate when it
        # accesses attention_mask[:, -1].  Skip them and return 0.0.
        scores: list[float] = [0.0] * len(prompts)
        details: list[dict] = [
            {"score": 0.0, "error": "empty prompt"} for _ in prompts
        ]
        valid = [(i, p, s) for i, (p, s) in enumerate(zip(prompts, samples)) if p.strip()]
        if not valid:
            return details if detailed else scores

        use_chat = _is_chat_model(self._model)
        requests = [
            _make_instance(
                "generate_until",
                sample.get("_doc", {}),
                (
                    _wrap_chat_prompt(prompt) if use_chat
                    else self._apply_chat_template(prompt),
                    self._gen_kwargs,
                ),
                i,
            )
            for i, (_, prompt, sample) in enumerate(valid)
        ]
        generated_texts = _call_model(self._model, "generate_until", requests)

        for (orig_i, _, sample), generated, request in zip(valid, generated_texts, requests):
            task = self._get_task(sample)
            doc = sample.get("_doc")
            filtered = self._apply_filter_list(generated, task, doc)
            detail: dict = {
                "output": str(generated),
                "provider_usage": getattr(generated, "usage", None),
                "provider_reasoning": getattr(generated, "reasoning", None),
                "sent_prompt": str(request.args[0] if hasattr(request, "args") else request[0]),
                "filtered": filtered,
                "gen_kwargs": self._gen_kwargs,
            }
            try:
                metrics: dict = task.process_results(doc, filtered)
                value = float(self._pick_metric(metrics))
                scores[orig_i] = max(0.0, min(1.0, value))
                detail["metrics"] = metrics
            except Exception as exc:
                logger.warning(
                    "LMEvalEvaluator: process_results failed (%s); returning 0.0", exc
                )
                detail["error"] = f"process_results failed: {exc!r}"
            detail["score"] = scores[orig_i]
            details[orig_i] = detail
        return details if detailed else scores

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    def _get_task(self, sample: dict):
        """Return the task object for this sample.

        For single-task evaluators returns ``self._task`` directly.  For
        group evaluators (``self._task is None``) routes to the correct
        subtask via ``sample["_task_name"]``, falling back to the first
        subtask when the key is absent.
        """
        if self._task is not None:
            return self._task
        task_name = sample.get("_task_name")
        if task_name and self._tasks and task_name in self._tasks:
            return self._tasks[task_name]
        if self._tasks:
            return next(iter(self._tasks.values()))
        return None

    def _pick_metric(self, metrics: dict) -> float:
        """Extract the scalar score from a ``process_results`` metrics dict.

        Uses ``self._metric_key`` when set; otherwise falls back to the first
        value in the dict (insertion order matches lm-eval's reporting order).
        Raises ``KeyError`` with a helpful message when an explicit key is not
        found so the caller can surface it as a warning.
        """
        if self._metric_key is not None:
            if self._metric_key not in metrics:
                available = list(metrics.keys())
                raise KeyError(
                    f"metric_key {self._metric_key!r} not found in process_results output. "
                    f"Available keys: {available}"
                )
            return metrics[self._metric_key]
        return next(iter(metrics.values()))

    def _apply_chat_template(self, prompt: str) -> str:
        """Wrap the prompt in the model's chat template for instruct/chat models.

        lm-eval's ``generate_until`` tokenises the raw string directly via
        ``tok_encode`` without applying any chat template.  For instruct models
        loaded with ``apply_chat_template=True`` this means the model receives
        plain text instead of the ``<|user|>…<|assistant|>`` tokens it was
        fine-tuned on, causing poor or off-format generation.
        """
        if not self._apply_chat:
            return prompt
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            return prompt
        try:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception as exc:
            logger.warning(
                "LMEvalEvaluator: chat template failed (%s); using raw prompt.", exc
            )
            return prompt

    def _apply_filter_list(self, generated: str, task, doc: dict | None = None) -> list[str]:
        """Apply each filter in the task's ``filter_list`` independently.

        The full lm-eval harness applies every filter spec (e.g. GSM8K's
        strict-match *and* flexible-extract) to the raw generated text and
        passes the full list of results to ``process_results``.  We reproduce
        that here: each filter spec is run independently and its output is
        collected in a list whose order matches the filter_list order.

        Supports two filter function types:

        * ``"regex"`` — standard regex extraction (group_select, etc.).
        * ``"custom"`` — calls ``filter_fn([[result]], [doc])`` as a 1-sample
          batch, matching the signature used by tasks like MBPP whose
          ``build_predictions`` function extracts code from the raw generation.

        When a filter finds no match it falls back to the raw generated text
        for that slot, so ``process_results`` always receives a list of the
        same length as ``filter_list``.

        When the task has no filter_list, ``[generated]`` is returned so the
        call to ``process_results`` is still valid.
        """
        filter_list = (
            getattr(getattr(task, "config", None), "filter_list", None) or []
        )
        if not filter_list:
            return [generated]

        results: list[str] = []
        for filter_spec in filter_list:
            result = generated
            ok = True
            fallback = None
            for step in filter_spec.get("filter", []):
                fn = step.get("function")
                if fn == "regex":
                    # lm-eval's RegexFilter returns its fallback ("[invalid]" by
                    # default) when nothing matches; never the raw generation.
                    fallback = step.get("fallback", "[invalid]")
                    pattern = step.get("regex_pattern", "")
                    # group_select picks which match from finditer (0=first, -1=last)
                    group_select = step.get("group_select", 0)
                    try:
                        matches = list(re.finditer(pattern, result))
                        if not matches:
                            ok = False
                            break
                        m = matches[group_select]
                        # If the pattern has capture groups take the first non-empty one;
                        # otherwise use the full match.
                        if m.lastindex:
                            result = next(
                                (m.group(i) for i in range(1, m.lastindex + 1) if m.group(i)),
                                m.group(0),
                            )
                        else:
                            result = m.group(0)
                    except (re.error, IndexError):
                        ok = False
                        break
                elif fn == "custom":
                    # Custom batch filter: filter_fn(resps_batch, docs_batch)
                    # where resps_batch[i] is a list of candidates for sample i.
                    # Wrap as a 1-sample batch and unwrap the result.
                    filter_fn = step.get("filter_fn")
                    if callable(filter_fn):
                        try:
                            batch_out = filter_fn([[result]], [doc or {}])
                            extracted = batch_out[0][0] if (batch_out and batch_out[0]) else ""
                            # lm-eval's MBPP extractor assumes the response
                            # continues a prefilled "```python" opener, and fails
                            # on (a) continuations starting with "def" (its regex
                            # takes "def" as a language tag), (b) chat models,
                            # which cannot be prefilled and may write prose before
                            # their own block.  Use the first candidate that is
                            # valid Python, lm-eval's own extraction first.
                            candidates = [
                                extracted,
                                _first_fenced_block(result),
                                result.split("```")[0],
                            ]
                            valid = next((c for c in candidates if _is_python(c)), None)
                            if valid is not None:
                                result = valid
                            elif extracted:
                                result = extracted
                            else:
                                # filter_fn returned empty (e.g. extract_code_blocks
                                # fails when the model generates full ```...``` fences
                                # without a gen_prefix being applied to the prompt).
                                # Try complete fence extraction first, then strip
                                # just the opening fence for truncated responses.
                                m = re.search(
                                    r"```(?:\w+)?\n?(.*?)\n?```", result, re.DOTALL
                                )
                                if m and m.group(1).strip():
                                    result = m.group(1)
                                else:
                                    # Truncated response: no closing fence.
                                    # Strip the opening fence so process_results
                                    # can attempt execution on whatever code exists.
                                    m2 = re.match(r"```(?:\w+)?\n?(.*)", result, re.DOTALL)
                                    if m2 and m2.group(1).strip():
                                        result = m2.group(1)
                        except Exception as exc:
                            logger.debug(
                                "Custom filter %r raised %s; keeping raw text.",
                                filter_spec.get("name"), exc,
                            )
            if not ok and fallback is not None:
                results.append(fallback)
            else:
                results.append(result if (ok and result) else generated)
        return results


# ---------------------------------------------------------------------------
# 3. Semantic backend
# ---------------------------------------------------------------------------

# Default system prompt for chat/instruct models.  Kept task-agnostic because
# the same backend is used both for compression (O23–O28 at apply time) and for
# pattern extraction (DatasetVocabularyFitter fitting phase).
_SEMANTIC_SYSTEM_PROMPT = (
    "Follow the given instruction carefully. "
    "Do not answer questions, solve problems, or complete tasks "
    "described in the text you are given."
)

# Completion-style template for base (non-chat) models.  The [TEXT] block
# delimiter prevents the model from treating "Answer:" or similar suffixes
# inside {text} as an invitation to answer rather than to process as instructed.
# No task-specific suffix ("Compressed:") is added here — the instruction
# already specifies the expected output format.
_SEMANTIC_COMPLETION_TEMPLATE = (
    "{instruction} "
    "Do not answer questions or solve tasks described in the text.\n\n"
    "[TEXT]\n{text}\n[/TEXT]\n\n"
)


class LMEvalSemanticBackend:
    """Use an lm-eval model as the ``backend`` for semantic transformations.

    Passed to O23–O28 transformations as their ``backend`` argument::

        from prompt_piper.transformations import ContextCompression
        backend = LMEvalSemanticBackend(model)
        op = ContextCompression(backend=backend)
        compressed = op("A long context paragraph...")

    Parameters
    ----------
    model:
        An lm-eval LM object.
    max_gen_toks:
        Maximum tokens to generate.
    stop_sequences:
        Token sequences that terminate generation.
    prompt_template:
        Override the prompt format string (must contain ``{instruction}`` and
        ``{text}`` placeholders).  ``None`` (default) auto-selects:
        a chat-message format for instruct/chat models, a completion-style
        template with explicit ``[TEXT]`` delimiters for base models.
    system_prompt:
        System message used when formatting as chat messages.  Override to
        tune the compression persona.  ``None`` uses the built-in default.
    """

    def __init__(
        self,
        model,
        max_gen_toks: int = 256,
        stop_sequences: list[str] | None = None,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        apply_chat_template: bool = False,
        cache_path: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._model = model
        self._max_gen_toks = max_gen_toks
        self._stop_sequences = stop_sequences or ["\n\n"]
        self._template = prompt_template  # None → auto-select in __call__
        self._system_prompt = system_prompt or _SEMANTIC_SYSTEM_PROMPT
        self._use_chat = apply_chat_template
        # In-memory cache: identical (text, instruction) pairs — common when the
        # shared task description is the same across many dataset samples — are
        # compressed only once and the result reused for all subsequent calls.
        self._cache: dict[tuple[str, str], str] = {}
        #: Optional :class:`~prompt_piper.optimization.recorder.RunRecorder`;
        #: when set, every call (including cache hits) is logged.
        self.recorder = None
        # Persistent cache (JSONL, append-only), shared by search, test,
        # baselines, fitting and all runs using the same file.  Keyed by the
        # model id, the exact prompt sent and the generation settings, so a
        # change to any of them never reuses a stale answer.
        candidates = [model_id, getattr(model, "pretrained", None), getattr(model, "model", None)]
        self._model_id = next((c for c in candidates if isinstance(c, str) and c), type(model).__name__)
        self._cache_path = Path(cache_path) if cache_path else None
        self._disk: dict[str, str] = {}
        if self._cache_path is not None and self._cache_path.exists():
            with open(self._cache_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        self._disk[entry["key"]] = entry["output"]

    def __call__(self, text: str, instruction: str) -> str:
        key = (text, instruction)
        if key in self._cache:
            self._log(text, instruction, self._cache[key], source="memory")
            return self._cache[key]

        actual_prompt, gen_kwargs = self._build_request(text, instruction)
        disk_key = hashlib.sha1(json.dumps(
            [self._model_id, str(actual_prompt), gen_kwargs], sort_keys=True,
        ).encode("utf-8")).hexdigest()
        if disk_key in self._disk:
            compressed = self._disk[disk_key]
            self._cache[key] = compressed
            self._log(text, instruction, compressed, source="disk", disk_key=disk_key)
            return compressed

        t0 = time.perf_counter()
        instance = _make_instance("generate_until", {}, (actual_prompt, gen_kwargs), 0)
        result = _call_model(self._model, "generate_until", [instance])
        compressed = result[0].strip()
        usage = getattr(result[0], "usage", None)  # provider usage, when returned
        elapsed = time.perf_counter() - t0
        self._cache[key] = compressed
        self._disk[disk_key] = compressed
        if self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "key": disk_key, "model_id": self._model_id,
                    "prompt": str(actual_prompt), "gen_kwargs": gen_kwargs,
                    "instruction": instruction, "text": text,
                    "output": compressed, "raw_output": str(result[0]),
                    "provider_usage": usage, "elapsed_seconds": elapsed,
                }, ensure_ascii=False) + "\n")
                f.flush()
        self._log(text, instruction, compressed, source="llm", disk_key=disk_key,
                  elapsed=elapsed, prompt=actual_prompt, raw=str(result[0]), usage=usage)
        return compressed

    def _log(self, text, instruction, output, source, disk_key=None, elapsed=None,
             prompt=None, raw=None, usage=None) -> None:
        if self.recorder is None:
            return
        record = {
            **self.recorder.context,
            "kind": "semantic_backend",
            "source": source,              # "llm" | "disk" | "memory"
            "cached": source != "llm",
            "cache_key": disk_key,
            "text_sha": self.recorder.store_text(text),
            "instruction": instruction,
            "output": output,
        }
        if source == "llm":
            record.update(
                id=self.recorder.next_id("llm_calls"),
                prompt=str(prompt), raw_output=raw, elapsed_seconds=elapsed,
                provider_usage=usage,
            )
        self.recorder.write("llm_calls", record)

    def _build_request(self, text: str, instruction: str):
        """Return ``(prompt_to_send, gen_kwargs)`` for one compression request."""
        chat = _is_chat_model(self._model) or self._use_chat or _hf_model_has_chat_template(self._model)
        # The rewrite of a multi-paragraph section can itself span paragraphs:
        # chat models stop at end-of-turn, so no "\n\n" stop (it would keep only
        # the first paragraph, which the length guard would then accept).  The
        # token cap scales with the input; API chat models may spend part of
        # the budget on hidden reasoning tokens, hence a higher floor.
        floor = 4096 if _is_chat_model(self._model) else self._max_gen_toks
        gen_kwargs = {
            "until": [] if chat else self._stop_sequences,
            "max_gen_toks": max(floor, len(text) // 2 + 256),
        }
        if _is_chat_model(self._model):
            user_content = f"{instruction}\n\n[TEXT]\n{text}\n[/TEXT]"
            actual_prompt = _wrap_chat_prompt(user_content, system_prompt=self._system_prompt)
        elif self._use_chat or _hf_model_has_chat_template(self._model):
            actual_prompt = self._format_chat(instruction, text)
        else:
            template = self._template or _SEMANTIC_COMPLETION_TEMPLATE
            actual_prompt = template.format(instruction=instruction, text=text)
        return actual_prompt, gen_kwargs

    def _format_chat(self, instruction: str, text: str) -> str:
        """Build a chat-formatted prompt using the model's own chat template.

        The system message instructs the model to compress, not answer.
        The user message contains only the instruction and the text to
        compress — no "Compressed:" suffix that could look like an answer
        prompt to the model.
        """
        if self._template:
            user_content = self._template.format(instruction=instruction, text=text)
        else:
            user_content = f"{instruction}\n\n[TEXT]\n{text}\n[/TEXT]"

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": user_content},
        ]

        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                try:
                    # enable_thinking=False suppresses thinking-mode preamble on
                    # Qwen3 and other reasoning models that support this kwarg.
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            except Exception as exc:
                logger.warning(
                    "LMEvalSemanticBackend: chat template failed (%s); "
                    "falling back to completion format.", exc
                )

        # Fallback: prepend system prompt as plain text.
        return f"{self._system_prompt}\n\n{user_content}"


# ---------------------------------------------------------------------------
# 4. Facade
# ---------------------------------------------------------------------------

class LMEvalBackend:
    """Facade that bundles model + task into dataset, evaluator, and backend.

    Parameters
    ----------
    model:
        An lm-eval LM object (see :func:`build` for convenient construction).
    task:
        An initialised lm-eval task object.

    Example
    -------
    >>> backend = LMEvalBackend.build(
    ...     model_type="hf",
    ...     model_name="mistralai/Mistral-7B-v0.1",
    ...     task_name="hellaswag",
    ... )
    >>> config = ExperimentConfig(n_gen=50, pop_size=30, n_subset=200)
    >>> result = backend.build_experiment(config).run(
    ...     baselines=["no_compression", "random_search"]
    ... )
    >>> print(result.summary())
    """

    def __init__(self, model, task, metric_key: str | None = None) -> None:
        self.model = model
        self.task = task
        self.tokenizer = LMEvalTokenizer(model)
        self.semantic_backend = LMEvalSemanticBackend(model)
        self.evaluator = LMEvalEvaluator(model, task, metric_key=metric_key)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model_type: str,
        model_name: str,
        task_name: str,
        num_fewshot: int = 0,
        metric_key: str | None = None,
        **model_kwargs,
    ) -> "LMEvalBackend":
        """Load model and task by name and return a ready :class:`LMEvalBackend`.

        Parameters
        ----------
        model_type:
            One of ``"hf"``/``"huggingface"``, ``"openai"``, ``"vllm"``.
        model_name:
            Model identifier (HuggingFace hub name, OpenAI model string, …).
        task_name:
            lm-eval task name, e.g. ``"hellaswag"``, ``"mmlu"``, ``"gsm8k"``.
        num_fewshot:
            Number of few-shot examples.
        metric_key:
            Forwarded to :class:`LMEvalEvaluator`.  Pass e.g.
            ``"exact_match,flexible-extract"`` to pin the optimisation to a
            specific lm-eval metric for generation tasks.  ``None`` uses the
            first metric in ``process_results`` output.
        **model_kwargs:
            Extra arguments forwarded to the model constructor (e.g.
            ``device="cuda"``, ``dtype="float16"``).
        """
        model = _load_model(model_type, model_name, **model_kwargs)
        task = _load_task(task_name, num_fewshot=num_fewshot)
        return cls(model, task, metric_key=metric_key)

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def get_dataset(
        self,
        split: str = "test",
        num_fewshot: int = 0,
        limit: int | None = None,
        prompt_field: str = "prompt",
        template_field: str | None = None,
    ) -> LMEvalDataset:
        """Return a :class:`LMEvalDataset` for the given split."""
        return LMEvalDataset(
            self.task,
            split=split,
            num_fewshot=num_fewshot,
            limit=limit,
            prompt_field=prompt_field,
            template_field=template_field,
        )

    # ------------------------------------------------------------------
    # Experiment builder
    # ------------------------------------------------------------------

    def build_experiment(
        self,
        config,
        split: str = "test",
        num_fewshot: int = 0,
        transformations: list[type[Transformation]] | None = None,
        use_llm_for_semantic: bool = True,
        tokenizer: Tokenizer | None = None,
        transformation_factory=None,
    ):
        """Return a fully configured :class:`~prompt_piper.optimization.runner.Experiment`.

        Parameters
        ----------
        config:
            :class:`~prompt_piper.optimization.config.ExperimentConfig`.
        split:
            Dataset split to use for the optimisation search.
        num_fewshot:
            Few-shot count for prompt formatting.
        transformations:
            Transformation classes to include.  ``None`` = all 28.
        use_llm_for_semantic:
            If ``True`` (default), semantic transformations (O23–O28) are
            instantiated with the LLM backend.  If ``False``, the heuristic
            fallbacks are used.  Ignored when ``transformation_factory`` is
            provided.
        tokenizer:
            :class:`~prompt_piper.optimization.tokenizer.Tokenizer` used to
            count tokens in the compressed prompt.  Defaults to
            :class:`LMEvalTokenizer` which wraps the loaded model's own
            tokenizer (HuggingFace ``PreTrainedTokenizer`` for ``hf``/``vllm``
            models; ``tiktoken`` for OpenAI models).  Pass an explicit
            tokenizer to override, or ``False`` to fall back to the
            :class:`~prompt_piper.optimization.tokenizer.WhitespaceTokenizer`.
        transformation_factory:
            Optional callable ``(cls: type[Transformation]) -> Transformation``
            that overrides the internal factory.  Use this to inject a
            :class:`~prompt_piper.transformations.fitting.DatasetVocabularyFitter`
            factory (combined with LLM backend injection for semantic operators
            as needed).  When ``None`` (default), the internal factory derived
            from ``use_llm_for_semantic`` is used.

        Notes
        -----
        When ``config.compress_field`` differs from ``config.prompt_field``,
        the dataset is built with ``template_field=config.compress_field`` so
        each sample contains the 0-shot prompt under that key.
        """
        from prompt_piper.optimization.runner import Experiment

        template_field = (
            config.compress_field
            if config.compress_field and config.compress_field != config.prompt_field
            else None
        )
        dataset = self.get_dataset(
            split=split,
            num_fewshot=num_fewshot,
            limit=config.n_subset,
            prompt_field=config.prompt_field,
            template_field=template_field,
        ).as_list()

        factory = (
            transformation_factory
            if transformation_factory is not None
            else self._make_transformation_factory(use_llm_for_semantic)
        )

        # Resolve tokenizer: use model-native by default, allow explicit override.
        if tokenizer is False:
            resolved_tokenizer = None   # Experiment will use WhitespaceTokenizer
        elif tokenizer is None:
            resolved_tokenizer = self.tokenizer
        else:
            resolved_tokenizer = tokenizer

        return Experiment(
            config=config,
            dataset=dataset,
            evaluator=self.evaluator,
            tokenizer=resolved_tokenizer,
            transformations=transformations,
            transformation_factory=factory,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_transformation_factory(
        self, use_llm_for_semantic: bool
    ) -> Callable[[type[Transformation]], Transformation]:
        """Return a factory that injects the LLM backend into semantic ops."""
        import inspect
        backend = self.semantic_backend if use_llm_for_semantic else None

        def factory(cls: type[Transformation]) -> Transformation:
            if backend is not None and cls.category == "semantic":
                if "backend" in inspect.signature(cls.__init__).parameters:
                    return cls(backend=backend)
            return cls()

        return factory