"""Smoke-test the model backend of a config with one short request.

Usage::

    python -m prompt_piper.tools.check_backend --config resources/configs/experiments/experiments_qwen35_9b.yaml

Loads the configured model exactly as the compressor does, then reports:

* (llama-swap) the model ids served, and whether the configured name is among them;
* one chat completion: the answer, the provider's token ``usage`` and any
  reasoning text returned (it must be empty when thinking is disabled);
* the token count of a test string with the counter used for the objective.

Exits with status 1 if a check fails.  Reads API keys from ``.env``.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="What is 17 + 25? Reply with the number only.")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    from prompt_piper.backends.lm_eval import (
        LMEvalTokenizer, _load_model, _make_instance, _wrap_chat_prompt,
    )
    from prompt_piper.tools.compressor.config import CompressorConfig

    cfg = CompressorConfig.from_yaml(args.config)
    print(f"model: {cfg.model.type} / {cfg.model.name}  kwargs: {json.dumps(cfg.model.kwargs)}")
    model = _load_model(cfg.model.type, cfg.model.name, **dict(cfg.model.kwargs))
    ok = True

    if cfg.model.kwargs.get("llama_swap"):
        import requests

        r = requests.get(f"{model.server_url}/v1/models", headers=model.header, timeout=30)
        print(f"GET /v1/models -> HTTP {r.status_code}")
        if r.ok:
            ids = [m.get("id") for m in r.json().get("data", [])]
            print(f"  served models: {ids}")
            if cfg.model.name not in ids:
                print(f"  FAIL: configured name {cfg.model.name!r} is not served")
                ok = False
        else:
            print(f"  FAIL: {r.text[:300]}")
            ok = False

    request = _make_instance(
        "generate_until", {}, (_wrap_chat_prompt(args.prompt), {"max_gen_toks": 64}), 0,
    )
    try:
        out = model.generate_until([request], disable_tqdm=True)[0]
    except Exception as exc:  # report and fail
        print(f"FAIL: request raised {exc!r}")
        return 1
    usage = getattr(out, "usage", None)
    reasoning = getattr(out, "reasoning", None)
    print(f"answer: {str(out)!r}")
    print(f"provider usage: {json.dumps(usage)}")
    print(f"reasoning returned: {reasoning!r}")
    details = (usage or {}).get("completion_tokens_details") or {}
    if reasoning or details.get("reasoning_tokens"):
        print("  NOTE: the model produced reasoning (expected only where thinking cannot be disabled)")
    if not str(out).strip():
        print("  FAIL: empty answer")
        ok = False

    try:
        n = LMEvalTokenizer(model).count("The quick brown fox jumps over the lazy dog.")
        print(f"token count of a 9-word sentence: {n}")
    except Exception as exc:
        print(f"FAIL: token counting raised {exc!r}")
        ok = False

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
