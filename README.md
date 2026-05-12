# Prompt Piper

Prompt compression for token-efficient Large Language Models usage.

Prompt Piper searches for the best sequence of text transformations to shorten a prompt while preserving task performance. The search is framed as a multi-objective optimisation problem (minimise token count, maximise task accuracy) solved via evolutionary algorithms (NSGA-II, NSGA-III, MOEA/D).

## Quick start

```bash
# Install dependencies
pip install -e .

# Dry-run (no GPU or API key needed)
python -m prompt_piper.tools.compressor \
    --config resources/configs/example.yaml \
    --dry-run
```

Copy `.env.example` to `.env` and fill in API keys before running real experiments.

## Transformations

28 rule-based operators (O1–O28) in four categories:

| Category | IDs | Examples |
|---|---|---|
| Lexical | O1–O8 | whitespace normalisation, stopword removal, synonym shortening, abbreviation substitution |
| Syntactic | O9–O13 | sentence merging, imperative rewrite, enumeration compression, clause deletion |
| Structural | O14–O22 | deduplication, header removal, section/context/example pruning |
| Semantic | O23–O28 | context compression, instruction summarisation, example distillation; some require an LLM backend |

## Prompt fields and few-shot compression

Each dataset sample is a dict. The two most important fields are:

| Field | Content |
|---|---|
| `"prompt"` | The fully formatted prompt sent to the model — may include few-shot demonstration examples prepended by lm-eval |
| `"prompt_0shot"` | The same test item formatted without any few-shot examples (instruction + question only) |

`"prompt_0shot"` is **not** a format string with placeholders — it is already fully rendered, just without the demonstration block. The distinction matters when the dataset is loaded with `num_fewshot > 0`.

### Two compression modes

**Compress the full prompt** (default):

```
compress_field = "prompt"   (or omit compress_field)

sample["prompt"]  ──► [transformations] ──► compressed
                                                 │
                                            evaluator + tokenizer
```

Transformations see everything, including any few-shot demonstrations. Operators such as `ExampleRemoval` (O19) or `ExampleSubsetSelection` (O20) can reduce the number of shots.

**Compress only the 0-shot instruction** (protect demonstrations):

```
compress_field = "prompt_0shot"

sample["prompt_0shot"]  ──► [transformations] ──► compressed
                                                        │
                                                   evaluator + tokenizer
```

Transformations only see the instruction and test item. The few-shot demonstrations are never modified. The model is evaluated with the compressed instruction in 0-shot mode.

### Choosing a mode

| Goal | `num_fewshot` | `compress_field` |
|---|---|---|
| Compress a 0-shot prompt | 0 | `null` / omit |
| Compress the instruction only, leave shots untouched | ≥ 1 | `"prompt_0shot"` |
| Compress the entire few-shot prompt as a single block | ≥ 1 | `null` / omit |

### Operator availability by compression mode

Several structural operators are only meaningful when transformations can see the few-shot demonstrations:

| Operator | `compress_field="prompt"` | `compress_field="prompt_0shot"` |
|---|---|---|
| O19 ExampleRemoval | ✅ removes last shot | no-op (no example blocks in 0-shot text) |
| O20 ExampleSubsetSelection | ✅ keeps first *k* shots | no-op |
| O21 ExampleSimplification | ✅ compacts `Input:/Output:` blocks | no-op |
| O27 ExampleDistillation | ✅ distils shots to a rule | no-op |
| O17 ContextPruning | ✅ few-shot paragraphs protected | ✅ operates on instruction only |
| O23–O26, O28 | ✅ | ✅ |

When using `compress_field="prompt_0shot"`, the effective search space is reduced from 28 to ~24 operators.

### Model compatibility

The `"prompt"` field's format depends on how the dataset is loaded. The `"prompt_0shot"` field is **always plain text** regardless of model type (the chat template is intentionally not applied to it, so transformations always operate on a readable string).

| Model type | `compress_field="prompt"` | `compress_field="prompt_0shot"` |
|---|---|---|
| Base HF (GPT-2, Mistral-7B, …) | ✅ plain text, all operators work | ✅ |
| HF instruct/chat (Llama-3-Instruct, Qwen-chat, …) | ⚠️ chat-template markup in `prompt`; structural operators may mangle special tokens | ✅ recommended |
| ChatGPT / OpenAI chat API | ❌ `prompt` is JSON-encoded when `apply_chat_template=True`; transformations cannot parse it sensibly | ✅ required |

**Rule of thumb**: use `compress_field="prompt_0shot"` for any instruct or chat model. Use `compress_field="prompt"` (or omit it) for base/completion models.

### Code example

```python
from prompt_piper.optimization import Experiment, ExperimentConfig
from prompt_piper.backends.lm_eval import LMEvalDataset

# Mode A — compress everything (pure 0-shot task)
config = ExperimentConfig(prompt_field="prompt", n_subset=200)
dataset = LMEvalDataset.from_task("hellaswag", num_fewshot=0).as_list()

# Mode B — protect few-shot demonstrations, compress instruction only
config = ExperimentConfig(
    prompt_field="prompt",
    compress_field="prompt_0shot",
    n_subset=200,
)
dataset = LMEvalDataset.from_task(
    "hellaswag",
    num_fewshot=3,
    template_field="prompt_0shot",   # populates the field in each sample
).as_list()

from prompt_piper.transformations import ALL_TRANSFORMATIONS
exp = Experiment(config, dataset, transformations=ALL_TRANSFORMATIONS)
result = exp.run(baselines=["no_compression", "all_transformations"])
print(result.summary())
```

### YAML config equivalent

```yaml
runs:
  # Compress full prompt (0-shot task)
  - name: hellaswag_full
    task: hellaswag
    num_fewshot: 0

  # Compress instruction only (3-shot task, demonstrations protected)
  - name: hellaswag_0shot_compression
    task: hellaswag
    num_fewshot: 3
    compress_field: prompt_0shot
```

## Dataset-adaptive fitting

`DatasetVocabularyFitter` analyses a corpus of prompts before optimisation and returns transformation instances pre-configured for the dataset's vocabulary and structure:

```python
from prompt_piper.transformations.fitting import DatasetVocabularyFitter

fitter = DatasetVocabularyFitter(train_dataset, seed=42)

# Fit all eligible operators at once
fitted = fitter.fit_all()          # returns {operator_id: instance}

# Or fit a specific subset
fitted = fitter.fit_all(operator_ids=["O3", "O21", "O22"])

# Inspect what was discovered
print(fitter.report())

# Build a factory for use with Experiment
factory = fitter.make_transformation_factory()
result  = Experiment(config, eval_dataset,
                     transformation_factory=factory).run()
```

Corpus-only fitters (no LLM needed): O3, O5, O17, O18, O21, O22.
LLM-backed fitters (pass `backend=` to the constructor): O4, O10, O15, O28.

### Boundary detection

When compressing full prompts that include few-shot demonstrations, the fitter needs to know where the fixed few-shot block ends and the per-sample question begins, so that operators like `ContextPruning` (O17) do not prune the demonstration paragraphs.

`DatasetVocabularyFitter.fit_boundary()` detects this boundary automatically via the **longest common prefix** (LCP) across all corpus texts:

```python
fitter = DatasetVocabularyFitter(
    train_dataset,
    compress_field="prompt",      # full-prompt compression
    template_field="prompt_0shot",
)
prefix_text, n_paragraphs = fitter.fit_boundary()
# prefix_text  — the shared instruction + few-shot block
# n_paragraphs — number of blank-line-separated paragraphs to protect
```

The LCP approach uses every sample in the corpus (unlike the older fallback, which only inspected the first sample) and does not require `template_field` to be set — it derives the boundary purely from observed text divergence. It works correctly for base HF models and for `compress_field="prompt_0shot"` with any model type.

`fit_boundary()` is called automatically inside `fit_context_pruning()` and `make_transformation_factory()`, so in most cases you do not need to call it directly. The `template_field` fallback (substring search on the first sample) is retained for corpora where all samples happen to be identical, which makes the LCP degenerate.

## CLI

```bash
# YAML config (recommended for multi-run experiments)
python -m prompt_piper.tools.compressor --config resources/configs/example.yaml

# Dry-run
python -m prompt_piper.tools.compressor --config resources/configs/example.yaml --dry-run

# Single-run
python -m prompt_piper.tools.compressor \
    --model-type hf --model-name mistralai/Mistral-7B-v0.1 \
    --task hellaswag \
    --algorithm nsga2 --pop-size 50 --n-gen 100 \
    --output-dir ./results
```

Results are written as self-contained JSON files under `--output-dir`. Each file includes the full configuration, Pareto-optimal solutions with both search and test objectives, baseline comparisons, and (when fitting was used) the discovered vocabulary patterns.

## ICSE '26 experiments

Preliminary experiments across six benchmarks, run once with a self-hosted model and once with GPT-4o. Results land in `results/icse_26/`.

```bash
mkdir -p logs

# Qwen/Qwen3.5-9B (GPU required) — GSM8K, ARC Challenge, MMLU, IFEval, GPQA, BBH
nohup bash scripts/preliminary_run.sh > logs/preliminary_hf.log 2>&1 &

# GPT-4o (requires OPENAI_API_KEY in .env) — same tasks via generative variants
nohup bash scripts/preliminary_run_chatgpt.sh > logs/preliminary_chatgpt.log 2>&1 &
```

Both scripts load `.env` automatically, are restartable (completed runs are skipped), and accept `--dry-run` for a CPU-only smoke test.

## Notebooks

```bash
.venv/bin/jupyter lab
```

- `notebooks/example.ipynb` — basic API walkthrough with a synthetic dataset
- `notebooks/example_llm.ipynb` — full pipeline with an lm-eval task and HF model
- `notebooks/example_llm_complex.ipynb` — multi-run scenario with fitting, few-shot compression, and baselines
- `notebooks/example_chatgpt.ipynb` — same pipeline using ChatGPT via the OpenAI chat-completions API