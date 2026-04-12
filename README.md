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

## Notebooks

```bash
.venv/bin/jupyter lab
```

- `notebooks/example.ipynb` — basic API walkthrough with a synthetic dataset
- `notebooks/example_llm.ipynb` — full pipeline with an lm-eval task and HF model