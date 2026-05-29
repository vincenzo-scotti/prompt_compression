"""Multi-objective prompt compression optimisation.

Quick start::

    from prompt_piper.optimization import Experiment, ExperimentConfig

    config = ExperimentConfig(algorithm="nsga2", pop_size=50, n_gen=100, n_subset=200)
    dataset = [{"prompt": "...", "target": "..."}, ...]

    exp = Experiment(config, dataset)
    result = exp.run(baselines=["no_compression", "all_transformations", "random_search"])
    print(result.summary())
"""

from .config import ExperimentConfig
from .evaluator import ConstantEvaluator, Evaluator, MockEvaluator
from .tokenizer import CharTokenizer, Tokenizer, TiktokenTokenizer, WhitespaceTokenizer
from .problem import PromptCompressionProblem
from .algorithms import available_algorithms, build_algorithm
from .baselines import (
    AllTransformationsBaseline,
    BaselineResult,
    LLMLingua2Baseline,
    NoCompressionBaseline,
    RandomSearchBaseline,
)
from .runner import Experiment, ExperimentResult

__all__ = [
    "ExperimentConfig",
    "Evaluator",
    "MockEvaluator",
    "ConstantEvaluator",
    "Tokenizer",
    "WhitespaceTokenizer",
    "CharTokenizer",
    "TiktokenTokenizer",
    "PromptCompressionProblem",
    "build_algorithm",
    "available_algorithms",
    "NoCompressionBaseline",
    "AllTransformationsBaseline",
    "RandomSearchBaseline",
    "LLMLingua2Baseline",
    "BaselineResult",
    "Experiment",
    "ExperimentResult",
]