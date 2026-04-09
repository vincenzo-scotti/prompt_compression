"""Rule-based prompt transformations (O1–O28).

All transformation classes are exported from this package so callers can do::

    from prompt_piper.transformations.rule_based import RemoveExtraWhitespace, SentenceMerge
"""

from prompt_piper.transformations.base import Transformation
from prompt_piper.transformations.fitting import DatasetVocabularyFitter

# Lexical (O1–O8)
from prompt_piper.transformations.lexical import (
    RemoveExtraWhitespace,
    TrimSpaces,
    RemoveStopwords,
    ReplaceWithShortSynonym,
    AbbreviationSubstitution,
    SymbolSubstitution,
    ContractionSubstitution,
    RemovePolitenessMarkers,
)

# Syntactic (O9–O13)
from prompt_piper.transformations.syntactic import (
    SentenceMerge,
    ImperativeRewrite,
    EnumerationCompression,
    ClauseDeletion,
    SentenceSimplification,
)

# Structural (O14–O22)
from prompt_piper.transformations.structural import (
    DeduplicateInstructions,
    RemovePromptHeader,
    RemoveMetaText,
    ContextPruning,
    SectionPruning,
    ExampleRemoval,
    ExampleSubsetSelection,
    ExampleSimplification,
    LabelShortening,
)

# Semantic (O23–O28)
from prompt_piper.transformations.semantic import (
    ContextCompression,
    InstructionSummarization,
    InstructionAbstraction,
    SchemaEncoding,
    ExampleDistillation,
    ChunkCompression,
)

__all__ = [
    # Base
    "Transformation",
    # Fitting
    "DatasetVocabularyFitter",
    # Lexical
    "RemoveExtraWhitespace",   # O1
    "TrimSpaces",              # O2
    "RemoveStopwords",         # O3
    "ReplaceWithShortSynonym", # O4
    "AbbreviationSubstitution",# O5
    "SymbolSubstitution",      # O6
    "ContractionSubstitution", # O7
    "RemovePolitenessMarkers", # O8
    # Syntactic
    "SentenceMerge",           # O9
    "ImperativeRewrite",       # O10
    "EnumerationCompression",  # O11
    "ClauseDeletion",          # O12
    "SentenceSimplification",  # O13
    # Structural
    "DeduplicateInstructions", # O14
    "RemovePromptHeader",      # O15
    "RemoveMetaText",          # O16
    "ContextPruning",          # O17
    "SectionPruning",          # O18
    "ExampleRemoval",          # O19
    "ExampleSubsetSelection",  # O20
    "ExampleSimplification",   # O21
    "LabelShortening",         # O22
    # Semantic
    "ContextCompression",      # O23
    "InstructionSummarization",# O24
    "InstructionAbstraction",  # O25
    "SchemaEncoding",          # O26
    "ExampleDistillation",     # O27
    "ChunkCompression",        # O28
]

# Ordered list of all transformations matching paper table order.
ALL_TRANSFORMATIONS: list[type[Transformation]] = [
    RemoveExtraWhitespace,
    TrimSpaces,
    RemoveStopwords,
    ReplaceWithShortSynonym,
    AbbreviationSubstitution,
    SymbolSubstitution,
    ContractionSubstitution,
    RemovePolitenessMarkers,
    SentenceMerge,
    ImperativeRewrite,
    EnumerationCompression,
    ClauseDeletion,
    SentenceSimplification,
    DeduplicateInstructions,
    RemovePromptHeader,
    RemoveMetaText,
    ContextPruning,
    SectionPruning,
    ExampleRemoval,
    ExampleSubsetSelection,
    ExampleSimplification,
    LabelShortening,
    ContextCompression,
    InstructionSummarization,
    InstructionAbstraction,
    SchemaEncoding,
    ExampleDistillation,
    ChunkCompression,
]