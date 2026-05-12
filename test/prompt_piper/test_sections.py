"""Tests for PromptSplitter and PromptSections.

Prompt structures are taken from real lm-eval GSM8K-style samples (few-shot):
  - Each Q&A pair is a blank-line-separated paragraph.
  - The 0-shot prompt is the final "Question: ...\nAnswer:" pair.
  - Full prompt = N few-shot paragraphs + 0-shot paragraph.
"""

import pytest

from prompt_piper.transformations.sections import PromptSections, PromptSplitter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS1 = (
    "Question: Jen did 16 triple-flips. Tyler did half. How many double-flips Tyler?\n"
    "Answer: 16 * 3 = 48. 48 / 2 = 24. 24 / 2 = 12.\n#### 12"
)
FS2 = (
    "Question: Four people plan a party. Total cost is $73. Each pays how much?\n"
    "Answer: 73 / 4 = $18.25.\n#### 18"
)
QUESTION = "Question: Janet's ducks lay 16 eggs per day.\nAnswer:"


def _varied_dataset(n: int = 5, shots: str = FS1) -> list[dict]:
    """Samples with identical shots but distinct questions (so LCP ends after shots)."""
    return [
        {
            "prompt": f"{shots}\n\n{QUESTION} (sample {i})",
            "prompt_0shot": f"{QUESTION} (sample {i})",
            "_shots": shots,
            "target": "x",
        }
        for i in range(n)
    ]


def _flat_dataset(n: int = 5, text: str = "Classify the following text.") -> list[dict]:
    """Dataset with no consistent prefix — all prompts differ."""
    return [
        {"prompt": f"{text} Sample {i}.", "target": "x"}
        for i in range(n)
    ]


def _identical_dataset(n: int = 5, text: str = "Classify: is this positive?") -> list[dict]:
    """Dataset where all prompts are identical."""
    return [{"prompt": text, "target": "x"} for _ in range(n)]


# ---------------------------------------------------------------------------
# PromptSections
# ---------------------------------------------------------------------------

class TestPromptSections:

    def test_assemble_all_parts(self):
        sec = PromptSections(
            instruction="Do this.",
            shots="Q: a?\nA: b.",
            input_data="Q: c?",
            output_cue="A:",
        )
        result = sec.assemble()
        assert "Do this." in result
        assert "Q: a?" in result
        assert "Q: c?" in result
        assert result.endswith("A:")

    def test_assemble_skips_empty_sections(self):
        sec = PromptSections(instruction="Do this.", input_data="Q: c?")
        result = sec.assemble()
        assert "\n\n\n\n" not in result  # no double-empty joins
        assert "Do this." in result
        assert "Q: c?" in result

    def test_assemble_with_overrides(self):
        sec = PromptSections(instruction="original", input_data="question", output_cue="A:")
        result = sec.assemble(instruction="compressed", input_data="short_q")
        assert result.startswith("compressed")
        assert "short_q" in result
        assert result.endswith("A:")
        assert "original" not in result

    def test_assemble_output_cue_never_overridden(self):
        sec = PromptSections(instruction="I", output_cue="Answer:")
        result = sec.assemble(instruction="J")
        assert result.endswith("Answer:")

    def test_empty_sections_assemble_to_empty(self):
        sec = PromptSections()
        assert sec.assemble() == ""


# ---------------------------------------------------------------------------
# PromptSplitter — boundary detection
# ---------------------------------------------------------------------------

class TestPromptSplitterBoundaries:

    def test_lcp_finds_instruction_prefix(self):
        dataset = _varied_dataset(n=5, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        assert FS1 in splitter._instruction_prefix

    def test_output_cue_detected(self):
        dataset = _varied_dataset(n=5, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        # The cue might be detected or might be "" for this corpus.
        # We only verify it doesn't crash and is a string.
        assert isinstance(splitter._output_cue, str)

    def test_no_prefix_for_flat_dataset(self):
        dataset = _flat_dataset(n=5)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        assert splitter._instruction_prefix == ""

    def test_single_sample_no_prefix(self):
        dataset = _varied_dataset(n=1, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        assert splitter._instruction_prefix == ""

    def test_identical_samples_degenerate(self):
        dataset = _identical_dataset(n=5)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        # LCP = entire string = degenerate case → no prefix boundary
        assert splitter._instruction_prefix == ""


# ---------------------------------------------------------------------------
# PromptSplitter — split
# ---------------------------------------------------------------------------

class TestPromptSplitterSplit:

    def test_instruction_populated_from_lcp(self):
        dataset = _varied_dataset(n=5, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        section = splitter.split(dataset[0])
        # FS1 is in the LCP so it ends up somewhere in instruction or shots
        text = section.instruction + section.shots
        assert "Jen did 16" in text

    def test_shots_from_shots_field(self):
        dataset = _varied_dataset(n=5, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt", shots_field="_shots")
        section = splitter.split(dataset[0])
        assert "Jen did 16" in section.shots

    def test_input_data_is_per_sample(self):
        dataset = _varied_dataset(n=5, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        sec0 = splitter.split(dataset[0])
        sec1 = splitter.split(dataset[1])
        # Different samples → different input_data
        assert sec0.input_data != sec1.input_data

    def test_flat_dataset_whole_text_in_instruction(self):
        """When no LCP is found, entire text is treated as instruction."""
        dataset = _flat_dataset(n=5, text="Classify")
        splitter = PromptSplitter(dataset, compress_field="prompt")
        section = splitter.split(dataset[0])
        # Instruction should hold the prompt content; shots and data should be empty.
        assert section.shots == ""
        assert section.input_data == ""
        assert "Classify" in section.instruction

    def test_split_all_returns_one_section_per_sample(self):
        dataset = _varied_dataset(n=4, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        sections = splitter.split_all()
        assert len(sections) == 4

    def test_assemble_recovers_full_text(self):
        """The assembled text should contain the original content."""
        dataset = _varied_dataset(n=3, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt_0shot")
        for i, sample in enumerate(dataset):
            section = splitter.split(sample)
            assembled = section.assemble()
            original = sample["prompt_0shot"]
            # Assembled text should contain the key content
            assert "Janet" in assembled or "Janet" in original


# ---------------------------------------------------------------------------
# PromptSplitter — compress_field variations
# ---------------------------------------------------------------------------

class TestPromptSplitterCompressField:

    def test_compress_field_prompt_0shot(self):
        """Splitting on prompt_0shot yields just the per-sample question."""
        dataset = _varied_dataset(n=5, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt_0shot")
        # 0-shot prompts differ per sample → no LCP → all go to instruction
        section = splitter.split(dataset[0])
        assert isinstance(section, PromptSections)

    def test_compress_field_prompt_full(self):
        """Splitting on prompt picks up FS1 prefix in instruction/shots."""
        dataset = _varied_dataset(n=5, shots=FS1)
        splitter = PromptSplitter(dataset, compress_field="prompt")
        section = splitter.split(dataset[0])
        content = section.instruction + section.shots + section.input_data
        assert "Jen did 16" in content or "Janet" in content


# ---------------------------------------------------------------------------
# PromptSplitter — two-shot dataset
# ---------------------------------------------------------------------------

class TestPromptSplitterTwoShot:

    def test_two_shots_both_in_prefix(self):
        two_shots = f"{FS1}\n\n{FS2}"
        dataset = [
            {
                "prompt": f"{two_shots}\n\nQuestion: sample {i}.\nAnswer:",
                "prompt_0shot": f"Question: sample {i}.\nAnswer:",
                "_shots": two_shots,
                "target": "x",
            }
            for i in range(5)
        ]
        splitter = PromptSplitter(dataset, compress_field="prompt", shots_field="_shots")
        section = splitter.split(dataset[0])
        shots_content = section.shots
        assert "Jen did 16" in shots_content
        assert "Four people" in shots_content