"""End-to-end tests for the complete Novel Writer pipeline."""

import json
import pytest
from pathlib import Path

from tests.e2e.conftest import SAMPLE_NOVEL_TEXT, SAMPLE_ENGLISH_TEXT
from novel_writer.processing.clean import clean_data
from novel_writer.processing.format import format_data
from novel_writer.processing.segment import ChapterSegmenter, segment_directory
from novel_writer.evaluation.metrics import evaluate_text, EvaluationResult
from novel_writer.processing.preference import generate_preference_pairs
from novel_writer.processing.curriculum import sort_by_curriculum, compute_difficulty, DifficultyScore


# A long novel excerpt that comfortably exceeds all validation thresholds
LONG_NOVEL_TEXT = """\
Chapter 1: The Awakening

The morning sun crept slowly over the mountain ridge, painting the valley below in hues \
of gold and amber. In the village of Thornfield, nestled between two ancient oaks, the \
first signs of life stirred. A rooster crowed from the weathered barn beside the mill, \
and soon the rhythmic sound of the waterwheel turning joined the dawn chorus.

Margaret stepped outside her cottage and breathed deeply. The air carried the sharp scent \
of pine from the northern forests, mixed with the earthy sweetness of freshly turned soil \
from the fields beyond the river. She had lived in Thornfield all her life, and yet each \
morning brought something new to marvel at. Today, however, was different. Today the \
messenger would arrive.

"Are you ready?" called her brother Edward from the doorway, his face still creased with \
sleep. He was a tall man, broad-shouldered and weathered by years of working the land. \
His hands, rough and calloused, held a steaming cup of tea that sent wisps of vapor into \
the cool morning air.

"As ready as one can be," Margaret replied, not turning from the view. "The letter said \
noon, but I suspect they will come earlier. They always do."

Edward grunted in response and joined her at the fence. Together they watched the mist \
lift from the valley floor, revealing the patchwork of fields and hedgerows that had \
sustained their family for generations. It was a peaceful scene, one that belied the \
turbulence that was about to enter their lives.

Chapter 2: The Messenger

The messenger arrived not at noon but at half past nine, just as Margaret had predicted. \
He was a young man on a tired horse, his cloak dusty from the road. He carried a leather \
satchel embossed with the royal seal, and his expression was one of practiced neutrality.

"Margaret Ashworth?" he asked, dismounting with the ease of long practice.

"I am she."

He reached into the satchel and produced a sealed envelope. The wax was crimson, pressed \
with the lion and crown that Margaret recognized from her father's old correspondence. \
She took it with steady hands, though her heart hammered beneath her ribs.

Edward appeared beside her, his presence solid and reassuring. "What does it say?" he \
asked, unable to contain his curiosity.

Margaret broke the seal and unfolded the heavy parchment within. The handwriting was \
elegant, each letter formed with the precision of a trained scribe. As she read, her \
eyes widened and the color drained from her face.

"We are summoned to the capital," she said quietly. "Both of us. By order of the Crown."

The weight of those words settled over them like a fog. The capital was three days' ride \
to the south, a place neither of them had ever visited. In Thornfield, the capital was \
spoken of in whispers, a distant world of politics and intrigue that had nothing to do \
with their quiet farming life. Until now.

Chapter 3: Preparations

The next two days were a whirlwind of activity. Margaret organized provisions while Edward \
made arrangements for the farm in their absence. Old Thomas, their neighbor, agreed to tend \
the animals and keep watch over the property. His wife, a kind woman named Helen, promised \
to look after the garden.

Margaret packed carefully, selecting clothes that were practical but presentable. She had \
no idea what to expect at the capital, but she knew that appearances mattered in such \
places. Her mother's silver brooch, the one family heirloom they still possessed, was \
pinned carefully to her traveling cloak.

"Do you think we will be gone long?" Edward asked as they loaded the cart on the morning \
of their departure. His tone was casual, but Margaret could hear the undercurrent of anxiety.

"I cannot say," she answered honestly. "But we must be prepared for anything."

The village turned out to see them off. Faces she had known her entire life lined the \
road, offering words of encouragement and small gifts for the journey. Mrs. Patterson \
pressed a loaf of fresh bread into Margaret's hands. Young Jamie ran alongside the cart \
for a quarter mile before finally waving goodbye with a gap-toothed grin.

As Thornfield disappeared behind the first hill, Margaret felt a chapter of her life \
closing. Whatever lay ahead, she knew nothing would ever be quite the same again.
"""


class TestFullPipeline:
    """Test the complete end-to-end pipeline from raw text to quality-sorted output."""

    @pytest.mark.slow
    def test_complete_pipeline_flow(self, sample_config):
        """Full pipeline: create files -> clean -> format -> evaluate -> preference -> curriculum."""
        input_dir = sample_config.data.input_dir

        # --- Stage 1: Create input files ---
        novel_file = input_dir / "full_novel.txt"
        novel_file.write_text(LONG_NOVEL_TEXT, encoding="utf-8")

        assert novel_file.exists()
        assert len(LONG_NOVEL_TEXT) > 500

        # --- Stage 2: Clean ---
        cleaned_files = clean_data(sample_config)
        assert len(cleaned_files) > 0, "Cleaning should produce at least one file"

        for cf in cleaned_files:
            assert cf.exists()
            content = cf.read_text(encoding="utf-8")
            assert len(content) >= 500

        # --- Stage 3: Format into JSONL ---
        num_entries = format_data(sample_config)
        assert num_entries > 0, "Formatting should produce JSONL entries"

        train_jsonl = sample_config.data.output_dir / "train.jsonl"
        assert train_jsonl.exists()

        entries = []
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        assert len(entries) == num_entries
        for entry in entries:
            assert "instruction" in entry
            assert "output" in entry
            assert len(entry["output"]) >= 100

        # --- Stage 4: Evaluate quality ---
        for entry in entries:
            result = evaluate_text(entry["output"])
            assert isinstance(result, EvaluationResult)
            assert 0.0 <= result.overall_score <= 1.0
            assert 0.0 <= result.repetition_rate <= 1.0
            assert 0.0 <= result.vocabulary_diversity <= 1.0

        # --- Stage 5: Generate preference pairs ---
        pref_output = sample_config.data.output_dir / "preferences.jsonl"
        num_pairs = generate_preference_pairs(
            input_path=train_jsonl,
            output_path=pref_output,
            min_score_diff=0.001,
            seed=42,
        )

        assert pref_output.exists()
        if num_pairs > 0:
            with open(pref_output, "r", encoding="utf-8") as f:
                pairs = [json.loads(line) for line in f if line.strip()]
            assert len(pairs) == num_pairs
            for pair in pairs:
                assert pair["chosen_score"] >= pair["rejected_score"]

        # --- Stage 6: Curriculum sort ---
        curriculum_output = sample_config.data.output_dir / "curriculum.jsonl"
        stats = sort_by_curriculum(
            input_path=train_jsonl,
            output_path=curriculum_output,
            num_buckets=3,
        )

        assert curriculum_output.exists()
        assert stats["total"] == num_entries

        with open(curriculum_output, "r", encoding="utf-8") as f:
            sorted_entries = [json.loads(line) for line in f if line.strip()]

        difficulties = [e["_difficulty"] for e in sorted_entries]
        assert difficulties == sorted(difficulties)

    def test_pipeline_with_chinese_text(self, sample_config):
        """Test the pipeline with Chinese text input."""
        input_dir = sample_config.data.input_dir

        # Use the Chinese sample text, repeated to exceed thresholds
        long_chinese = SAMPLE_NOVEL_TEXT * 3
        txt_file = input_dir / "chinese_novel.txt"
        txt_file.write_text(long_chinese, encoding="utf-8")

        # Clean
        cleaned_files = clean_data(sample_config)
        assert len(cleaned_files) > 0

        # Format
        num_entries = format_data(sample_config)
        assert num_entries > 0

        # Verify output
        train_jsonl = sample_config.data.output_dir / "train.jsonl"
        with open(train_jsonl, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Chinese text should be preserved in output
        all_output = " ".join(e["output"] for e in entries)
        assert any(
            char >= "\u4e00" and char <= "\u9fff" for char in all_output
        ), "Chinese characters should be present in the output"

    def test_pipeline_with_mixed_inputs(self, sample_config, sample_txt_files):
        """Test pipeline with both Chinese and English input files."""
        # sample_txt_files creates both Chinese and English files

        # Clean
        cleaned_files = clean_data(sample_config)
        assert len(cleaned_files) >= 1, "At least one file should survive cleaning"

        # Format
        num_entries = format_data(sample_config)

        if num_entries > 0:
            train_jsonl = sample_config.data.output_dir / "train.jsonl"
            with open(train_jsonl, "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]

            assert len(entries) == num_entries

    def test_segmentation_in_pipeline(self, sample_config):
        """Test that segmentation integrates with the pipeline."""
        input_dir = sample_config.data.input_dir

        # Write the long novel text which has chapter markers
        txt_file = input_dir / "chapters_novel.txt"
        txt_file.write_text(LONG_NOVEL_TEXT, encoding="utf-8")

        # Segment by chapters
        segmenter = ChapterSegmenter(min_chapter_length=100)
        chapters = segmenter.segment_text(LONG_NOVEL_TEXT)

        # Should detect multiple chapters
        assert len(chapters) >= 3, f"Expected at least 3 chapters, got {len(chapters)}"

        titles = [title for title, _ in chapters]
        assert any("Chapter" in t for t in titles)

        # Each chapter should have substantial content
        for title, content in chapters:
            assert len(content) > 50, f"Chapter '{title}' too short: {len(content)} chars"

    def test_difficulty_scoring_across_pipeline(self, sample_config):
        """Verify difficulty scoring produces consistent results across pipeline stages."""
        # Compute difficulty for texts of varying complexity
        simple_text = (
            "The cat sat on the mat. The dog ran in the park. "
            "The bird flew in the sky. The fish swam in the sea."
        )
        complex_text = (
            "The juxtaposition of ecclesiastical architecture with contemporary "
            "urban development creates an anachronistic aesthetic that simultaneously "
            "evokes nostalgia and disorientation. Phenomenological interpretations "
            "of such spatial contradictions illuminate the epistemological tensions "
            "inherent in postmodern discourse."
        )

        simple_score = compute_difficulty(simple_text)
        complex_score = compute_difficulty(complex_text)

        assert isinstance(simple_score, DifficultyScore)
        assert isinstance(complex_score, DifficultyScore)

        # Complex text should have higher vocabulary complexity
        assert complex_score.vocabulary_complexity > simple_score.vocabulary_complexity

    @pytest.mark.slow
    def test_pipeline_produces_deterministic_output(self, sample_config):
        """Verify the pipeline produces consistent results across runs."""
        input_dir = sample_config.data.input_dir

        txt_file = input_dir / "deterministic_test.txt"
        txt_file.write_text(LONG_NOVEL_TEXT, encoding="utf-8")

        # Run 1
        clean_data(sample_config)
        format_data(sample_config)

        train_jsonl = sample_config.data.output_dir / "train.jsonl"
        with open(train_jsonl, "r", encoding="utf-8") as f:
            run1_entries = [json.loads(line) for line in f if line.strip()]

        # Clean up for run 2
        for f in sample_config.data.temp_dir.glob("*"):
            f.unlink()
        train_jsonl.unlink()

        # Run 2
        clean_data(sample_config)
        format_data(sample_config)

        with open(train_jsonl, "r", encoding="utf-8") as f:
            run2_entries = [json.loads(line) for line in f if line.strip()]

        # Both runs should produce the same number of entries
        assert len(run1_entries) == len(run2_entries)

        # Content should be identical
        for e1, e2 in zip(run1_entries, run2_entries):
            assert e1["output"] == e2["output"]
            assert e1["instruction"] == e2["instruction"]
