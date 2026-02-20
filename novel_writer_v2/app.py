"""Gradio web UI for the multi-agent story generator."""

import argparse
import json
import threading
import traceback

import gradio as gr

from .config import AppConfig, GeminiConfig, QwenConfig, GenerationConfig
from .models.story_state import StoryState
from .models.story_bible import StoryBible
from .pipeline import Pipeline
from .utils import detect_language

# ---------------------------------------------------------------------------
# Global mutable state (per-session in single-user mode)
# ---------------------------------------------------------------------------
_state = {
    "pipeline": None,
    "story_state": StoryState(),
    "bible": StoryBible(),
    "log_text": "",
    "generating": False,
}
_lock = threading.Lock()


def _log(msg: str) -> None:
    with _lock:
        _state["log_text"] += msg + "\n"


def _progress_callback(msg: str, ch: int, total: int, rnd: int) -> None:
    _log(f"[Ch {ch}/{total} R{rnd}] {msg}")


# ---------------------------------------------------------------------------
# Helper: rebuild pipeline from config inputs
# ---------------------------------------------------------------------------
def _rebuild_pipeline(
    gemini_key: str,
    gemini_model: str,
    qwen_key: str,
    qwen_model: str,
    language: str,
    max_rounds: int,
    min_chars: int,
    max_chars: int,
) -> str:
    cfg = AppConfig(
        gemini=GeminiConfig(api_key=gemini_key, model=gemini_model),
        qwen=QwenConfig(api_key=qwen_key, model=qwen_model),
        generation=GenerationConfig(
            max_debate_rounds=max_rounds,
            chapter_min_chars=min_chars,
            chapter_max_chars=max_chars,
            language=language,
        ),
    )
    _state["pipeline"] = Pipeline(cfg)
    _state["story_state"].language = language
    return "Pipeline initialized successfully."


# ---------------------------------------------------------------------------
# Tab 2: Story Workshop handlers
# ---------------------------------------------------------------------------
def _generate_outline(
    premise: str,
    num_chapters: int,
    gemini_key: str,
    gemini_model: str,
    qwen_key: str,
    qwen_model: str,
    language: str,
    max_rounds: int,
    min_chars: int,
    max_chars: int,
) -> str:
    if not gemini_key:
        return "Error: Please enter a Gemini API key in the Setup tab."
    _rebuild_pipeline(
        gemini_key, gemini_model, qwen_key, qwen_model,
        language, max_rounds, min_chars, max_chars,
    )
    pipe = _state["pipeline"]
    try:
        outline = pipe.generate_outline(premise, num_chapters, language)
        _state["story_state"].outline = outline
        _log(f"Outline generated ({len(outline)} chars)")
        return outline
    except Exception as e:
        return f"Error generating outline: {e}\n{traceback.format_exc()}"


def _generate_guidance(
    premise: str,
    style_desc: str,
    gemini_key: str,
    gemini_model: str,
    qwen_key: str,
    qwen_model: str,
    language: str,
    max_rounds: int,
    min_chars: int,
    max_chars: int,
) -> str:
    if not gemini_key:
        return "Error: Please enter a Gemini API key in the Setup tab."
    _rebuild_pipeline(
        gemini_key, gemini_model, qwen_key, qwen_model,
        language, max_rounds, min_chars, max_chars,
    )
    pipe = _state["pipeline"]
    try:
        guidance = pipe.generate_guidance(premise, style_desc, language)
        _state["story_state"].guidance = guidance
        _log(f"Guidance generated ({len(guidance)} chars)")
        return guidance
    except Exception as e:
        return f"Error generating guidance: {e}\n{traceback.format_exc()}"


def _init_bible(
    outline: str,
    gemini_key: str,
    gemini_model: str,
    qwen_key: str,
    qwen_model: str,
    language: str,
    max_rounds: int,
    min_chars: int,
    max_chars: int,
):
    if not gemini_key:
        return "Error: Gemini API key required.", "", "", ""
    _rebuild_pipeline(
        gemini_key, gemini_model, qwen_key, qwen_model,
        language, max_rounds, min_chars, max_chars,
    )
    pipe = _state["pipeline"]
    try:
        bible = pipe.initialize_bible(outline, language)
        _state["bible"] = bible
        _state["story_state"].outline = outline
        _log("Story Bible initialized from outline")
        return _bible_display(bible)
    except Exception as e:
        err = f"Error: {e}"
        return err, err, err, err


def _bible_display(bible: StoryBible):
    chars = "\n".join(
        f"**{c.name}**: {c.description} | Location: {c.location or 'N/A'} | Arc: {c.arc_stage or 'N/A'}"
        for c in bible.characters
    ) or "No characters yet."
    threads = "\n".join(
        f"**[{pt.status}] {pt.name}**: {pt.description}"
        for pt in bible.plot_threads
    ) or "No plot threads yet."
    timeline = "\n".join(bible.timeline) or "No timeline entries yet."
    notes = bible.world_notes or "No world notes yet."
    return chars, threads, timeline, notes


def _generate_single_chapter(
    chapter_num: int,
    outline: str,
    guidance: str,
    gemini_key: str,
    gemini_model: str,
    qwen_key: str,
    qwen_model: str,
    language: str,
    max_rounds: int,
    min_chars: int,
    max_chars: int,
):
    if not gemini_key or not qwen_key:
        return "Error: Both API keys required.", "", "", "", ""
    if _state["generating"]:
        return "Generation already in progress.", "", "", "", ""

    _state["generating"] = True
    try:
        _rebuild_pipeline(
            gemini_key, gemini_model, qwen_key, qwen_model,
            language, max_rounds, min_chars, max_chars,
        )
        pipe = _state["pipeline"]
        ss = _state["story_state"]
        bible = _state["bible"]

        ss.outline = outline
        if guidance:
            ss.guidance = guidance
            ss.style_rules = pipe.analyze_guidance(guidance)
            _log("Style rules extracted from guidance")

        # Ensure chapter plans exist
        if not ss.chapter_plans:
            _log("Breaking down outline into chapter plans...")
            num_chs = outline.count("Chapter") or outline.count("章") or 10
            ss.chapter_plans = pipe.breakdown_outline(outline, max(num_chs, 1))
            _log(f"Created {len(ss.chapter_plans)} chapter plans")

        # Find the chapter info
        ch_idx = chapter_num - 1
        if ch_idx < 0 or ch_idx >= len(ss.chapter_plans):
            return f"Error: Chapter {chapter_num} out of range (1-{len(ss.chapter_plans)})", "", "", "", ""

        ch_info = ss.chapter_plans[ch_idx]
        ch_info["chapter_number"] = chapter_num

        chapter, bible = pipe.generate_single_chapter(
            chapter_num, ch_info, ss, bible, _progress_callback,
        )
        ss.chapters.append(chapter)
        _state["bible"] = bible

        # Build debate summary
        debate_summary = ""
        for dr in chapter.debate_rounds:
            v = dr.verdict
            debate_summary += (
                f"**Round {dr.round_number}**: "
                f"Continuity={dr.continuity_feedback.score}/10, "
                f"Style={dr.style_feedback.score}/10, "
                f"Judge={'APPROVED' if v.approved else 'REVISE'} "
                f"(score {v.overall_score})\n"
            )
            if v.revision_instructions:
                debate_summary += f"  Revision: {v.revision_instructions[:200]}...\n"

        chars, threads, timeline, notes = _bible_display(bible)
        return chapter.text, debate_summary, chars, threads, _state["log_text"]
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}", "", "", "", _state["log_text"]
    finally:
        _state["generating"] = False


def _generate_batch(
    outline: str,
    guidance: str,
    gemini_key: str,
    gemini_model: str,
    qwen_key: str,
    qwen_model: str,
    language: str,
    max_rounds: int,
    min_chars: int,
    max_chars: int,
):
    if not gemini_key or not qwen_key:
        return "Error: Both API keys required.", "", "", "", ""
    if _state["generating"]:
        return "Generation already in progress.", "", "", "", ""

    _state["generating"] = True
    try:
        _rebuild_pipeline(
            gemini_key, gemini_model, qwen_key, qwen_model,
            language, max_rounds, min_chars, max_chars,
        )
        pipe = _state["pipeline"]
        ss = _state["story_state"]
        bible = _state["bible"]

        ss.outline = outline
        if guidance:
            ss.guidance = guidance
            ss.style_rules = pipe.analyze_guidance(guidance)

        if not ss.chapter_plans:
            num_chs = max(outline.count("Chapter"), outline.count("章"), 1)
            ss.chapter_plans = pipe.breakdown_outline(outline, num_chs)
            _log(f"Created {len(ss.chapter_plans)} chapter plans")

        ss, bible = pipe.generate_all_chapters(
            ss, bible, _progress_callback,
        )
        _state["bible"] = bible

        # Build full story text
        full_text = ""
        debate_summary = ""
        for ch in ss.chapters:
            full_text += f"\n\n{'='*60}\n# Chapter {ch.number}: {ch.title}\n{'='*60}\n\n{ch.text}"
            debate_summary += f"\n**Chapter {ch.number}** (score {ch.final_score}/10):\n"
            for dr in ch.debate_rounds:
                v = dr.verdict
                debate_summary += (
                    f"  Round {dr.round_number}: C={dr.continuity_feedback.score} "
                    f"S={dr.style_feedback.score} → {'OK' if v.approved else 'REVISE'}\n"
                )

        chars, threads, timeline, notes = _bible_display(bible)
        return full_text.strip(), debate_summary, chars, threads, _state["log_text"]
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}", "", "", "", _state["log_text"]
    finally:
        _state["generating"] = False


def _get_full_story():
    ss = _state["story_state"]
    if not ss.chapters:
        return "No chapters generated yet.", [], ""

    full_text = ""
    choices = []
    scores_text = ""
    for ch in ss.chapters:
        full_text += f"\n\n{'='*60}\n# Chapter {ch.number}: {ch.title}\n{'='*60}\n\n{ch.text}"
        choices.append(f"Chapter {ch.number}: {ch.title}")
        scores_text += f"Chapter {ch.number}: {ch.final_score}/10\n"

    return full_text.strip(), gr.update(choices=choices), scores_text


def _navigate_chapter(choice: str):
    if not choice:
        return ""
    ss = _state["story_state"]
    try:
        num = int(choice.split(":")[0].replace("Chapter ", "").strip())
    except (ValueError, IndexError):
        return ""
    for ch in ss.chapters:
        if ch.number == num:
            return ch.text
    return ""


def _export_story():
    ss = _state["story_state"]
    if not ss.chapters:
        return None
    text = ""
    for ch in ss.chapters:
        text += f"\n\nChapter {ch.number}: {ch.title}\n\n{ch.text}"
    path = "/tmp/novel_export.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip())
    return path


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------
def create_app() -> gr.Blocks:
    with gr.Blocks(title="Multi-Agent Story Generator v2") as app:
        gr.Markdown("# Multi-Agent Story Generator v2\n8 specialized AI agents collaborate to write your novel.")

        # ---- Tab 1: Setup ----
        with gr.Tab("Setup"):
            gr.Markdown("## API Configuration")
            with gr.Row():
                with gr.Column():
                    gemini_key = gr.Textbox(label="Gemini API Key", type="password")
                    gemini_model = gr.Dropdown(
                        choices=["gemini-3.1-pro-preview", "gemini-2.5-pro-preview-05-06", "gemini-2.5-flash-preview-05-20"],
                        value="gemini-3.1-pro-preview",
                        label="Gemini Model",
                    )
                with gr.Column():
                    qwen_key = gr.Textbox(label="Qwen (DashScope) API Key", type="password")
                    qwen_model = gr.Dropdown(
                        choices=["qwen3.5-plus", "qwen-plus", "qwen-max"],
                        value="qwen3.5-plus",
                        label="Qwen Model",
                    )
            with gr.Row():
                language = gr.Dropdown(
                    choices=["auto", "en", "zh"],
                    value="auto",
                    label="Language",
                )
                max_rounds = gr.Slider(
                    minimum=1, maximum=5, value=3, step=1,
                    label="Max Debate Rounds",
                )
            with gr.Row():
                min_chars = gr.Slider(
                    minimum=2000, maximum=10000, value=5000, step=500,
                    label="Min Chapter Chars",
                )
                max_chars = gr.Slider(
                    minimum=3000, maximum=15000, value=8000, step=500,
                    label="Max Chapter Chars",
                )
            setup_status = gr.Textbox(label="Status", interactive=False)
            setup_btn = gr.Button("Save Configuration")
            setup_btn.click(
                _rebuild_pipeline,
                inputs=[gemini_key, gemini_model, qwen_key, qwen_model, language, max_rounds, min_chars, max_chars],
                outputs=[setup_status],
            )

        # Shared config inputs for other tabs
        config_inputs = [gemini_key, gemini_model, qwen_key, qwen_model, language, max_rounds, min_chars, max_chars]

        # ---- Tab 2: Story Workshop ----
        with gr.Tab("Story Workshop"):
            gr.Markdown("## Story Workshop")
            with gr.Row():
                with gr.Column(scale=2):
                    premise = gr.Textbox(
                        label="Story Premise",
                        lines=4,
                        placeholder="Describe your story idea...",
                    )
                    num_chapters = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="Number of Chapters",
                    )
                with gr.Column(scale=1):
                    style_desc = gr.Textbox(
                        label="Style Description (for guidance generation)",
                        lines=3,
                        placeholder="e.g., literary fiction, dark tone, first person...",
                    )
                    guidance_file = gr.File(label="Upload Guidance File (.txt)")

            with gr.Row():
                gen_outline_btn = gr.Button("Generate Outline", variant="primary")
                gen_guidance_btn = gr.Button("Generate Guidance")

            outline_box = gr.Textbox(label="Outline", lines=15, interactive=True)
            guidance_box = gr.Textbox(label="Writing Guidance", lines=10, interactive=True)

            def _load_guidance_file(file):
                if file is None:
                    return ""
                with open(file.name, "r", encoding="utf-8") as f:
                    return f.read()

            guidance_file.change(_load_guidance_file, inputs=[guidance_file], outputs=[guidance_box])

            gen_outline_btn.click(
                _generate_outline,
                inputs=[premise, num_chapters] + config_inputs,
                outputs=[outline_box],
            )
            gen_guidance_btn.click(
                _generate_guidance,
                inputs=[premise, style_desc] + config_inputs,
                outputs=[guidance_box],
            )

            gr.Markdown("## Generate Chapters")
            with gr.Row():
                chapter_num_input = gr.Number(
                    label="Chapter Number", value=1, precision=0,
                )
                gen_single_btn = gr.Button("Generate Single Chapter", variant="primary")
                gen_batch_btn = gr.Button("Generate All Chapters", variant="secondary")

            chapter_output = gr.Textbox(label="Chapter Output", lines=20)
            debate_output = gr.Markdown(label="Debate Summary")

            # Hidden state displays that update from generation
            with gr.Row():
                gen_chars_display = gr.Markdown(label="Characters (updated)")
                gen_threads_display = gr.Markdown(label="Plot Threads (updated)")
            gen_log_display = gr.Textbox(label="Agent Log", lines=10)

            gen_single_btn.click(
                _generate_single_chapter,
                inputs=[chapter_num_input, outline_box, guidance_box] + config_inputs,
                outputs=[chapter_output, debate_output, gen_chars_display, gen_threads_display, gen_log_display],
            )
            gen_batch_btn.click(
                _generate_batch,
                inputs=[outline_box, guidance_box] + config_inputs,
                outputs=[chapter_output, debate_output, gen_chars_display, gen_threads_display, gen_log_display],
            )

        # ---- Tab 3: Story Bible ----
        with gr.Tab("Story Bible"):
            gr.Markdown("## Story Bible")
            init_bible_btn = gr.Button("Initialize from Outline", variant="primary")
            bible_chars = gr.Markdown(label="Characters")
            bible_threads = gr.Markdown(label="Plot Threads")
            bible_timeline = gr.Markdown(label="Timeline")
            bible_notes = gr.Markdown(label="World Notes")

            init_bible_btn.click(
                _init_bible,
                inputs=[outline_box] + config_inputs,
                outputs=[bible_chars, bible_threads, bible_timeline, bible_notes],
            )

        # ---- Tab 4: Full Story ----
        with gr.Tab("Full Story"):
            gr.Markdown("## Full Story")
            with gr.Row():
                refresh_story_btn = gr.Button("Refresh")
                export_btn = gr.Button("Export as .txt")
            chapter_nav = gr.Dropdown(label="Navigate to Chapter", choices=[])
            scores_display = gr.Textbox(label="Chapter Scores", lines=5, interactive=False)
            full_story_display = gr.Textbox(label="Full Story", lines=30, interactive=False)
            single_chapter_display = gr.Textbox(label="Selected Chapter", lines=20, interactive=False)
            export_file = gr.File(label="Download")

            refresh_story_btn.click(
                _get_full_story,
                outputs=[full_story_display, chapter_nav, scores_display],
            )
            chapter_nav.change(
                _navigate_chapter,
                inputs=[chapter_nav],
                outputs=[single_chapter_display],
            )
            export_btn.click(_export_story, outputs=[export_file])

        # ---- Tab 5: Agent Log ----
        with gr.Tab("Agent Log"):
            gr.Markdown("## Agent Activity Log")
            refresh_log_btn = gr.Button("Refresh Log")
            log_display = gr.Textbox(label="Log", lines=30, interactive=False)

            def _refresh_log():
                pipe = _state["pipeline"]
                if pipe:
                    log_lines = []
                    for l in pipe.all_logs:
                        log_lines.append(
                            f"[{l.agent_name}] {l.action} ({l.elapsed_seconds:.1f}s) "
                            f"| prompt: {l.prompt_preview[:80]}... "
                            f"| response: {l.response_preview[:80]}..."
                        )
                    detailed = "\n".join(log_lines)
                else:
                    detailed = ""
                return _state["log_text"] + "\n\n--- Detailed Agent Logs ---\n" + detailed

            refresh_log_btn.click(_refresh_log, outputs=[log_display])

    return app


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Story Generator v2")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
