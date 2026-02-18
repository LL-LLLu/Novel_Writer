"""
Novel Writer Web UI - Story Workshop with LoRA fine-tuned models.

Plot outlines are generated via cloud APIs (Gemini / GPT) for best quality.
Chapter writing uses your local fine-tuned LoRA model for style-specific prose.

Usage:
    python3 scripts/webui.py
    python3 scripts/webui.py --port 7861

Requirements:
    pip install gradio torch transformers peft accelerate bitsandbytes sentencepiece
    pip install google-generativeai openai
"""

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from threading import Lock

import gradio as gr
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# System prompts (must match training)
# ---------------------------------------------------------------------------
ZH_SYSTEM = (
    "你是一位经验丰富的中文小说作家，擅长构建沉浸式的叙事场景。请根据给定的上下文续写故事，要求：\n"
    "1. 保持与原文一致的叙事视角和文风\n"
    "2. 通过具体的动作、对话和环境描写推动情节发展\n"
    "3. 角色的言行应符合其性格特征和当前情境\n"
    "4. 善用感官细节（视觉、听觉、触觉、嗅觉）营造氛围\n"
    "5. 对话要自然生动，符合角色身份和说话习惯\n"
    "6. 避免空洞的心理独白，用行动和细节展现人物内心"
)

EN_SYSTEM = (
    "You are an accomplished fiction author with a gift for immersive storytelling. "
    "Continue the narrative following these principles:\n"
    "1. Maintain the established point of view, voice, and tonal register\n"
    "2. Advance the plot through concrete action, dialogue, and environmental detail\n"
    "3. Show character emotion through behavior, body language, and subtext — not exposition\n"
    "4. Engage multiple senses (sight, sound, touch, smell, taste) to ground scenes\n"
    "5. Write dialogue that reveals character, creates tension, and sounds natural\n"
    "6. Vary sentence rhythm — mix short punchy lines with longer flowing passages"
)

# ---------------------------------------------------------------------------
# Cloud API plot generation
# ---------------------------------------------------------------------------

def _build_plot_prompt(idea: str, num_chapters: int, lang: str) -> tuple[str, str]:
    """Build the system prompt and user prompt for plot development."""
    if lang == "zh":
        system = (
            "你是一位资深的小说策划编辑和故事架构师。你擅长从简单的故事构思中发展出完整、"
            "引人入胜的小说大纲。你的大纲应该包含极其丰富的细节，足以直接指导AI模型逐章生成高质量的小说内容。"
            "每个章节的大纲都应该详细到可以独立作为写作指南。"
        )
        prompt = f"""请基于以下故事构思，创作一个非常详细的小说大纲：

故事构思：{idea}

请严格按照以下格式输出（使用中文）：

## 小说标题
[一个引人入胜的标题]

## 故事背景
[详细的世界观和背景设定，至少200字。包括：时代背景、地理环境、社会体制、文化风俗、特殊设定（如武功体系、魔法规则等）]

## 主要人物
（至少4个主要角色，每个角色需包含详细信息）
- **[角色全名]**（[年龄/外貌简述]）：[性格特点——至少3个性格关键词]，[身份背景——家族/门派/职业]，[核心动机——驱动角色行动的内在欲望]，[人物弧光——角色在故事中的成长变化轨迹]，[与其他角色的关键关系]

## 核心冲突
[故事的主要矛盾和驱动力，包括外部冲突和内部冲突，至少100字]

## 章节大纲
（共{num_chapters}章，每章需要非常具体的情节描述）
"""
        for i in range(1, num_chapters + 1):
            prompt += f"""
### 第{i}章：[章节标题]
- **开场场景**：[具体的时间、地点、氛围描写]
- **主要事件**：[本章发生的1-3个关键事件，按顺序详细描述]
- **人物互动**：[哪些角色出场，他们之间的对话和冲突要点]
- **情感节奏**：[本章的情感基调变化，从开头到结尾的情绪走向]
- **关键细节**：[需要着重描写的场景细节、物品、环境元素]
- **章末转折**：[本章结尾的悬念、伏笔或转折点]
"""
        prompt += """
## 伏笔与线索
[列出3-5个贯穿全文的伏笔和线索，说明它们在哪些章节出现和揭示]

## 写作风格指导
[对本小说整体风格的建议：叙事视角、语言风格、节奏控制等]

请确保：
1. 章节之间有清晰的因果关系和情节递进
2. 整体故事有完整的起承转合
3. 人物发展有合理的成长曲线
4. 伏笔和悬念合理分布，不要集中在某几章
5. 每章大纲足够详细，能直接指导AI生成2000-4000字的章节内容"""
    else:
        system = (
            "You are a senior fiction editor and story architect. You excel at developing "
            "simple story concepts into complete, compelling novel outlines. Your outlines "
            "must contain exceptionally rich detail — enough to directly guide an AI model "
            "in generating high-quality chapter content. Each chapter outline should work "
            "as a standalone writing brief."
        )
        prompt = f"""Based on the following story idea, create a very detailed novel outline:

Story idea: {idea}

Please use this exact format:

## Title
[A compelling title]

## Setting
[Detailed world-building, at least 200 words. Include: time period, geography, social systems, cultural norms, special rules (magic systems, technology, etc.)]

## Main Characters
(At least 4 main characters, each with detailed profiles)
- **[Full Name]** ([Age/Appearance]): [Personality — at least 3 key traits], [Background — family/organization/profession], [Core motivation — the inner desire driving their actions], [Character arc — how they change through the story], [Key relationships with other characters]

## Central Conflict
[Main tension driving the story, including external and internal conflicts, at least 100 words]

## Chapter Outline
({num_chapters} chapters, each with very specific plot details)
"""
        for i in range(1, num_chapters + 1):
            prompt += f"""
### Chapter {i}: [Title]
- **Opening scene**: [Specific time, place, and atmosphere]
- **Key events**: [1-3 major events in this chapter, described in sequence]
- **Character interactions**: [Which characters appear, dialogue and conflict points]
- **Emotional rhythm**: [The emotional tone arc from beginning to end of chapter]
- **Key details**: [Important scene details, objects, environmental elements to emphasize]
- **Chapter-end hook**: [The cliffhanger, foreshadowing, or turning point at chapter's end]
"""
        prompt += """
## Foreshadowing & Threads
[List 3-5 narrative threads running through the story, noting where they appear and resolve]

## Style Guide
[Recommendations for overall style: narrative POV, prose register, pacing approach]

Ensure:
1. Clear cause-and-effect between chapters
2. Complete narrative arc (setup, confrontation, resolution)
3. Believable character growth curves
4. Foreshadowing and suspense distributed evenly
5. Each chapter outline is detailed enough to guide generation of 2000-4000 words"""

    return system, prompt


def generate_plot_gemini(api_key: str, idea: str, num_chapters: int, lang: str) -> str:
    """Generate plot outline using Google Gemini."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    system, prompt = _build_plot_prompt(idea, num_chapters, lang)

    response = client.models.generate_content(
        model="gemini-3-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.9,
            max_output_tokens=8192,
        ),
    )
    return response.text


def generate_plot_gpt(api_key: str, idea: str, num_chapters: int, lang: str) -> str:
    """Generate plot outline using OpenAI GPT."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    system, prompt = _build_plot_prompt(idea, num_chapters, lang)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=8192,
    )
    return response.choices[0].message.content


def develop_plot_api(
    idea: str,
    num_chapters: int,
    api_provider: str,
    api_key: str,
    progress=gr.Progress(),
):
    """Generate a detailed plot outline via cloud API."""
    if not idea.strip():
        return "Please enter a story idea first."
    if not api_key.strip():
        return f"Please enter your {api_provider} API key in the settings above."

    lang = detect_language(idea)
    progress(0.1, desc=f"Generating outline via {api_provider}...")

    try:
        if api_provider == "Gemini":
            result = generate_plot_gemini(api_key, idea, num_chapters, lang)
        elif api_provider == "GPT":
            result = generate_plot_gpt(api_key, idea, num_chapters, lang)
        else:
            return f"Unknown provider: {api_provider}"
        progress(1.0, desc="Plot developed!")
        return result
    except Exception as e:
        return f"API Error ({api_provider}): {e}"


# ---------------------------------------------------------------------------
# Local model state
# ---------------------------------------------------------------------------
_model = None
_tokenizer = None
_model_name = ""
_lock = Lock()


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_language(text: str) -> str:
    cjk = sum(1 for c in text[:300] if '\u4e00' <= c <= '\u9fff')
    return "zh" if cjk > len(text[:300]) * 0.15 else "en"


def scan_lora_adapters():
    """Scan the models/ directory for LoRA adapters."""
    adapters = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and (d / "adapter_config.json").exists():
                try:
                    cfg = json.loads((d / "adapter_config.json").read_text())
                    base = cfg.get("base_model_name_or_path", "unknown")
                    adapters.append({
                        "name": d.name,
                        "path": str(d),
                        "base_model": base,
                    })
                except Exception:
                    adapters.append({
                        "name": d.name,
                        "path": str(d),
                        "base_model": "unknown",
                    })
    return adapters


def get_base_model_name(lora_base: str) -> str:
    """Convert unsloth bnb model names to standard HF names for Mac loading."""
    mapping = {
        "unsloth/qwen3-32b-bnb-4bit": "Qwen/Qwen3-32B",
        "unsloth/qwen3-8b-bnb-4bit": "Qwen/Qwen3-8B",
        "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
        "unsloth/Qwen3-32B": "Qwen/Qwen3-32B",
        "unsloth/Qwen3-14B": "Qwen/Qwen3-14B",
        "unsloth/Qwen3-4B": "Qwen/Qwen3-4B",
        "unsloth/llama-3-8b-instruct-bnb-4bit": "meta-llama/Llama-3.1-8B-Instruct",
        "unsloth/mistral-nemo-instruct-2407-bnb-4bit": "mistralai/Mistral-Nemo-Instruct-2407",
    }
    return mapping.get(lora_base, lora_base)


def load_model_fn(lora_choice: str, load_in_4bit: bool, progress=gr.Progress()):
    """Load a model + LoRA adapter."""
    global _model, _tokenizer, _model_name

    with _lock:
        if _model is not None:
            del _model
            del _tokenizer
            _model = None
            _tokenizer = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        adapters = scan_lora_adapters()
        if not adapters:
            return "No LoRA adapters found in models/ directory."

        adapter = next((a for a in adapters if a["name"] == lora_choice), None)
        if adapter is None:
            return f"Adapter '{lora_choice}' not found."

        lora_path = adapter["path"]
        base_model = get_base_model_name(adapter["base_model"])

        progress(0.1, desc=f"Loading base model: {base_model}")

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        device = get_device()
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if load_in_4bit:
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["quantization_config"] = bnb_config
                load_kwargs["device_map"] = "auto"
            except Exception:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = {"": device}
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = {"": device}

        try:
            _tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            progress(0.4, desc="Base model downloading/loading...")
            _model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
            progress(0.7, desc="Applying LoRA adapter...")
            _model = PeftModel.from_pretrained(_model, lora_path)
            _model.eval()
            _model_name = f"{base_model} + {adapter['name']}"
            progress(1.0, desc="Done!")
            if device == "mps":
                mem = torch.mps.current_allocated_memory() / 1e9
                return f"Loaded: {_model_name}\nDevice: {next(_model.parameters()).device}\nMemory: {mem:.1f} GB"
            return f"Loaded: {_model_name}"
        except Exception as e:
            _model = None
            _tokenizer = None
            return f"Failed to load model: {e}\n\nTip: Try toggling '4-bit quantization' or use a smaller model."


def generate_text(
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
):
    """Generate text from the loaded local model."""
    if _model is None or _tokenizer is None:
        return "Please load a local model first (Model Settings above)."

    if not system_prompt:
        cjk = sum(1 for c in prompt[:200] if '\u4e00' <= c <= '\u9fff')
        system_prompt = ZH_SYSTEM if cjk > len(prompt[:200]) * 0.15 else EN_SYSTEM

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": max(temperature, 0.01),
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "do_sample": True,
    }

    with torch.no_grad():
        outputs = _model.generate(**inputs, **gen_kwargs)

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Chapter generation (local model)
# ---------------------------------------------------------------------------

def generate_chapter(
    plot_outline: str,
    chapter_num: int,
    previous_text: str,
    style_notes: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    progress=gr.Progress(),
):
    """Generate a single chapter using the local fine-tuned model."""
    if _model is None:
        return "Please load a local model first (Model Settings above)."

    lang = detect_language(plot_outline)
    progress(0.1, desc=f"Generating chapter {chapter_num}...")

    # Extract the specific chapter outline
    chapter_pattern = rf"(?:###?\s*(?:第{chapter_num}章|Chapter\s*{chapter_num})[：:\s]*)(.+?)(?=(?:###?\s*(?:第\d+章|Chapter\s*\d+))|$)"
    match = re.search(chapter_pattern, plot_outline, re.DOTALL)
    chapter_outline = match.group(0).strip() if match else f"Chapter {chapter_num}"

    if lang == "zh":
        prompt = f"## 小说大纲\n{plot_outline}\n\n"
        if previous_text:
            ctx = previous_text[-2000:] if len(previous_text) > 2000 else previous_text
            prompt += f"## 上一章结尾\n{ctx}\n\n"
        prompt += f"## 当前任务\n请根据以上大纲，撰写第{chapter_num}章的完整内容。\n"
        prompt += f"本章大纲：{chapter_outline}\n\n"
        if style_notes:
            prompt += f"风格要求：{style_notes}\n\n"
        prompt += (
            "要求：\n"
            "1. 以具体的场景描写开头，营造氛围\n"
            "2. 通过对话和动作推动情节\n"
            "3. 注意人物性格的一致性\n"
            "4. 章节结尾要有悬念或转折\n"
            "5. 写出完整的章节内容，不要省略或概括\n\n"
            f"第{chapter_num}章正文：\n"
        )
    else:
        prompt = f"## Novel Outline\n{plot_outline}\n\n"
        if previous_text:
            ctx = previous_text[-2000:] if len(previous_text) > 2000 else previous_text
            prompt += f"## End of Previous Chapter\n{ctx}\n\n"
        prompt += f"## Current Task\nWrite the complete text of Chapter {chapter_num}.\n"
        prompt += f"Chapter outline: {chapter_outline}\n\n"
        if style_notes:
            prompt += f"Style notes: {style_notes}\n\n"
        prompt += (
            "Requirements:\n"
            "1. Open with vivid scene-setting\n"
            "2. Drive the plot through dialogue and action\n"
            "3. Maintain consistent characterization\n"
            "4. End with a hook or turning point\n"
            "5. Write the complete chapter — do not summarize or skip ahead\n\n"
            f"Chapter {chapter_num}:\n"
        )

    system = ZH_SYSTEM if lang == "zh" else EN_SYSTEM
    result = generate_text(
        prompt,
        system_prompt=system,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    progress(1.0, desc="Chapter generated!")
    return result


def continue_writing(
    existing_text: str,
    instruction: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    """Continue writing from existing text using local model."""
    if _model is None:
        return "Please load a local model first (Model Settings above)."

    lang = detect_language(existing_text)
    if not instruction:
        instruction = "续写这段叙事，保持原文的风格和节奏。" if lang == "zh" else "Continue the narrative in the established style."

    prompt = instruction + "\n\n" + existing_text
    system = ZH_SYSTEM if lang == "zh" else EN_SYSTEM
    return generate_text(
        prompt, system_prompt=system, max_new_tokens=max_tokens,
        temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
    )


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    adapters = scan_lora_adapters()
    adapter_names = [a["name"] for a in adapters] if adapters else ["No adapters found"]
    adapter_info = {a["name"]: a["base_model"] for a in adapters}

    _theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="violet")
    _css = """
    .chapter-output { min-height: 400px; }
    .plot-output { min-height: 300px; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Novel Writer Studio") as app:
        gr.Markdown("# Novel Writer Studio")
        gr.Markdown(
            "*Cloud AI (Gemini / GPT) develops your story outline — "
            "your fine-tuned local model writes the chapters*"
        )

        # ---- API Settings ----
        with gr.Accordion("API Settings (for Plot Generation)", open=True):
            gr.Markdown(
                "Plot outlines are generated by Gemini or GPT for best quality. "
                "Enter your API key below. Keys are only used in-memory and never stored."
            )
            with gr.Row():
                api_provider = gr.Radio(
                    choices=["Gemini", "GPT"],
                    value="Gemini",
                    label="Provider",
                    info="Gemini = Google AI, GPT = OpenAI",
                )
                api_key_input = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Paste your API key here...",
                    scale=3,
                )
            api_status = gr.Markdown("")

            def check_api(provider, key):
                if not key.strip():
                    return ""
                try:
                    if provider == "Gemini":
                        from google import genai
                        from google.genai import types
                        client = genai.Client(api_key=key)
                        client.models.generate_content(
                            model="gemini-3-pro-preview",
                            contents="Say OK",
                            config=types.GenerateContentConfig(max_output_tokens=5),
                        )
                        return "Gemini API key verified."
                    else:
                        from openai import OpenAI
                        client = OpenAI(api_key=key)
                        client.chat.completions.create(
                            model="gpt-4o-mini", messages=[{"role": "user", "content": "Say OK"}], max_tokens=5,
                        )
                        return "OpenAI API key verified."
                except Exception as e:
                    return f"API key error: {e}"

            api_key_input.change(check_api, [api_provider, api_key_input], api_status)

        # ---- Local Model Settings ----
        with gr.Accordion("Local Model Settings (for Chapter Writing)", open=False):
            gr.Markdown("Load your fine-tuned LoRA model for chapter generation.")
            with gr.Row():
                with gr.Column(scale=2):
                    lora_dropdown = gr.Dropdown(
                        choices=adapter_names,
                        value=adapter_names[0] if adapter_names else None,
                        label="LoRA Adapter",
                        info="Select from models/ directory",
                    )
                    base_model_info = gr.Textbox(
                        value=adapter_info.get(adapter_names[0], "") if adapter_names else "",
                        label="Base Model",
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    load_4bit = gr.Checkbox(
                        value=True,
                        label="4-bit Quantization",
                        info="~18GB for 32B model. Uncheck for fp16 (~64GB).",
                    )
                    load_btn = gr.Button("Load Model", variant="primary", size="lg")
                    model_status = gr.Textbox(label="Status", interactive=False, lines=3)

            def update_base_info(choice):
                return adapter_info.get(choice, "unknown")

            lora_dropdown.change(update_base_info, lora_dropdown, base_model_info)
            load_btn.click(load_model_fn, [lora_dropdown, load_4bit], model_status)

        # ---- Generation settings ----
        with gr.Accordion("Chapter Generation Settings", open=False):
            with gr.Row():
                max_tokens_slider = gr.Slider(128, 4096, value=2048, step=128, label="Max New Tokens")
                temperature_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            with gr.Row():
                top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                rep_penalty_slider = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")

        # ---- Tabs ----
        with gr.Tabs():
            # ===== Tab 1: Story Workshop =====
            with gr.Tab("Story Workshop"):
                gr.Markdown("### Step 1: Your Story Idea")
                with gr.Row():
                    with gr.Column(scale=3):
                        story_idea = gr.Textbox(
                            label="Story Idea",
                            placeholder=(
                                "例：在一个武林高手辈出的时代，一个失忆的少年在雪山醒来，"
                                "身上带着一柄无人能拔出的奇剑...\n\n"
                                "Or: In a world where magic is fueled by music, a deaf girl "
                                "discovers she can cast the most powerful spells..."
                            ),
                            lines=4,
                        )
                    with gr.Column(scale=1):
                        num_chapters = gr.Slider(3, 20, value=8, step=1, label="Chapters")
                        develop_btn = gr.Button(
                            "Develop Plot (Cloud AI)", variant="primary", size="lg",
                        )

                gr.Markdown("### Step 2: Plot Outline")
                gr.Markdown(
                    "Generated by Gemini/GPT. Review and edit freely — "
                    "the more detailed the outline, the better your chapters will be."
                )
                plot_outline = gr.Textbox(
                    label="Plot Outline (editable)",
                    lines=25,
                    placeholder="Click 'Develop Plot' to generate an outline from your story idea...",
                    elem_classes=["plot-output"],
                )

                develop_btn.click(
                    develop_plot_api,
                    [story_idea, num_chapters, api_provider, api_key_input],
                    plot_outline,
                )

                gr.Markdown("---")
                gr.Markdown("### Step 3: Generate Chapters (Local Model)")
                gr.Markdown(
                    "Uses your fine-tuned LoRA model to write each chapter "
                    "in the trained literary style. Make sure the local model is loaded above."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        chapter_num = gr.Slider(1, 20, value=1, step=1, label="Chapter Number")
                        style_notes = gr.Textbox(
                            label="Style Notes (optional)",
                            placeholder="e.g., 多用对话 / focus on action / noir tone",
                            lines=2,
                        )
                        gen_chapter_btn = gr.Button(
                            "Generate Chapter (Local)", variant="primary", size="lg",
                        )
                    with gr.Column(scale=3):
                        chapter_output = gr.Textbox(
                            label="Generated Chapter",
                            lines=25,
                            elem_classes=["chapter-output"],
                        )

                all_chapters = gr.State("")

                def gen_and_accumulate(outline, ch_num, prev, style, max_t, temp, tp, rp, progress=gr.Progress()):
                    result = generate_chapter(outline, ch_num, prev, style, max_t, temp, tp, rp, progress)
                    sep = f"\n\n{'='*40}\n第{int(ch_num)}章 / Chapter {int(ch_num)}\n{'='*40}\n\n"
                    new_acc = prev + sep + result if prev else result
                    return result, new_acc

                gen_chapter_btn.click(
                    gen_and_accumulate,
                    [plot_outline, chapter_num, all_chapters, style_notes,
                     max_tokens_slider, temperature_slider, top_p_slider, rep_penalty_slider],
                    [chapter_output, all_chapters],
                )

                with gr.Accordion("All Generated Chapters", open=False):
                    all_chapters_display = gr.Textbox(
                        label="Full Story So Far", lines=30, interactive=False,
                    )
                    refresh_btn = gr.Button("Refresh")
                    refresh_btn.click(lambda x: x, all_chapters, all_chapters_display)

                    export_btn = gr.Button("Export Story to File")
                    export_status = gr.Textbox(label="Export Status", interactive=False)

                    def export_story(text):
                        if not text:
                            return "Nothing to export."
                        out_path = PROJECT_ROOT / "output"
                        out_path.mkdir(exist_ok=True)
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        fpath = out_path / f"story_{ts}.txt"
                        fpath.write_text(text, encoding="utf-8")
                        return f"Exported to: {fpath}"

                    export_btn.click(export_story, all_chapters, export_status)

            # ===== Tab 2: Free Write =====
            with gr.Tab("Free Write"):
                gr.Markdown("### Direct Generation (Local Model)")
                gr.Markdown("Write freely — provide context for continuation or start fresh.")

                with gr.Row():
                    with gr.Column():
                        free_context = gr.Textbox(
                            label="Context / Previous Text",
                            placeholder="Paste existing story text here for continuation (optional)...",
                            lines=10,
                        )
                        free_instruction = gr.Textbox(
                            label="Instruction (optional)",
                            placeholder="e.g., 续写这段战斗场景 / Continue with more dialogue",
                            lines=2,
                        )
                        free_gen_btn = gr.Button("Generate", variant="primary", size="lg")
                    with gr.Column():
                        free_output = gr.Textbox(label="Generated Text", lines=20)
                        append_btn = gr.Button("Append to Context")

                free_gen_btn.click(
                    continue_writing,
                    [free_context, free_instruction,
                     max_tokens_slider, temperature_slider, top_p_slider, rep_penalty_slider],
                    free_output,
                )

                def append_to_context(ctx, out):
                    return ctx + "\n" + out if ctx else out

                append_btn.click(append_to_context, [free_context, free_output], free_context)

            # ===== Tab 3: Quick Test =====
            with gr.Tab("Quick Test"):
                gr.Markdown("### Test the local model with a single prompt")
                test_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="月色如霜，照在悬崖边两道对峙的身影上...",
                    lines=4,
                )
                test_system = gr.Textbox(
                    label="System Prompt (optional, auto-detected if empty)",
                    lines=3,
                )
                test_btn = gr.Button("Generate", variant="primary")
                test_output = gr.Textbox(label="Output", lines=15)

                def run_quick_test(prompt, sys_prompt, max_t, temp, tp, rp):
                    return generate_text(prompt, sys_prompt, max_t, temp, tp, 50, rp)

                test_btn.click(
                    run_quick_test,
                    [test_prompt, test_system,
                     max_tokens_slider, temperature_slider, top_p_slider, rep_penalty_slider],
                    test_output,
                )

    return app, _theme, _css


def main():
    parser = argparse.ArgumentParser(description="Novel Writer Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    app, theme, css = build_ui()
    app.launch(server_port=args.port, share=args.share, theme=theme, css=css)


if __name__ == "__main__":
    main()
