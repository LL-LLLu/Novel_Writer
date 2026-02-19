"""
Novel Writer Web UI - Story Workshop with LoRA fine-tuned models.

Plot outlines are generated via cloud APIs (Gemini / GPT) for best quality.
Chapter writing uses your local fine-tuned LoRA model for style-specific prose.
Multi-agent system (Mastermind + Story Tracker) orchestrates chapter generation
for plot adherence, continuity, and quality.

Usage:
    python3 scripts/webui.py
    python3 scripts/webui.py --port 7861

Requirements:
    pip install gradio torch transformers peft accelerate bitsandbytes sentencepiece
    pip install google-genai openai
"""

import argparse
import copy
import gc
import json
import os
import random
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
# Instruction pools (from training data — format.py)
# ---------------------------------------------------------------------------
_ZH_INSTRUCTIONS = [
    "续写这段叙事，保持原文的风格和节奏。",
    "以相同的文风继续这个故事。",
    "根据已有的情节和人物设定，续写下一段。",
    "保持叙事视角不变，继续推进故事发展。",
    "用生动的细节描写续写这个场景。",
    "通过对话和动作描写推进下面的情节。",
    "延续当前的叙事氛围，写出接下来发生的事。",
    "以细腻的笔触续写这段文字。",
    "按照原文的叙事节奏，写出故事的下一部分。",
    "继续描绘这个场景中的人物和事件。",
    "用符合原文风格的语言续写故事。",
    "展开叙述，让故事自然地向前发展。",
    "保持文风一致，续写接下来的情节。",
    "以沉浸式的叙事方式继续这段故事。",
    "描绘接下来的场景，注意环境和人物的刻画。",
    "用简洁有力的文字续写这段叙事。",
    "继续讲述这个故事，注意情感的表达。",
    "以自然流畅的文笔续写下一段。",
    "延续原文的基调，推进故事走向。",
    "用丰富的感官描写续写这个场景。",
]

_EN_INSTRUCTIONS = [
    "Continue the narrative in the established style.",
    "Write the next passage, maintaining the existing voice and tone.",
    "Advance the story using vivid sensory details.",
    "Continue this scene with natural dialogue and action.",
    "Extend the narrative, preserving the point of view and pacing.",
    "Write what happens next, staying true to the characters.",
    "Continue the story with concrete, immersive description.",
    "Carry the narrative forward in the same literary register.",
    "Write the next segment, matching the established rhythm.",
    "Develop this scene further with authentic detail.",
    "Push the story forward through action and dialogue.",
    "Continue in the same voice, advancing the plot naturally.",
    "Write the following passage in the style of the preceding text.",
    "Extend this scene with attention to atmosphere and character.",
    "Continue the narrative arc with engaging prose.",
    "Write what comes next, maintaining tension and pacing.",
    "Advance the story, weaving in environmental detail.",
    "Continue with prose that matches the tone and texture of the original.",
    "Develop the next beat of the story with precise language.",
    "Carry the scene forward, balancing action with description.",
]

# ---------------------------------------------------------------------------
# Token utilities
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH = 4096


def count_tokens(text: str) -> int:
    """Count tokens using the loaded tokenizer, or estimate if not loaded."""
    if _tokenizer is not None:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    # Rough estimate: ~1.5 chars/token for CJK, ~4 chars/token for English
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    en_chars = len(text) - cjk
    return int(cjk / 1.5) + int(en_chars / 4)


def _find_sentence_boundary(text: str, max_chars: int, from_end: bool = False) -> int:
    """Find the nearest sentence boundary within max_chars.

    If from_end=True, search from the end of text backwards.
    Returns char index for slicing.
    """
    sentence_ends_cjk = re.compile(r'[。！？…]+')
    sentence_ends_en = re.compile(r'[.!?][\u201c\u201d\u2018\u2019"\')\u300d\uff09]*(?:\s|\n)')

    if from_end:
        search_region = text[-max_chars:] if len(text) > max_chars else text
        offset = max(0, len(text) - max_chars)
        # Find the first sentence boundary in the search region
        best = 0
        for m in sentence_ends_cjk.finditer(search_region):
            best = m.end()
            break
        for m in sentence_ends_en.finditer(search_region):
            if m.end() < best or best == 0:
                best = m.end()
            break
        return offset + best if best > 0 else offset
    else:
        search_region = text[:max_chars]
        # Find the last sentence boundary
        best = max_chars
        for m in sentence_ends_cjk.finditer(search_region):
            best = m.end()
        for m in sentence_ends_en.finditer(search_region):
            best = m.end()
        return best


def trim_to_token_budget(text: str, max_tokens: int, keep_end: bool = True) -> str:
    """Trim text to fit within token budget, snapping to sentence boundaries.

    keep_end=True: keep the end of text (truncate beginning) — for context.
    keep_end=False: keep the beginning (truncate end) — for primers.
    """
    current = count_tokens(text)
    if current <= max_tokens:
        return text

    # Estimate chars to keep (rough: tokens * avg_chars_per_token)
    ratio = max_tokens / max(current, 1)
    target_chars = int(len(text) * ratio * 0.95)  # 5% safety margin

    if keep_end:
        # Keep the end, cut from beginning
        start = _find_sentence_boundary(text, len(text) - target_chars, from_end=True)
        trimmed = text[start:]
    else:
        # Keep the beginning, cut from end
        end = _find_sentence_boundary(text, target_chars, from_end=False)
        trimmed = text[:end]

    # Verify and re-trim if needed
    if count_tokens(trimmed) > max_tokens:
        if keep_end:
            trimmed = text[-target_chars:]
        else:
            trimmed = text[:target_chars]

    return trimmed.strip()


# ---------------------------------------------------------------------------
# Cloud API: plot generation
# ---------------------------------------------------------------------------

def _build_plot_prompt(idea: str, target_chapters: int, lang: str) -> tuple[str, str]:
    """Build the system prompt and user prompt for plot development.

    Generates a high-level story structure with acts/arcs rather than
    chapter-by-chapter outlines. The user can edit the outline and the
    Mastermind agent will plan individual chapters from it.
    """
    if lang == "zh":
        system = (
            "你是一位资深的小说策划编辑和故事架构师。你擅长从简单的故事构思中发展出宏大、"
            "引人入胜的长篇小说大纲。你的大纲以故事弧线和幕（Act）为结构单位——不要逐章列举。"
            "大纲应包含极其丰富的细节：人物关系网、世界观设定、核心冲突的多层递进、"
            "以及每个故事弧线中的关键转折点。这份大纲将作为AI逐章生成小说的总蓝图。"
        )
        prompt = f"""请基于以下故事构思，创作一个宏大而详细的长篇小说大纲。
目标篇幅：约{target_chapters}章的长篇故事。

故事构思：{idea}

请严格按照以下格式输出（使用中文）：

## 小说标题
[一个引人入胜的标题]

## 故事背景
[详细的世界观和背景设定，至少300字。包括：时代背景、地理环境、社会体制、文化风俗、历史渊源、特殊设定（如武功体系、魔法规则、科技水平等）。要有层次感——大世界背景、故事发生的具体区域、以及主角的生活圈]

## 主要人物
（至少6个角色：主角、对手、盟友、导师、情感关联角色、关键配角）
- **[角色全名]**（[年龄/外貌简述]）：[性格特点——至少3个关键词]，[身份背景]，[核心动机]，[人物弧光——从故事开始到结束的完整变化轨迹]，[秘密或隐藏面]，[关键人际关系]

## 人物关系网
[用文字描述主要人物之间的关系图谱：谁和谁是盟友/对手/师徒/恋人/亲人，关系如何随故事发展变化]

## 核心冲突
[故事的多层矛盾结构，至少200字：
- 外部冲突：主角面对的外在威胁或障碍
- 内部冲突：主角的内心挣扎和道德困境
- 社会冲突：更宏大的社会/世界层面的矛盾
- 关系冲突：人物之间的矛盾和张力]

## 故事弧线

### 第一幕：起（约占全书前20%）
**核心目标**：[这一幕要完成什么——建立世界、引入角色、设置悬念]
**起始状态**：[故事开始时主角和世界的状态]
**关键场景**：
1. [开篇场景：具体描述，包括环境、人物、事件]
2. [引发事件：打破日常的事件，详细描述]
3. [第一个转折：推动主角踏上旅程的催化剂]
**情感基调**：[从平静到不安，或从混乱到希望等]
**埋下的伏笔**：[在这一幕中埋下哪些伏笔，将在后面揭示]

### 第二幕上半：承（约占全书20%-50%）
**核心目标**：[这一幕要完成什么——展开冒险、加深冲突、发展关系]
**关键场景**：
1. [重要事件1：详细描述]
2. [重要事件2：详细描述]
3. [重要事件3：详细描述]
4. [中点转折：故事中点的重大转折或启示]
**人物发展**：[各角色在这一段的成长和变化]
**情感基调**：[逐渐升温的紧张感]
**关键谜团/悬念**：[让读者欲罢不能的谜团]

### 第二幕下半：转（约占全书50%-75%）
**核心目标**：[加剧危机、黑暗时刻、联盟瓦解或重组]
**关键场景**：
1. [危机加剧的关键事件]
2. [背叛/揭秘/重大失败]
3. [最低谷：主角面临最大的困境]
**人物发展**：[角色弧光的关键转变]
**情感基调**：[从希望到绝望的转变]
**伏笔揭示**：[哪些之前埋下的伏笔在这里揭示]

### 第三幕：合（约占全书后25%）
**核心目标**：[走向高潮、解决冲突、完成角色弧光]
**关键场景**：
1. [重整旗鼓/获得关键力量或信息]
2. [最终对决/高潮场景的详细描述]
3. [结局：各人物的命运，故事的收束]
**人物发展**：[角色弧光的完成]
**情感基调**：[从绝境到高潮的爆发]
**主题升华**：[故事的核心主题如何在结尾得到升华]

## 伏笔与线索网络
[列出5-8个贯穿全文的伏笔和线索，说明它们在哪个弧线中出现和揭示。形成交叉的线索网]

## 主题与象征
[故事的深层主题——不只是情节，还有故事想要表达的思想。包括反复出现的象征物、意象]

## 写作风格指导
[叙事视角（第一人称/第三人称/多视角）、语言风格、节奏控制、对话风格、描写偏重]

请确保：
1. 故事弧线之间有清晰的因果递进和张力升级
2. 人物发展有合理的成长曲线，不要突然转变
3. 伏笔和悬念形成网络，前后呼应
4. 大纲足够详细，能支撑{target_chapters}章的长篇创作
5. 每个弧线的关键场景描写具体到可以直接指导写作"""
    else:
        system = (
            "You are a senior fiction editor and story architect. You excel at developing "
            "simple story concepts into expansive, compelling novel outlines. Your outlines "
            "use story arcs and acts as structural units — NOT individual chapters. "
            "Include exceptionally rich detail: character relationship webs, world-building, "
            "multi-layered conflict escalation, and key turning points in each arc. "
            "This outline will serve as the master blueprint for AI chapter-by-chapter generation."
        )
        prompt = f"""Based on the following story idea, create an expansive and detailed novel outline.
Target length: approximately {target_chapters} chapters.

Story idea: {idea}

Please use this exact format:

## Title
[A compelling title]

## Setting
[Detailed world-building, at least 300 words. Include: time period, geography, social systems, cultural norms, history, special rules (magic systems, technology, etc.). Layer it — broad world, specific region, protagonist's immediate circle]

## Main Characters
(At least 6 characters: protagonist, antagonist, ally, mentor, love interest, key supporting)
- **[Full Name]** ([Age/Appearance]): [Personality — at least 3 key traits], [Background], [Core motivation], [Character arc — complete transformation from start to end], [Secret or hidden side], [Key relationships]

## Relationship Web
[Describe the relationship map: who are allies/rivals/mentor-student/lovers/family, how relationships evolve through the story]

## Central Conflict
[Multi-layered conflict structure, at least 200 words:
- External conflict: threats or obstacles the protagonist faces
- Internal conflict: inner struggles and moral dilemmas
- Social conflict: larger societal/world-level tensions
- Relationship conflict: interpersonal tensions and rifts]

## Story Arcs

### Act I: Setup (roughly the first 20% of the story)
**Core goal**: [What this act must accomplish — establish world, introduce characters, set up mysteries]
**Starting state**: [The protagonist's and world's status quo]
**Key scenes**:
1. [Opening scene: specific description with setting, characters, events]
2. [Inciting incident: the event that disrupts the status quo]
3. [First turning point: the catalyst that launches the protagonist's journey]
**Emotional tone**: [e.g., calm to uneasy, chaos to fragile hope]
**Foreshadowing planted**: [What seeds are sown here for later payoff]

### Act II-A: Rising Action (roughly 20%-50%)
**Core goal**: [Expand the adventure, deepen conflicts, develop relationships]
**Key scenes**:
1. [Major event 1: detailed description]
2. [Major event 2: detailed description]
3. [Major event 3: detailed description]
4. [Midpoint twist: the major revelation or reversal at the story's center]
**Character development**: [How each character grows in this section]
**Emotional tone**: [Building tension]
**Key mysteries/suspense**: [What keeps readers turning pages]

### Act II-B: Complications (roughly 50%-75%)
**Core goal**: [Escalate crisis, dark moment, alliances fracture or reform]
**Key scenes**:
1. [Key event escalating the crisis]
2. [Betrayal/revelation/major failure]
3. [All-is-lost moment: protagonist's lowest point]
**Character development**: [Critical arc turning points]
**Emotional tone**: [Hope to despair]
**Foreshadowing revealed**: [Which earlier seeds pay off here]

### Act III: Resolution (roughly the final 25%)
**Core goal**: [Build to climax, resolve conflicts, complete character arcs]
**Key scenes**:
1. [Rallying/gaining crucial power or knowledge]
2. [Final confrontation/climax — detailed description]
3. [Resolution: fates of characters, story closure]
**Character development**: [Arcs completed]
**Emotional tone**: [From desperation to cathartic climax]
**Thematic payoff**: [How the story's core theme is crystallized in the ending]

## Foreshadowing & Thread Network
[List 5-8 narrative threads running through the story, noting which arc they appear in and where they resolve. Show how they interconnect]

## Themes & Symbolism
[The deeper themes — not just plot, but the ideas the story explores. Include recurring symbols and imagery]

## Style Guide
[Narrative POV (first person/third person/multiple), prose register, pacing, dialogue style, descriptive emphasis]

Ensure:
1. Clear cause-and-effect escalation between arcs
2. Believable character growth — no sudden personality shifts
3. Foreshadowing and suspense form an interconnected web
4. The outline is detailed enough to support {target_chapters} chapters of writing
5. Each arc's key scenes are specific enough to directly guide prose generation"""

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
# Unified cloud API helper (for agent calls)
# ---------------------------------------------------------------------------

def call_cloud_api(
    system: str,
    prompt: str,
    provider: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str:
    """Call cloud API (Gemini or GPT) with unified interface."""
    if provider == "Gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text
    elif provider == "GPT":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _parse_json_response(text: str) -> dict:
    """Extract JSON from a cloud API response that may contain markdown fences."""
    # Try to find JSON block in markdown code fence
    m = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try parsing the whole text as JSON
    # Strip leading/trailing whitespace and common prefixes
    cleaned = text.strip()
    if cleaned.startswith('{'):
        return json.loads(cleaned)
    # Last resort: find first { to last }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1:
        return json.loads(cleaned[start:end + 1])
    raise ValueError(f"Could not parse JSON from response: {text[:200]}...")


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


def upload_lora_zip(zip_file) -> str:
    """Upload and extract a LoRA adapter from a zip file into models/ directory."""
    import zipfile
    if zip_file is None:
        return "No file uploaded."
    zip_path = zip_file.name if hasattr(zip_file, "name") else str(zip_file)
    if not zip_path.endswith(".zip"):
        return "Please upload a .zip file containing your LoRA adapter."
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # Find the adapter_config.json to determine folder structure
            adapter_configs = [n for n in z.namelist() if n.endswith("adapter_config.json")]
            if not adapter_configs:
                return "No adapter_config.json found in zip. Make sure your zip contains a valid LoRA adapter."
            # Determine the top-level folder name
            config_path = adapter_configs[0]
            parts = config_path.split("/")
            if len(parts) > 1:
                # Files are inside a subfolder — extract normally
                z.extractall(MODELS_DIR)
                folder_name = parts[0]
            else:
                # Files are at the root of zip — create a folder from zip filename
                folder_name = Path(zip_path).stem
                dest = MODELS_DIR / folder_name
                dest.mkdir(exist_ok=True)
                for member in z.namelist():
                    # Extract flat files into the named folder
                    if not member.endswith("/"):
                        data = z.read(member)
                        (dest / Path(member).name).write_bytes(data)
        adapter_path = MODELS_DIR / folder_name
        if (adapter_path / "adapter_config.json").exists():
            cfg = json.loads((adapter_path / "adapter_config.json").read_text())
            base = cfg.get("base_model_name_or_path", "unknown")
            return f"Uploaded: {folder_name}\nBase model: {base}\nPath: {adapter_path}"
        return f"Extracted to {adapter_path}, but adapter_config.json not found at expected location."
    except Exception as e:
        return f"Upload failed: {e}"


def download_lora_hf(repo_id: str, progress=gr.Progress()) -> str:
    """Download a LoRA adapter from Hugging Face Hub into models/ directory."""
    if not repo_id or not repo_id.strip():
        return "Please enter a Hugging Face repo ID (e.g., username/model_name)."
    repo_id = repo_id.strip()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    folder_name = repo_id.split("/")[-1]
    dest = MODELS_DIR / folder_name
    try:
        from huggingface_hub import snapshot_download
        progress(0.1, desc=f"Downloading {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=str(dest))
        progress(1.0, desc="Download complete!")
        if (dest / "adapter_config.json").exists():
            cfg = json.loads((dest / "adapter_config.json").read_text())
            base = cfg.get("base_model_name_or_path", "unknown")
            return f"Downloaded: {folder_name}\nBase model: {base}\nPath: {dest}"
        return f"Downloaded to {dest}, but no adapter_config.json found. Is this a LoRA adapter repo?"
    except ImportError:
        return "huggingface_hub not installed. Run: pip install huggingface_hub"
    except Exception as e:
        return f"Download failed: {e}"


def refresh_lora_list():
    """Rescan models/ directory and return updated adapter list."""
    adapters = scan_lora_adapters()
    names = [a["name"] for a in adapters] if adapters else ["No adapters found"]
    info = {a["name"]: a["base_model"] for a in adapters}
    first = names[0] if adapters else None
    base = info.get(first, "") if first else ""
    return names, first, base, info


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
    repetition_penalty: float = 1.0,
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


def generate_text_cloud(
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    provider: str = "Gemini",
    api_key: str = "",
):
    """Generate text using cloud API (Gemini/GPT) instead of a local model.

    This enables running without any GPU — the cloud API does all the writing.
    """
    if not api_key.strip():
        return "Please enter your API key in Settings above."

    if not system_prompt:
        cjk = sum(1 for c in prompt[:200] if '\u4e00' <= c <= '\u9fff')
        system_prompt = ZH_SYSTEM if cjk > len(prompt[:200]) * 0.15 else EN_SYSTEM

    try:
        return call_cloud_api(
            system_prompt, prompt, provider, api_key,
            temperature=temperature, max_tokens=max_new_tokens,
        )
    except Exception as e:
        return f"Cloud API Error ({provider}): {e}"


# ---------------------------------------------------------------------------
# Story Tracker — maintains the story bible
# ---------------------------------------------------------------------------

class StoryTracker:
    """Maintains a 'story bible' tracking character states, plot threads,
    and chapter summaries for continuity across chapters."""

    @staticmethod
    def empty_bible() -> dict:
        return {
            "characters": {},
            "plot_threads": [],
            "chapter_summaries": [],
            "style_notes": "",
        }

    @staticmethod
    def initialize_from_outline(
        outline: str, provider: str, api_key: str, lang: str,
    ) -> dict:
        """Parse outline into a structured story bible via cloud API. (1 API call)"""
        if lang == "zh":
            system = "你是一位小说编辑助手。请从小说大纲中提取结构化信息，以JSON格式输出。"
            prompt = f"""请分析以下小说大纲，提取关键信息并以JSON格式输出：

{outline}

输出格式（必须是合法JSON）：
```json
{{
  "characters": {{
    "角色名": {{
      "description": "外貌和身份简述",
      "personality": "性格特点",
      "motivation": "核心动机",
      "relationships": {{"其他角色名": "关系描述"}},
      "location": "初始位置",
      "emotional_state": "初始情感状态"
    }}
  }},
  "plot_threads": [
    {{
      "name": "线索名称",
      "status": "active",
      "description": "线索描述"
    }}
  ],
  "style_notes": "写作风格要点摘要"
}}
```"""
        else:
            system = "You are a fiction editor assistant. Extract structured information from the novel outline as JSON."
            prompt = f"""Analyze the following novel outline and extract key information as JSON:

{outline}

Output format (must be valid JSON):
```json
{{
  "characters": {{
    "Character Name": {{
      "description": "Appearance and role",
      "personality": "Key traits",
      "motivation": "Core motivation",
      "relationships": {{"Other Character": "relationship"}},
      "location": "Starting location",
      "emotional_state": "Starting emotional state"
    }}
  }},
  "plot_threads": [
    {{
      "name": "Thread name",
      "status": "active",
      "description": "Thread description"
    }}
  ],
  "style_notes": "Style guide summary"
}}
```"""

        try:
            response = call_cloud_api(system, prompt, provider, api_key, temperature=0.3)
            parsed = _parse_json_response(response)
            bible = StoryTracker.empty_bible()
            bible["characters"] = parsed.get("characters", {})
            bible["plot_threads"] = parsed.get("plot_threads", [])
            bible["style_notes"] = parsed.get("style_notes", "")
            return bible
        except Exception as e:
            # Return empty bible on failure rather than crashing
            bible = StoryTracker.empty_bible()
            bible["style_notes"] = f"(Bible init failed: {e})"
            return bible

    @staticmethod
    def apply_updates(bible: dict, updates: dict) -> dict:
        """Apply chapter updates to the bible. (No API call)"""
        bible = copy.deepcopy(bible)

        # Update character states
        char_updates = updates.get("characters", {})
        for name, changes in char_updates.items():
            if name in bible["characters"]:
                bible["characters"][name].update(changes)
            else:
                bible["characters"][name] = changes

        # Update plot threads
        thread_updates = updates.get("plot_threads", [])
        existing_names = {t["name"] for t in bible["plot_threads"]}
        for thread in thread_updates:
            if thread["name"] in existing_names:
                for i, t in enumerate(bible["plot_threads"]):
                    if t["name"] == thread["name"]:
                        bible["plot_threads"][i].update(thread)
                        break
            else:
                bible["plot_threads"].append(thread)

        # Add chapter summary
        if "chapter_summary" in updates:
            bible.setdefault("chapter_summaries", []).append(updates["chapter_summary"])

        return bible

    @staticmethod
    def get_context_summary(bible: dict, ch_num: int, lang: str) -> str:
        """Format bible as compact text for cloud API context. (No API call)"""
        if not bible or not bible.get("characters"):
            return ""

        parts = []
        if lang == "zh":
            # Characters (compact)
            chars = []
            for name, info in bible.get("characters", {}).items():
                loc = info.get("location", "")
                emo = info.get("emotional_state", "")
                chars.append(f"{name}：{loc}，{emo}")
            if chars:
                parts.append("【人物状态】" + "；".join(chars))

            # Active plot threads
            active = [t for t in bible.get("plot_threads", []) if t.get("status") == "active"]
            if active:
                threads = "、".join(t["name"] for t in active[:5])
                parts.append(f"【活跃线索】{threads}")

            # Recent chapter summaries (last 3)
            summaries = bible.get("chapter_summaries", [])
            if summaries:
                recent = summaries[-3:]
                for s in recent:
                    ch = s.get("chapter", "?")
                    text = s.get("summary", "")
                    parts.append(f"【第{ch}章】{text}")
        else:
            chars = []
            for name, info in bible.get("characters", {}).items():
                loc = info.get("location", "")
                emo = info.get("emotional_state", "")
                chars.append(f"{name}: {loc}, {emo}")
            if chars:
                parts.append("[Characters] " + "; ".join(chars))

            active = [t for t in bible.get("plot_threads", []) if t.get("status") == "active"]
            if active:
                threads = ", ".join(t["name"] for t in active[:5])
                parts.append(f"[Active threads] {threads}")

            summaries = bible.get("chapter_summaries", [])
            if summaries:
                recent = summaries[-3:]
                for s in recent:
                    ch = s.get("chapter", "?")
                    text = s.get("summary", "")
                    parts.append(f"[Ch.{ch}] {text}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Mastermind — plans chapters and reviews output
# ---------------------------------------------------------------------------

class Mastermind:
    """Plans each chapter (converts outline to prose scene primers) and reviews
    generated text for quality. Also triggers story bible updates."""

    @staticmethod
    def _extract_chapter_outline(outline: str, ch_num: int) -> str:
        """Extract relevant context from the outline for a given chapter.

        If the outline has chapter-level sections (### Chapter N), extract that section.
        Otherwise (arc-based outline), return the full outline so the cloud API can
        determine what should happen in this chapter based on the story arcs.
        """
        # Try chapter-specific extraction first
        pattern = rf'(?:###?\s*(?:第{ch_num}章|Chapter\s*{ch_num})[：:\s]*)(.+?)(?=(?:###?\s*(?:第\d+章|Chapter\s*\d+))|$)'
        match = re.search(pattern, outline, re.DOTALL)
        if match:
            return match.group(0).strip()
        # Arc-based outline — return full outline (cloud API will interpret it)
        # Trim if excessively long to stay within API limits
        if len(outline) > 12000:
            return outline[:12000] + "\n\n[...outline truncated...]"
        return outline

    @staticmethod
    def plan_chapter(
        outline: str,
        ch_num: int,
        bible: dict,
        prev_ending: str,
        provider: str,
        api_key: str,
        lang: str,
    ) -> dict:
        """Plan a chapter: produce a prose scene primer + writing guidance. (1 API call)

        Returns dict with keys: scene_primer, key_events, guidance
        """
        chapter_outline = Mastermind._extract_chapter_outline(outline, ch_num)
        bible_summary = StoryTracker.get_context_summary(bible, ch_num, lang)

        if lang == "zh":
            system = (
                "你是一位资深小说编辑和连载小说策划。你的任务是根据故事总大纲和已有进展，"
                "为下一章制定详细的写作计划，并生成散文体的场景引子供AI写手使用。\n"
                "关键要求：\n"
                "1. 场景引子必须是纯散文，不能包含标题、列表或Markdown格式\n"
                "2. 必须确保与前文的连续性——人物位置、情感状态、已发生的事件\n"
                "3. 根据章节在全书中的位置，从大纲弧线中选择合适的情节推进\n"
                "4. 维持故事节奏——不要跳跃太快，也不要原地踏步"
            )
            prompt = f"""请为第{ch_num}章创建写作计划。

【故事大纲】
{chapter_outline}

【故事圣经——人物状态和剧情进展】
{bible_summary if bible_summary else "（首章，无历史记录）"}

【上一章结尾】
{prev_ending[-800:] if prev_ending else "（首章——从故事的起点开始）"}

【任务】
这是全书的第{ch_num}章。请根据大纲中的故事弧线，判断当前应该处于哪个阶段，
然后为本章制定具体的写作计划。确保：
- 与上一章的结尾自然衔接
- 人物的言行与故事圣经中记录的状态一致
- 情节推进符合大纲中对应弧线的方向
- 每章有明确的情节推进，不要重复之前已写过的内容

请输出JSON格式（必须合法JSON）：
```json
{{
  "scene_primer": "用散文体写一段场景引子（300-500字），描述本章开场的具体场景——时间、地点、环境细节、在场人物的状态和动作。必须与上一章结尾自然衔接。必须是纯叙事散文，不要使用任何标题、列表符号或Markdown格式。",
  "key_events": ["本章必须包含的关键事件1", "关键事件2", "关键事件3"],
  "guidance": "给AI写手的具体写作指导：本章的情感基调、节奏建议、需要重点刻画的人物互动、应避免的情节方向"
}}
```"""
        else:
            system = (
                "You are a senior fiction editor and serial novel planner. Your task is to take "
                "the story's master outline and existing progress, then create a detailed writing "
                "plan for the next chapter with a prose scene primer for the AI writer.\n"
                "Key requirements:\n"
                "1. The scene primer MUST be pure prose — no headings, bullet points, or markdown\n"
                "2. Ensure continuity with previous chapters — character locations, emotional states, events\n"
                "3. Based on the chapter's position in the overall story, select appropriate plot progression from the outline arcs\n"
                "4. Maintain story pacing — don't jump too fast or stall"
            )
            prompt = f"""Create a writing plan for Chapter {ch_num}.

[Story outline]
{chapter_outline}

[Story bible — character states and plot progress]
{bible_summary if bible_summary else "(First chapter, no history)"}

[End of previous chapter]
{prev_ending[-800:] if prev_ending else "(First chapter — start from the story's beginning)"}

[Task]
This is Chapter {ch_num} of the full novel. Based on the story arcs in the outline,
determine what stage the story should be at, then create a specific writing plan.
Ensure:
- Natural continuation from the previous chapter's ending
- Character behavior consistent with the story bible's recorded states
- Plot progression aligned with the corresponding arc in the outline
- Clear plot advancement — do NOT repeat content already written in previous chapters

Output as JSON (must be valid JSON):
```json
{{
  "scene_primer": "Write a prose scene primer (200-400 words) describing the chapter's opening scene — specific time, place, environmental details, character states and actions. Must naturally follow from the previous chapter's ending. Must be pure narrative prose — no headings, bullet points, or markdown.",
  "key_events": ["Key event 1 that must happen", "Key event 2", "Key event 3"],
  "guidance": "Specific writing guidance: emotional tone, pacing, character interactions to emphasize, plot directions to avoid"
}}
```"""

        try:
            response = call_cloud_api(system, prompt, provider, api_key, temperature=0.7)
            plan = _parse_json_response(response)
            # Ensure required keys exist
            plan.setdefault("scene_primer", chapter_outline)
            plan.setdefault("key_events", [])
            plan.setdefault("guidance", "")
            return plan
        except Exception as e:
            # Fallback: use chapter outline as-is
            return {
                "scene_primer": chapter_outline,
                "key_events": [],
                "guidance": f"(Planning failed: {e})",
            }

    @staticmethod
    def review_and_update(
        plan: dict,
        text: str,
        bible: dict,
        ch_num: int,
        provider: str,
        api_key: str,
        lang: str,
    ) -> dict:
        """Review generated chapter AND produce bible updates in one call. (1 API call)

        Returns dict with keys: review (approved, scores, issues, regen_guidance),
                                bible_updates (characters, plot_threads, chapter_summary)
        """
        key_events = ", ".join(plan.get("key_events", []))
        bible_summary = StoryTracker.get_context_summary(bible, ch_num, lang)

        if lang == "zh":
            system = (
                "你同时扮演两个角色：1）小说编辑——审核章节质量；"
                "2）故事记录员——更新故事圣经。请以JSON格式输出。"
            )
            prompt = f"""请审核以下生成的第{ch_num}章，并更新故事圣经。

【写作计划的关键事件】
{key_events}

【生成的章节文本】（前2000字）
{text[:3000]}

【当前故事圣经】
{bible_summary if bible_summary else "（空）"}

请输出JSON格式：
```json
{{
  "review": {{
    "approved": true,
    "scores": {{
      "plot_adherence": 8,
      "prose_quality": 7,
      "character_consistency": 8,
      "engagement": 7
    }},
    "issues": ["问题1（如有）"],
    "regen_guidance": "如果approved为false，给出修改建议"
  }},
  "bible_updates": {{
    "characters": {{
      "角色名": {{
        "location": "本章结束时的位置",
        "emotional_state": "本章结束时的情感状态"
      }}
    }},
    "plot_threads": [
      {{
        "name": "线索名称",
        "status": "active",
        "latest": "本章最新进展"
      }}
    ],
    "chapter_summary": {{
      "chapter": {ch_num},
      "summary": "本章概要（50-100字）"
    }}
  }}
}}
```"""
        else:
            system = (
                "You play two roles: 1) Fiction editor — review chapter quality; "
                "2) Story recorder — update the story bible. Output as JSON."
            )
            prompt = f"""Review the generated Chapter {ch_num} and update the story bible.

[Writing plan key events]
{key_events}

[Generated chapter text] (first 2000 words)
{text[:3000]}

[Current story bible]
{bible_summary if bible_summary else "(empty)"}

Output as JSON:
```json
{{
  "review": {{
    "approved": true,
    "scores": {{
      "plot_adherence": 8,
      "prose_quality": 7,
      "character_consistency": 8,
      "engagement": 7
    }},
    "issues": ["Issue 1 if any"],
    "regen_guidance": "Guidance for revision if approved is false"
  }},
  "bible_updates": {{
    "characters": {{
      "Character Name": {{
        "location": "Location at end of chapter",
        "emotional_state": "Emotional state at end of chapter"
      }}
    }},
    "plot_threads": [
      {{
        "name": "Thread name",
        "status": "active",
        "latest": "Latest development in this chapter"
      }}
    ],
    "chapter_summary": {{
      "chapter": {ch_num},
      "summary": "Chapter summary (50-100 words)"
    }}
  }}
}}
```"""

        try:
            response = call_cloud_api(system, prompt, provider, api_key, temperature=0.3)
            result = _parse_json_response(response)
            result.setdefault("review", {
                "approved": True,
                "scores": {},
                "issues": [],
                "regen_guidance": "",
            })
            result.setdefault("bible_updates", {})
            return result
        except Exception as e:
            return {
                "review": {
                    "approved": True,
                    "scores": {},
                    "issues": [f"Review failed: {e}"],
                    "regen_guidance": "",
                },
                "bible_updates": {
                    "chapter_summary": {
                        "chapter": ch_num,
                        "summary": text[:100] + "...",
                    }
                },
            }


# ---------------------------------------------------------------------------
# Prompt Builder — enforces token budget for local model
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Builds prompts for the local model that fit within the 4096 token
    training budget. Uses instruction + prose context format matching training."""

    @staticmethod
    def build_chapter_prompt(
        instruction: str,
        scene_primer: str,
        prev_text: str,
        max_new_tokens: int,
        lang: str,
    ) -> tuple:
        """Build a prompt within token budget.

        Returns (system_prompt, user_content, token_info_dict)
        """
        system = ZH_SYSTEM if lang == "zh" else EN_SYSTEM
        system_tokens = count_tokens(system)
        instruction_tokens = count_tokens(instruction)
        template_overhead = 15  # chat template special tokens

        # Sanitize primer: strip any markdown that the cloud API may have included
        scene_primer = re.sub(r'^#+\s.*$', '', scene_primer, flags=re.MULTILINE)
        scene_primer = re.sub(r'^\s*[-*]\s+', '', scene_primer, flags=re.MULTILINE)
        scene_primer = re.sub(r'\*\*([^*]+)\*\*', r'\1', scene_primer)
        scene_primer = scene_primer.strip()

        # Available budget for scene primer + narrative context
        available = (
            MAX_SEQ_LENGTH
            - system_tokens
            - instruction_tokens
            - template_overhead
            - max_new_tokens
        )
        available = max(available, 100)  # safety floor

        # Split: ~1/3 for primer, ~2/3 for narrative context
        primer_budget = min(count_tokens(scene_primer), available // 3)
        context_budget = available - primer_budget

        trimmed_primer = trim_to_token_budget(scene_primer, primer_budget, keep_end=False)
        trimmed_context = trim_to_token_budget(prev_text, context_budget, keep_end=True)

        # Build user content: instruction + prose (matches training format)
        parts = [instruction]
        if trimmed_primer:
            parts.append(trimmed_primer)
        if trimmed_context:
            parts.append(trimmed_context)
        user_content = "\n\n".join(parts)

        user_tokens = count_tokens(user_content)
        total = system_tokens + user_tokens + template_overhead + max_new_tokens

        return system, user_content, {
            "system_tokens": system_tokens,
            "user_tokens": user_tokens,
            "max_new_tokens": max_new_tokens,
            "total_estimated": total,
            "within_budget": total <= MAX_SEQ_LENGTH,
        }


# ---------------------------------------------------------------------------
# Multi-pass chapter generation (local model)
# ---------------------------------------------------------------------------

def generate_chapter_multipass(
    scene_primer: str,
    prev_text: str,
    num_passes: int,
    tokens_per_pass: int,
    lang: str,
    temperature: float,
    top_p: float,
    use_cloud: bool = False,
    cloud_provider: str = "",
    cloud_api_key: str = "",
) -> tuple:
    """Generate chapter text in multiple passes with sliding context.

    When use_cloud=True, uses cloud API instead of local model (no GPU needed).
    Returns (accumulated_text, pass_log_list)
    """
    instructions = _ZH_INSTRUCTIONS if lang == "zh" else _EN_INSTRUCTIONS
    builder = PromptBuilder()
    accumulated = ""
    pass_log = []

    for i in range(num_passes):
        instruction = random.choice(instructions)

        if i == 0:
            context = prev_text[-1200:] if prev_text else ""
            primer = scene_primer
        else:
            # Sliding context: use end of accumulated text
            context = accumulated[-1000:]
            primer = ""  # Only use primer for first pass

        system, user_content, token_info = builder.build_chapter_prompt(
            instruction, primer, context, tokens_per_pass, lang,
        )

        if use_cloud:
            result = generate_text_cloud(
                user_content,
                system_prompt=system,
                max_new_tokens=tokens_per_pass,
                temperature=temperature,
                provider=cloud_provider,
                api_key=cloud_api_key,
            )
        else:
            result = generate_text(
                user_content,
                system_prompt=system,
                max_new_tokens=tokens_per_pass,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.0,  # Match training (no penalty during training)
            )

        accumulated += result
        mode_label = "cloud" if use_cloud else "local"
        pass_log.append(
            f"Pass {i+1}/{num_passes} ({mode_label}): +{len(result)} chars "
            f"({token_info['total_estimated']}/{MAX_SEQ_LENGTH} tokens used)"
        )

    return accumulated, pass_log


# ---------------------------------------------------------------------------
# Legacy chapter generation (fallback when agents disabled)
# ---------------------------------------------------------------------------

def generate_chapter_legacy(
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
    """Generate a single chapter using the local fine-tuned model (legacy mode).

    This is the original generate_chapter — kept as fallback when agents are disabled.
    Note: this sends the full outline to the model, which is out-of-distribution
    for the training format. Use agent mode for better results.
    """
    if _model is None:
        return "Please load a local model first (Model Settings above)."

    lang = detect_language(plot_outline)
    progress(0.1, desc=f"Generating chapter {chapter_num}...")

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


# ---------------------------------------------------------------------------
# Full pipeline: agents + local model
# ---------------------------------------------------------------------------

def generate_chapter_pipeline(
    outline: str,
    ch_num: int,
    chapters_state: str,
    style_notes: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    enable_agents: bool,
    num_passes: int,
    api_provider: str,
    api_key: str,
    bible_state: dict,
    writer_mode: str = "Local Model",
    progress=gr.Progress(),
):
    """Full chapter generation pipeline.

    writer_mode: "Local Model" or "Cloud API"
    When agents enabled: Tracker -> Mastermind plan -> multi-pass gen -> review -> bible update
    When agents disabled: Legacy single-pass generation

    Returns (chapter_text, updated_all_chapters, updated_bible, generation_log, review_text)
    """
    use_cloud = writer_mode == "Cloud API"

    if not use_cloud and _model is None:
        msg = "Please load a local model first (Model Settings above), or switch Writer Mode to 'Cloud API'."
        return msg, chapters_state, bible_state, msg, ""

    if use_cloud and not api_key.strip():
        msg = "Cloud writer mode requires an API key. Please enter it in API Settings above."
        return msg, chapters_state, bible_state, msg, ""

    ch_num = int(ch_num)
    lang = detect_language(outline)

    # ---- Legacy mode (local model only, no agents) ----
    if not enable_agents:
        if use_cloud:
            msg = "Legacy mode requires a local model. Please enable agents for cloud-only writing."
            return msg, chapters_state, bible_state, msg, ""
        if not api_key.strip():
            pass  # Legacy mode doesn't need API key
        progress(0.1, desc=f"Generating chapter {ch_num} (legacy mode)...")
        result = generate_chapter_legacy(
            outline, ch_num, chapters_state, style_notes,
            max_tokens, temperature, top_p, repetition_penalty, progress,
        )
        sep = f"\n\n{'='*40}\n第{ch_num}章 / Chapter {ch_num}\n{'='*40}\n\n"
        new_acc = chapters_state + sep + result if chapters_state else result
        log = "Legacy mode (agents disabled).\n"
        log += f"Generated {len(result)} chars."
        return result, new_acc, bible_state, log, ""

    # Agents require API key
    if not api_key.strip():
        msg = "Agent mode requires an API key. Please enter it in API Settings above."
        return msg, chapters_state, bible_state, msg, ""

    # ---- Agent mode ----
    log_lines = []

    def log(msg):
        log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

    mode_label = "cloud" if use_cloud else "local"
    log(f"Writer mode: {mode_label}")

    try:
        # Step 1: Initialize bible if needed
        if not bible_state or not bible_state.get("characters"):
            log("Initializing story bible from outline...")
            progress(0.05, desc="Initializing story bible...")
            bible_state = StoryTracker.initialize_from_outline(
                outline, api_provider, api_key, lang,
            )
            n_chars = len(bible_state.get("characters", {}))
            n_threads = len(bible_state.get("plot_threads", []))
            log(f"Bible initialized: {n_chars} characters, {n_threads} plot threads")

        # Step 2: Mastermind plans the chapter
        log(f"Mastermind planning chapter {ch_num}...")
        progress(0.15, desc="Mastermind planning chapter...")
        prev_ending = chapters_state[-1200:] if chapters_state else ""
        plan = Mastermind.plan_chapter(
            outline, ch_num, bible_state, prev_ending,
            api_provider, api_key, lang,
        )
        primer_preview = plan["scene_primer"][:150].replace("\n", " ")
        log(f"Scene primer: {primer_preview}...")
        if plan["key_events"]:
            log(f"Key events: {', '.join(plan['key_events'][:3])}")
        if plan.get("guidance"):
            log(f"Guidance: {plan['guidance'][:100]}...")

        # Step 3: Multi-pass generation
        tokens_per_pass = min(max_tokens // num_passes, 1500)
        if tokens_per_pass < 512:
            log(f"Warning: tokens_per_pass ({tokens_per_pass}) below minimum 512, clamping up")
            tokens_per_pass = 512
        log(f"Generating chapter ({num_passes} passes, {tokens_per_pass} tokens/pass, {mode_label})...")
        progress(0.3, desc=f"Writing chapter ({num_passes} passes, {mode_label})...")

        try:
            result, pass_log = generate_chapter_multipass(
                plan["scene_primer"], prev_ending, num_passes, tokens_per_pass,
                lang, temperature, top_p,
                use_cloud=use_cloud, cloud_provider=api_provider, cloud_api_key=api_key,
            )
        except Exception as e:
            log(f"ERROR in chapter generation: {e}")
            error_msg = f"Chapter generation failed: {e}\nCheck API key and provider settings."
            return error_msg, chapters_state, bible_state, "\n".join(log_lines), ""

        for pl in pass_log:
            log(pl)
        log(f"Total generated: {len(result)} chars")

        if not result.strip():
            log("WARNING: Generation produced empty text")
            return "Generation produced empty text. Try again or check your API settings.", \
                chapters_state, bible_state, "\n".join(log_lines), ""

        # Step 4: Review + update bible (combined, 1 API call)
        log("Reviewing chapter + updating story bible...")
        progress(0.85, desc="Reviewing chapter...")
        review_result = Mastermind.review_and_update(
            plan, result, bible_state, ch_num,
            api_provider, api_key, lang,
        )

        review = review_result.get("review", {})
        bible_updates = review_result.get("bible_updates", {})

        # Apply bible updates
        bible_state = StoryTracker.apply_updates(bible_state, bible_updates)

        # Format review text
        approved = review.get("approved", True)
        scores = review.get("scores", {})
        issues = review.get("issues", [])

        status = "APPROVED" if approved else "NEEDS REVISION"
        log(f"Review: {status}")
        if scores:
            score_str = ", ".join(f"{k}: {v}/10" for k, v in scores.items())
            log(f"Scores: {score_str}")
        if issues:
            for issue in issues:
                log(f"Issue: {issue}")

        review_text = f"Status: {status}\n"
        if scores:
            review_text += "Scores:\n"
            for k, v in scores.items():
                review_text += f"  {k}: {v}/10\n"
        if issues:
            review_text += "Issues:\n"
            for issue in issues:
                review_text += f"  - {issue}\n"
        if not approved and review.get("regen_guidance"):
            review_text += f"\nRevision guidance: {review['regen_guidance']}\n"

        # Accumulate
        sep = f"\n\n{'='*40}\n第{ch_num}章 / Chapter {ch_num}\n{'='*40}\n\n"
        new_acc = chapters_state + sep + result if chapters_state else result

        progress(1.0, desc="Chapter complete!")
        return result, new_acc, bible_state, "\n".join(log_lines), review_text

    except Exception as e:
        log(f"PIPELINE ERROR: {e}")
        error_msg = f"Pipeline error: {e}\nSee Generation Log for details."
        return error_msg, chapters_state, bible_state, "\n".join(log_lines), ""


# ---------------------------------------------------------------------------
# Continue writing (direct, local model)
# ---------------------------------------------------------------------------

def continue_writing(
    existing_text: str,
    instruction: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    writer_mode: str = "Local Model",
    api_provider: str = "Gemini",
    api_key: str = "",
):
    """Continue writing from existing text using local model or cloud API."""
    use_cloud = writer_mode == "Cloud API"

    if not use_cloud and _model is None:
        return "Please load a local model first, or switch Writer Mode to 'Cloud API'."
    if use_cloud and not api_key.strip():
        return "Cloud writer mode requires an API key. Please enter it in API Settings."

    lang = detect_language(existing_text)
    if not instruction:
        instruction = "续写这段叙事，保持原文的风格和节奏。" if lang == "zh" else "Continue the narrative in the established style."

    prompt = instruction + "\n\n" + existing_text
    system = ZH_SYSTEM if lang == "zh" else EN_SYSTEM

    if use_cloud:
        return generate_text_cloud(
            prompt, system_prompt=system, max_new_tokens=max_tokens,
            temperature=temperature, provider=api_provider, api_key=api_key,
        )
    else:
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
    .gen-log { font-family: monospace; font-size: 0.85em; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Novel Writer Studio") as app:
        gr.Markdown("# Novel Writer Studio")
        gr.Markdown(
            "*Cloud AI (Gemini / GPT) develops your story outline and orchestrates agents. "
            "Chapters can be written by your fine-tuned local model OR entirely via cloud API (no GPU needed).*"
        )

        # ---- Shared state ----
        all_chapters = gr.State("")
        bible_state = gr.State({})

        # ---- API Settings ----
        with gr.Accordion("API Settings (for Plot & Agents)", open=True):
            gr.Markdown(
                "Enter your API key. Used for plot generation AND agent orchestration "
                "(Mastermind planning, story tracking, chapter review)."
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

        # ---- Writer Mode ----
        with gr.Accordion("Writer Mode", open=True):
            gr.Markdown(
                "**Local Model**: Uses your fine-tuned LoRA model (requires GPU). "
                "Best quality for style-specific prose.\n\n"
                "**Cloud API**: Uses Gemini/GPT for chapter writing too. "
                "No GPU needed — runs entirely on your Mac. "
                "Loses LoRA style but keeps all agent features (planning, review, story bible)."
            )
            writer_mode = gr.Radio(
                choices=["Local Model", "Cloud API"],
                value="Cloud API",
                label="Chapter Writer",
                info="Choose what writes the chapter prose",
            )

        # ---- Local Model Settings ----
        with gr.Accordion("Local Model Settings (optional for Cloud API mode)", open=False):
            gr.Markdown(
                "Load your fine-tuned LoRA model for chapter generation. "
                "**Only needed if Writer Mode is 'Local Model'.**"
            )
            # Hidden state to track adapter_info dict
            adapter_info_state = gr.State(adapter_info)

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

            def update_base_info(choice, info):
                return info.get(choice, "unknown") if info else "unknown"

            lora_dropdown.change(update_base_info, [lora_dropdown, adapter_info_state], base_model_info)
            load_btn.click(load_model_fn, [lora_dropdown, load_4bit], model_status)

            gr.Markdown("---")
            gr.Markdown("**Upload LoRA Adapter** — upload a zip or download from Hugging Face Hub")
            with gr.Row():
                with gr.Column(scale=1):
                    lora_zip_upload = gr.File(
                        label="Upload LoRA (.zip)",
                        file_types=[".zip"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("Upload & Extract", variant="secondary")
                with gr.Column(scale=1):
                    hf_repo_input = gr.Textbox(
                        label="Hugging Face Repo ID",
                        placeholder="username/qwen3_32b_novel_lora",
                    )
                    hf_download_btn = gr.Button("Download from HF Hub", variant="secondary")
            upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=3)
            refresh_adapters_btn = gr.Button("Refresh Adapter List", variant="secondary")

            def handle_zip_upload(zip_file):
                result = upload_lora_zip(zip_file)
                return result

            def handle_hf_download(repo_id, progress=gr.Progress()):
                return download_lora_hf(repo_id, progress)

            def handle_refresh():
                names, first, base, info = refresh_lora_list()
                return gr.update(choices=names, value=first), base, info, f"Found {len(info)} adapter(s)."

            upload_btn.click(handle_zip_upload, [lora_zip_upload], upload_status)
            hf_download_btn.click(handle_hf_download, [hf_repo_input], upload_status)
            refresh_adapters_btn.click(
                handle_refresh, [],
                [lora_dropdown, base_model_info, adapter_info_state, upload_status],
            )

        # ---- Agent Settings ----
        with gr.Accordion("Agent Settings", open=False):
            gr.Markdown(
                "**Multi-Agent System**: Cloud API agents (Mastermind + Story Tracker) "
                "orchestrate chapter generation for better plot adherence, continuity, and quality.\n\n"
                "- **Mastermind**: Converts outline to prose scene primers, reviews output\n"
                "- **Story Tracker**: Maintains character states, plot threads, chapter summaries\n"
                "- Uses 2-3 cloud API calls per chapter"
            )
            with gr.Row():
                enable_agents = gr.Checkbox(
                    value=True,
                    label="Enable Agents",
                    info="Use Mastermind + Story Tracker for chapter generation",
                )
                num_passes = gr.Slider(
                    1, 5, value=3, step=1,
                    label="Generation Passes",
                    info="More passes = longer chapters (each pass ~1000-1500 chars)",
                )

        # ---- Chapter Generation Settings ----
        with gr.Accordion("Chapter Generation Settings", open=False):
            with gr.Row():
                max_tokens_slider = gr.Slider(128, 4096, value=2048, step=128, label="Max New Tokens")
                temperature_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            with gr.Row():
                top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                rep_penalty_slider = gr.Slider(1.0, 2.0, value=1.0, step=0.05, label="Repetition Penalty",
                                               info="1.0 recommended (matches training). Higher values cause unnatural word avoidance.")

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
                        num_chapters_slider = gr.Slider(3, 200, value=30, step=1, label="Target Chapters",
                                                         info="Approximate story length — outline uses arcs, not individual chapters")
                        develop_btn = gr.Button(
                            "Develop Plot (Cloud AI)", variant="primary", size="lg",
                        )

                gr.Markdown("### Step 2: Plot Outline")
                gr.Markdown(
                    "Generated by Gemini/GPT as a high-level story arc structure (not chapter-by-chapter). "
                    "Review and edit freely — add detail, change plot points, extend arcs. "
                    "The Mastermind agent will plan each chapter from this outline."
                )
                plot_outline = gr.Textbox(
                    label="Plot Outline (editable)",
                    lines=25,
                    placeholder="Click 'Develop Plot' to generate an outline from your story idea...",
                    elem_classes=["plot-output"],
                )

                develop_btn.click(
                    develop_plot_api,
                    [story_idea, num_chapters_slider, api_provider, api_key_input],
                    plot_outline,
                )

                gr.Markdown("---")
                gr.Markdown("### Step 3: Generate Chapters")
                gr.Markdown(
                    "Writes each chapter using your chosen Writer Mode (local model or cloud API). "
                    "Enable agents for cloud-orchestrated multi-pass generation with "
                    "plot tracking and quality review."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        chapter_num = gr.Slider(1, 200, value=1, step=1, label="Chapter Number")
                        style_notes = gr.Textbox(
                            label="Style Notes (optional)",
                            placeholder="e.g., 多用对话 / focus on action / noir tone",
                            lines=2,
                        )
                        gen_chapter_btn = gr.Button(
                            "Generate Chapter", variant="primary", size="lg",
                        )
                    with gr.Column(scale=3):
                        chapter_output = gr.Textbox(
                            label="Generated Chapter",
                            lines=25,
                            elem_classes=["chapter-output"],
                        )

                # ---- Generation Log ----
                with gr.Accordion("Generation Log", open=False):
                    gen_log_display = gr.Textbox(
                        label="Agent Activity Log",
                        lines=12,
                        interactive=False,
                        elem_classes=["gen-log"],
                    )

                # ---- Review Results ----
                with gr.Accordion("Chapter Review", open=False):
                    review_display = gr.Textbox(
                        label="Mastermind Review",
                        lines=8,
                        interactive=False,
                    )

                gen_chapter_btn.click(
                    generate_chapter_pipeline,
                    [
                        plot_outline, chapter_num, all_chapters, style_notes,
                        max_tokens_slider, temperature_slider, top_p_slider, rep_penalty_slider,
                        enable_agents, num_passes, api_provider, api_key_input, bible_state,
                        writer_mode,
                    ],
                    [chapter_output, all_chapters, bible_state, gen_log_display, review_display],
                )

                # ---- All Chapters ----
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

            # ===== Tab 2: Story Bible =====
            with gr.Tab("Story Bible"):
                gr.Markdown(
                    "### Story Bible\n"
                    "The story bible tracks character states, plot threads, and chapter summaries "
                    "for continuity across chapters. It is automatically maintained by the Story Tracker agent."
                )
                bible_display = gr.JSON(label="Current Story Bible")
                with gr.Row():
                    refresh_bible_btn = gr.Button("Refresh Display")
                    init_bible_btn = gr.Button("Initialize Bible from Outline", variant="primary")
                    clear_bible_btn = gr.Button("Clear Bible", variant="stop")

                bible_manual_notes = gr.Textbox(
                    label="Manual Notes (added to style_notes)",
                    placeholder="Add your own notes here (e.g., 'the sword is named Frostbite')",
                    lines=3,
                )
                add_notes_btn = gr.Button("Add Notes to Bible")
                bible_status = gr.Textbox(label="Status", interactive=False, lines=1)

                def refresh_bible(bible):
                    return bible

                def init_bible_from_outline(outline, provider, key, bible):
                    if not outline.strip():
                        return bible, bible, "No outline to initialize from."
                    if not key.strip():
                        return bible, bible, f"Please enter your {provider} API key."
                    lang = detect_language(outline)
                    new_bible = StoryTracker.initialize_from_outline(outline, provider, key, lang)
                    n_chars = len(new_bible.get("characters", {}))
                    return new_bible, new_bible, f"Bible initialized: {n_chars} characters found."

                def clear_bible():
                    return StoryTracker.empty_bible()

                def add_manual_notes(bible, notes):
                    if not notes.strip():
                        return bible
                    bible = copy.deepcopy(bible)
                    existing = bible.get("style_notes", "")
                    bible["style_notes"] = (existing + "\n" + notes).strip() if existing else notes
                    return bible

                refresh_bible_btn.click(refresh_bible, bible_state, bible_display)
                init_bible_btn.click(
                    init_bible_from_outline,
                    [plot_outline, api_provider, api_key_input, bible_state],
                    [bible_state, bible_display, bible_status],
                )
                clear_bible_btn.click(clear_bible, outputs=bible_state)
                add_notes_btn.click(add_manual_notes, [bible_state, bible_manual_notes], bible_state)

            # ===== Tab 3: Free Write =====
            with gr.Tab("Free Write"):
                gr.Markdown("### Direct Generation")
                gr.Markdown("Write freely — provide context for continuation or start fresh. Uses your chosen Writer Mode.")

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
                     max_tokens_slider, temperature_slider, top_p_slider, rep_penalty_slider,
                     writer_mode, api_provider, api_key_input],
                    free_output,
                )

                def append_to_context(ctx, out):
                    return ctx + "\n" + out if ctx else out

                append_btn.click(append_to_context, [free_context, free_output], free_context)

            # ===== Tab 4: Quick Test =====
            with gr.Tab("Quick Test"):
                gr.Markdown("### Test with a single prompt")
                gr.Markdown("Uses your chosen Writer Mode (local model or cloud API).")
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

                def run_quick_test(prompt, sys_prompt, max_t, temp, tp, rp, wmode, provider, key):
                    if wmode == "Cloud API":
                        return generate_text_cloud(
                            prompt, system_prompt=sys_prompt, max_new_tokens=max_t,
                            temperature=temp, provider=provider, api_key=key,
                        )
                    return generate_text(prompt, sys_prompt, max_t, temp, tp, 50, rp)

                test_btn.click(
                    run_quick_test,
                    [test_prompt, test_system,
                     max_tokens_slider, temperature_slider, top_p_slider, rep_penalty_slider,
                     writer_mode, api_provider, api_key_input],
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
