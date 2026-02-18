"""
Test a LoRA fine-tuned model locally on Mac M-series.

Usage:
    python3 scripts/inference_mac.py --lora_path path/to/lora_adapter

    # With a different base model:
    python3 scripts/inference_mac.py --lora_path path/to/lora_adapter --base_model unsloth/Qwen3-8B

    # Merge LoRA into base model and save for faster future loads:
    python3 scripts/inference_mac.py --lora_path path/to/lora_adapter --merge --merge_output merged_model/

Requirements:
    pip install torch transformers peft accelerate sentencepiece protobuf
"""

import argparse
import sys
import torch
from pathlib import Path


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(base_model: str, lora_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model: {base_model}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map={"": device},
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    print("Model loaded successfully!\n")
    return model, tokenizer


def merge_and_save(model, tokenizer, output_path: str):
    print(f"Merging LoRA into base model...")
    merged = model.merge_and_unload()
    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Done! You can now load the merged model directly without --lora_path.")


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


def generate(model, tokenizer, prompt: str, context: str = "", max_new_tokens: int = 512, temperature: float = 0.8):
    # Detect language
    cjk_count = sum(1 for c in prompt[:200] if '\u4e00' <= c <= '\u9fff')
    is_zh = cjk_count > len(prompt[:200]) * 0.15
    system_prompt = ZH_SYSTEM if is_zh else EN_SYSTEM

    if context:
        instruction = "续写这段叙事，保持原文的风格和节奏。" if is_zh else "Continue the narrative in the established style."
        user_content = instruction + "\n\n" + context + "\n\n" + prompt
    else:
        user_content = prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            do_sample=True,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


def interactive_mode(model, tokenizer, max_new_tokens: int, temperature: float):
    print("=" * 60)
    print("Novel Writer - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /quit        - Exit")
    print("  /context      - Set context passage (for continuation)")
    print("  /clear        - Clear context")
    print("  /tokens N     - Set max new tokens (current: {})".format(max_new_tokens))
    print("  /temp N       - Set temperature (current: {})".format(temperature))
    print("=" * 60)
    print()

    context = ""

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt == "/quit":
            print("Goodbye!")
            break
        elif prompt == "/context":
            print("Paste your context passage (end with an empty line):")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                except EOFError:
                    break
            context = "\n".join(lines)
            print(f"Context set ({len(context)} chars)\n")
            continue
        elif prompt == "/clear":
            context = ""
            print("Context cleared.\n")
            continue
        elif prompt.startswith("/tokens "):
            try:
                max_new_tokens = int(prompt.split()[1])
                print(f"Max tokens set to {max_new_tokens}\n")
            except (ValueError, IndexError):
                print("Usage: /tokens N\n")
            continue
        elif prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}\n")
            except (ValueError, IndexError):
                print("Usage: /temp N\n")
            continue

        print("\nGenerating...\n")
        response = generate(model, tokenizer, prompt, context, max_new_tokens, temperature)
        print(f"Model>\n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="Test LoRA model on Mac M-series")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B",
                        help="Base model name or path (default: Qwen/Qwen3-8B)")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max new tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--merge", action="store_true",
                        help="Merge LoRA into base and save")
    parser.add_argument("--merge_output", type=str, default="merged_model",
                        help="Output path for merged model")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive mode)")
    args = parser.parse_args()

    device = get_device()
    model, tokenizer = load_model(args.base_model, args.lora_path, device)

    if args.merge:
        merge_and_save(model, tokenizer, args.merge_output)
        return

    if args.prompt:
        response = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature)
        print(response)
    else:
        interactive_mode(model, tokenizer, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
