import json
from pathlib import Path
from typing import List, Dict
import re

from loguru import logger
from ..config import Config

class InstructionGenerator:
    """Generate instruction-response pairs from novel text."""

    INSTRUCTION_PROMPTS = [
        "Write a scene that continues this story: {summary}",
        "Describe what happens next in this narrative: {summary}",
        "Create a continuation maintaining this writing style: {summary}",
    ]

    def __init__(self, use_llm: bool = False, api_key: str = None):
        """
        Args:
            use_llm: Use LLM for generating instructions (requires API key)
            api_key: OpenAI/Anthropic API key (optional)
        """
        self.use_llm = use_llm
        self.api_key = api_key

        if use_llm and api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai not installed, using rule-based generation")
                self.use_llm = False

    def generate_summary(self, text: str, max_chars: int = 200) -> str:
        """Generate a simple summary of text chunk."""
        # Use first few sentences as summary
        sentences = re.split(r'[.!?]+', text)[:3]
        summary = '. '.join(sentences).strip()
        return summary[:max_chars]

    def generate_instructions_rule_based(
        self,
        text: str,
        num_instructions: int = 3
    ) -> List[Dict[str, str]]:
        """Generate instructions using rule-based templates."""
        summary = self.generate_summary(text)
        entries = []

        for i, template in enumerate(self.INSTRUCTION_PROMPTS[:num_instructions]):
            instruction = template.format(summary=summary)
            entries.append({
                "instruction": instruction,
                "input": "",
                "output": text
            })

        return entries

    def generate_instructions_with_llm(
        self,
        text: str,
        num_instructions: int = 3
    ) -> List[Dict[str, str]]:
        """Generate instructions using LLM."""
        summary = self.generate_summary(text)

        prompt = f"""You are a creative writing assistant. Given a summary of a novel excerpt, generate {num_instructions} diverse and creative writing instructions.

Summary: {summary}

Generate instructions like:
- "Write a scene where [character] [action]..."
- "Describe [setting] with [mood]..."
- "Continue the story where [plot point]..."

Return each instruction on a new line, prefixed with "Instruction: ": """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.8
            )

            instructions_text = response.choices[0].message.content
            instructions = []

            for line in instructions_text.split('\n'):
                if line.strip().lower().startswith('instruction:'):
                    instruction = line.replace('Instruction:', '').strip()
                    instructions.append({
                        "instruction": instruction,
                        "input": "",
                        "output": text
                    })

            return instructions[:num_instructions]

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self.generate_instructions_rule_based(text, num_instructions)

    def generate_from_chunk(
        self,
        text: str,
        num_instructions: int = 3
    ) -> List[Dict[str, str]]:
        """Generate instruction pairs from a text chunk."""
        if self.use_llm:
            return self.generate_instructions_with_llm(text, num_instructions)
        else:
            return self.generate_instructions_rule_based(text, num_instructions)

def generate_instruct_dataset(
    input_file: Path,
    output_file: Path,
    config: Config,
    num_instructions: int = 3
) -> int:
    """Generate instruction dataset from completion dataset."""
    generator = InstructionGenerator(
        use_llm=False,  # Set True with API key
        api_key=None
    )

    # Read completion dataset
    logger.info(f"Reading: {input_file}")
    entries = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get('output', '')
            entries.append((line, text))

    # Generate instructions
    logger.info(f"Generating instructions for {len(entries)} chunks...")
    instruct_entries = []

    from ..utils.progress import process_with_progress

    def process_func(item):
        line, text = item
        try:
            new_entries = generator.generate_from_chunk(text, num_instructions)
            instruct_entries.extend(new_entries)
            return len(new_entries)
        except Exception as e:
            logger.error(f"Failed to generate instructions: {e}")
            return 0

    process_with_progress(
        entries,
        process_func,
        description="Generating instructions...",
        total=len(entries)
    )

    # Write to file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in instruct_entries:
            json.dump(entry, f)
            f.write('\n')

    logger.success(f"Generated {len(instruct_entries)} instruction entries")

    return len(instruct_entries)
