import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import Optional

from .utils.logger import setup_logger

logger = setup_logger()

class NovelGenerator:
    """Generate novel text with fine-tuned model."""

    def __init__(
        self,
        base_model_path: str,
        lora_path: Optional[Path] = None,
        device: str = "cuda"
    ):
        """
        Args:
            base_model_path: HuggingFace model path
            lora_path: Path to LoRA adapters
            device: Device to run on
        """
        logger.info(f"Loading base model: {base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map=device
        )

        if lora_path:
            logger.info(f"Loading LoRA adapters: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path
            )

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """Generate text from prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from output
        return generated[len(prompt):]

    def generate_chapter(
        self,
        context: str,
        max_tokens: int = 2000
    ) -> str:
        """Generate a chapter continuation."""
        return self.generate(context, max_new_tokens=max_tokens)
