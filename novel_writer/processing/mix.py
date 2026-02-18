from pathlib import Path
from typing import List, Optional, Dict
import json

from loguru import logger

class StyleMixer:
    """Merge multiple LoRA adapters for blended styles."""

    def __init__(self, base_model_path: str):
        """
        Args:
            base_model_path: HuggingFace model path
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.base_model_path = base_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )

    def load_lora(self, lora_path: Path):
        """Load a single LoRA adapter."""
        from peft import PeftModel

        logger.info(f"Loading LoRA: {lora_path}")
        return PeftModel.from_pretrained(
            self.base_model,
            lora_path
        )

    def merge_loras(
        self,
        lora_paths: List[Path],
        weights: List[float] = None
    ) -> PeftModel:
        """
        Merge multiple LoRA adapters with weights.

        Args:
            lora_paths: List of LoRA adapter paths
            weights: List of weights (sum to 1.0)

        Returns:
            Merged model
        """
        if not lora_paths:
            raise ValueError("No LoRA paths provided")

        if weights is None:
            weights = [1.0 / len(lora_paths)] * len(lora_paths)

        if len(weights) != len(lora_paths):
            raise ValueError("Weights must match number of LoRA paths")

        if abs(sum(weights) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")

        logger.info(f"Merging {len(lora_paths)} LoRAs with weights {weights}")

        # Load all LoRAs and collect their parameters
        lora_models = []
        for lora_path in lora_paths:
            lora_models.append(self.load_lora(lora_path))

        # Use first model as the target, blend all with proper weights
        merged = lora_models[0]

        for name, param in merged.named_parameters():
            if 'lora' in name:
                # Weighted sum across all adapters
                blended = param.data * weights[0]
                for model, weight in zip(lora_models[1:], weights[1:]):
                    other_param = model.get_parameter(name)
                    if other_param is not None:
                        blended = blended + other_param.data * weight
                param.data = blended

        return merged

    def create_style_config(
        self,
        loras: Dict[str, Path],
        output_path: Path
    ):
        """Create a style configuration file."""
        config = {
            "base_model": self.base_model_path,
            "styles": {name: str(path) for name, path in loras.items()}
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved style config to {output_path}")

def mix_styles(
    base_model: str,
    lora_paths: List[str],
    weights: List[float],
    output_path: str
):
    """
    Convenience function to mix styles.

    Args:
        base_model: Base model name/path
        lora_paths: Paths to LoRA adapters
        weights: Weights for each LoRA
        output_path: Where to save merged model
    """
    mixer = StyleMixer(base_model)
    paths = [Path(p) for p in lora_paths]

    merged = mixer.merge_loras(paths, weights)

    # Save merged model
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)

    merged.save_pretrained(output)
    mixer.tokenizer.save_pretrained(output)

    logger.success(f"Merged model saved to {output}")
