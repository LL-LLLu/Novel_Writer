import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from typing import Optional, Literal
import shutil

from loguru import logger

class ModelExporter:
    """Export models to various formats for deployment."""

    def __init__(
        self,
        base_model_path: str,
        lora_path: Optional[Path] = None
    ):
        """
        Args:
            base_model_path: HuggingFace model path
            lora_path: Path to LoRA adapters
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path

        # Load model
        logger.info(f"Loading model: {base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        if lora_path:
            logger.info(f"Loading LoRA: {lora_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_path
            )

    def merge_and_save(
        self,
        output_path: Path,
        safe_serialization: bool = True
    ):
        """
        Merge LoRA weights and save as full model.

        Args:
            output_path: Where to save merged model
            safe_serialization: Use safe serialization
        """
        logger.info("Merging LoRA weights...")

        if hasattr(self.model, 'merge_and_unload'):
            merged = self.model.merge_and_unload()
        else:
            # Already a base model
            merged = self.model

        output_path.mkdir(parents=True, exist_ok=True)

        merged.save_pretrained(
            output_path,
            safe_serialization=safe_serialization
        )
        self.tokenizer.save_pretrained(output_path)

        logger.success(f"Merged model saved to {output_path}")

        # Clean up cache
        if (output_path / "pytorch_model.bin").exists():
            (output_path / "pytorch_model.bin").unlink()

    def export_gguf(
        self,
        output_path: Path,
        quantization: Literal["q4_k_m", "q5_k_m", "q8_0"] = "q4_k_m"
    ):
        """
        Export to GGUF format for llama.cpp.

        Requires llama.cpp installation.

        Args:
            output_path: Output GGUF file path
            quantization: Quantization level
        """
        logger.info(f"Exporting to GGUF ({quantization})...")

        # First, merge to full model
        temp_path = output_path.parent / "temp_merged"
        self.merge_and_save(temp_path)

        # Use llama.cpp conversion script
        try:
            import subprocess

            cmd = [
                "convert-hf-to-gguf",
                str(temp_path),
                "--outfile",
                str(output_path),
                "--quantize",
                quantization
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(result.stdout)
            logger.success(f"GGUF model saved to {output_path}")

            # Cleanup temp
            shutil.rmtree(temp_path, ignore_errors=True)

        except FileNotFoundError:
            logger.error("llama.cpp not found. Install from: https://github.com/ggerganov/llama.cpp")
            raise
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            raise

    def export_vllm(
        self,
        output_path: Path,
        max_model_len: int = 8192
    ):
        """
        Export model optimized for vLLM.

        vLLM can load standard HF models directly.
        Just save in optimized format.

        Args:
            output_path: Output path
            max_model_len: Maximum sequence length
        """
        logger.info("Preparing model for vLLM...")

        output_path.mkdir(parents=True, exist_ok=True)

        # Save merged model (vLLM can load this)
        self.merge_and_save(output_path)

        # Create vLLM config
        config = {
            "model": str(output_path.absolute()),
            "max_model_len": max_model_len,
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1
        }

        import json
        with open(output_path / "vllm_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.success(f"vLLM model prepared at {output_path}")
        logger.info("Run with: vllm serve " + str(output_path))

    def export_onnx(
        self,
        output_path: Path,
        opset: int = 17
    ):
        """
        Export to ONNX format.

        Warning: Experimental for LLMs.

        Args:
            output_path: Output directory
            opset: ONNX opset version
        """
        logger.info(f"Exporting to ONNX (opset {opset})...")

        try:
            from optimum.onnxruntime import ORTModelForCausalLM

            output_path.mkdir(parents=True, exist_ok=True)

            # Convert to ONNX
            model = ORTModelForCausalLM.from_pretrained(
                self.base_model_path,
                export=True,
                opset=opset
            )

            model.save_pretrained(output_path)
            self.tokenizer.save_pretrained(output_path)

            logger.success(f"ONNX model saved to {output_path}")

        except ImportError:
            logger.error("optimum not installed. Run: pip install optimum[onnxruntime]")
            raise
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

def export_model(
    format: Literal["full", "gguf", "vllm", "onnx"],
    base_model: str,
    lora_path: Optional[str] = None,
    output: str = "exported_model",
    quantization: str = "q4_k_m"
):
    """
    Convenience function to export model.

    Args:
        format: Export format
        base_model: Base model path
        lora_path: LoRA adapter path
        output: Output directory/file
        quantization: Quantization for GGUF
    """
    exporter = ModelExporter(
        base_model_path=base_model,
        lora_path=Path(lora_path) if lora_path else None
    )

    output_path = Path(output)

    if format == "full":
        exporter.merge_and_save(output_path)
    elif format == "gguf":
        exporter.export_gguf(output_path, quantization)
    elif format == "vllm":
        exporter.export_vllm(output_path)
    elif format == "onnx":
        exporter.export_onnx(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")
