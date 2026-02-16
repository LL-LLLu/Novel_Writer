import optuna
from pathlib import Path
from typing import Dict, Optional
import json

from loguru import logger

class HyperparameterTuner:
    """Automated hyperparameter optimization using Optuna."""

    def __init__(
        self,
        study_name: str = "novel-writer-tuning",
        storage: Optional[str] = None,
        direction: str = "minimize"
    ):
        """
        Args:
            study_name: Name of the Optuna study
            storage: Database URL for persistent storage (optional)
            direction: "minimize" or "maximize" objective
        """
        self.study_name = study_name
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True
        )

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for a trial."""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "lora_rank": trial.suggest_int("lora_rank", 8, 64),
            "lora_alpha": trial.suggest_int("lora_alpha", 8, 32),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.1),
            "batch_size": trial.suggest_categorical("batch_size", [1, 2, 4]),
            "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 2, 8),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.2),
            "max_seq_length": trial.suggest_categorical("max_seq_length", [4096, 6144, 8192]),
        }

    def objective(
        self,
        trial: optuna.Trial,
        train_func: callable,
        **kwargs
    ) -> float:
        """
        Objective function to minimize/maximize.

        Args:
            trial: Optuna trial
            train_func: Training function that takes hyperparameters
            **kwargs: Additional arguments for training

        Returns:
            Metric to optimize (e.g., validation loss)
        """
        hyperparams = self.suggest_hyperparameters(trial)
        hyperparams["_trial_number"] = trial.number

        logger.info(f"Trial {trial.number}: {hyperparams}")

        try:
            # Train with suggested hyperparameters
            metric = train_func(**hyperparams, **kwargs)
            return metric
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf') if self.study.direction == "minimize" else float('-inf')

    def optimize(
        self,
        train_func: callable,
        n_trials: int = 20,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        Run optimization.

        Args:
            train_func: Training function
            n_trials: Number of trials to run
            timeout: Timeout in seconds (optional)
            **kwargs: Arguments for training function
        """
        logger.info(f"Starting optimization: {n_trials} trials")

        self.study.optimize(
            lambda trial: self.objective(trial, train_func, **kwargs),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Log best parameters
        logger.success(f"Best trial: {self.study.best_trial.params}")
        logger.success(f"Best value: {self.study.best_value}")

        return self.study.best_trial

    def save_best_params(self, output_path: Path):
        """Save best hyperparameters to JSON."""
        best_params = self.study.best_trial.params

        with open(output_path, 'w') as f:
            json.dump({
                "best_params": best_params,
                "best_value": self.study.best_value,
                "trial_number": self.study.best_trial.number
            }, f, indent=2)

        logger.info(f"Saved best params to {output_path}")

def create_training_function(base_model_path: str, dataset_path: str):
    """Factory function to create training objective for Optuna."""

    def train_and_evaluate(**hyperparams):
        """Train model and return validation loss."""
        # Import here to avoid early imports
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset
        import torch

        trial_number = hyperparams.pop("_trial_number", 0)

        # Load model with hyperparameters
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_path,
            max_seq_length=hyperparams["max_seq_length"],
            load_in_4bit=True,
            dtype=None
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=hyperparams["lora_rank"],
            lora_alpha=hyperparams["lora_alpha"],
            lora_dropout=hyperparams["lora_dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        train_data = dataset.train_test_split(test_size=0.1)
        train_dataset = train_data["train"]
        eval_dataset = train_data["test"]

        # Format
        alpaca_prompt = """Below is an instruction...

### Instruction:
{}

### Input:
{}

### Response:
{}"""

        def format_func(examples):
            texts = []
            for i, o in zip(examples["instruction"], examples["output"]):
                text = alpaca_prompt.format(i, "", o) + tokenizer.eos_token
                texts.append(text)
            return {"text": texts}

        train_dataset = train_dataset.map(format_func, batched=True)
        eval_dataset = eval_dataset.map(format_func, batched=True)

        # Train
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            args=TrainingArguments(
                output_dir=f"optuna_trial_{trial_number}",
                num_train_epochs=1,  # Short for tuning
                per_device_train_batch_size=hyperparams["batch_size"],
                gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
                learning_rate=hyperparams["learning_rate"],
                weight_decay=hyperparams["weight_decay"],
                warmup_ratio=hyperparams["warmup_ratio"],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                eval_strategy="steps",
                eval_steps=50,
                save_strategy="no",  # Don't save during tuning
                report_to="none",
            ),
        )

        trainer.train()

        # Evaluate
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]

    return train_and_evaluate

def run_hyperparameter_tuning(
    base_model: str,
    dataset: str,
    output: str,
    n_trials: int = 20
):
    """
    Run complete hyperparameter tuning.

    Args:
        base_model: Base model path
        dataset: Dataset JSONL path
        output: Output path for best params
        n_trials: Number of trials
    """
    tuner = HyperparameterTuner(study_name="novel-writer-tuning")
    train_func = create_training_function(base_model, dataset)

    best_trial = tuner.optimize(
        train_func,
        n_trials=n_trials
    )

    tuner.save_best_params(Path(output))

    return best_trial
