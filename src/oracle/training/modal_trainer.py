"""Modal-based LoRA fine-tuning pipeline for Mistral 7B Instruct."""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("oracle-finetune")

# Volumes for persistent storage
checkpoint_volume = modal.Volume.from_name("oracle-checkpoints", create_if_missing=True)
dataset_volume = modal.Volume.from_name("oracle-dataset", create_if_missing=True)

# Container image with training dependencies
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.1.0",
    "transformers>=4.38.0",
    "peft>=0.9.0",
    "trl>=0.7.0",
    "datasets>=2.18.0",
    "bitsandbytes>=0.42.0",
    "accelerate>=0.27.0",
    "scipy>=1.12.0",
)

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CHECKPOINT_DIR = "/checkpoints/oracle-lora-v1"
DATASET_DIR = "/dataset"


def _format_example(example: dict) -> str:
    """Format a training example into Mistral instruct format."""
    return (
        f"<s>[INST] {example['instruction']}\n\n{example['input']} [/INST] "
        f"{example['output']}</s>"
    )


@app.function(
    gpu="A10G",
    timeout=7200,
    image=training_image,
    volumes={
        CHECKPOINT_DIR: checkpoint_volume,
        DATASET_DIR: dataset_volume,
    },
)
def train(
    dataset_path: str = "/dataset/training_data.jsonl",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
) -> dict:
    """Run LoRA fine-tuning on Mistral 7B Instruct.

    Args:
        dataset_path: Path to JSONL training data on the dataset volume.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: Learning rate for AdamW optimizer.
        max_seq_length: Maximum sequence length for tokenization.

    Returns:
        Training metrics dict.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTTrainer

    print(f"Loading dataset from {dataset_path}...")
    examples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} training examples")

    # Format examples for instruction tuning
    formatted = [{"text": _format_example(ex)} for ex in examples]
    dataset = Dataset.from_list(formatted)

    print(f"Loading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True,
    )
    model.config.use_cache = False

    # Apply LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=max_seq_length,
    )

    print("Starting training...")
    result = trainer.train()

    # Save adapter
    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)

    # Commit volume changes
    checkpoint_volume.commit()

    metrics = {
        "train_loss": result.training_loss,
        "train_runtime": result.metrics.get("train_runtime", 0),
        "train_samples_per_second": result.metrics.get("train_samples_per_second", 0),
        "trainable_params": trainable_params,
        "total_params": total_params,
        "epochs": epochs,
        "examples": len(examples),
        "checkpoint_dir": CHECKPOINT_DIR,
    }
    print(f"Training complete: {metrics}")
    return metrics


@app.local_entrypoint()
def main(
    dataset_path: str = "/dataset/training_data.jsonl",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
) -> None:
    """Local entrypoint for `modal run src/oracle/training/modal_trainer.py`."""
    print("Starting Oracle LoRA fine-tuning on Modal...")
    metrics = train.remote(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    print(f"Training complete. Metrics: {json.dumps(metrics, indent=2)}")
