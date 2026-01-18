#!/usr/bin/env python3
"""
Finetune Qwen2.5-Coder for commit message generation on CPU.
Uses HuggingFace Transformers + PEFT (LoRA) with CPU optimizations.

Optimized for systems without GPU - expect slow training.
For 261 examples with 0.5B model: ~2-4 hours per epoch.
"""

import json
import argparse
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def load_dataset_jsonl(path: str) -> list[dict]:
    """Load JSONL dataset in chat format."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_chat_to_text(example: dict, tokenizer) -> str:
    """Convert chat messages or instruction format to training text."""
    # Handle chat format (messages array)
    messages = example.get("messages", [])

    # Handle instruction format (instruction/input/output)
    if not messages and "instruction" in example:
        user_content = example["instruction"]
        if example.get("input"):
            user_content += "\n\n" + example["input"]
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example.get("output", "")}
        ]

    if not messages:
        return ""

    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    # Fallback: simple format
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return text


def prepare_dataset(data: list[dict], tokenizer, max_length: int = 512) -> Dataset:
    """Prepare dataset for training."""

    def tokenize_function(examples):
        texts = [format_chat_to_text(ex, tokenizer) for ex in [examples]]

        # Tokenize
        tokenized = tokenizer(
            texts[0],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        # For causal LM, labels = input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Process each example
    processed = []
    for item in data:
        text = format_chat_to_text(item, tokenizer)
        if not text:
            continue

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        processed.append(tokenized)

    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser(description="CPU-based fine-tuning with LoRA")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="Base model (default: 0.5B for CPU training)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (keep small for CPU)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output-dir", default="joco-lora-cpu", help="Output directory")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("CPU Fine-tuning for Commit Message Generation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 60)

    # Force CPU
    device = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"\nUsing device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with CPU optimizations
    print("Loading model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,  # Full precision for CPU stability
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable training mode and ensure gradients flow
    model.train()
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()  # Ensure float32 for CPU training

    # Load datasets
    print("\nLoading datasets...")
    train_data = load_dataset_jsonl("dataset/train.jsonl")
    val_data = load_dataset_jsonl("dataset/validation.jsonl")
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Prepare datasets
    print("Tokenizing datasets...")
    train_dataset = prepare_dataset(train_data, tokenizer, args.max_length)
    val_dataset = prepare_dataset(val_data, tokenizer, args.max_length)
    print(f"Tokenized training: {len(train_dataset)}")
    print(f"Tokenized validation: {len(val_dataset)}")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        # CPU optimizations
        use_cpu=True,
        dataloader_num_workers=0,  # Avoid multiprocessing overhead
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,  # Disabled - conflicts with LoRA on CPU
        optim="adamw_torch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training... (this will be slow on CPU)")
    print("=" * 60 + "\n")

    if args.resume and Path(args.output_dir).exists():
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter saved to: {args.output_dir}/")
    print("=" * 60)
    print("\nTo test the model:")
    print(f"  python scripts/test-lora.py --adapter {args.output_dir}")


if __name__ == "__main__":
    main()
