#!/usr/bin/env python3
"""
Finetune a small model for conventional commit generation using MLX.
Optimized for Apple Silicon with limited RAM (8GB).
"""

import json
import argparse
from pathlib import Path

def load_dataset(train_path: str, val_path: str):
    """Load training and validation data."""
    train_data = []
    val_data = []

    with open(train_path) as f:
        for line in f:
            train_data.append(json.loads(line))

    with open(val_path) as f:
        for line in f:
            val_data.append(json.loads(line))

    return train_data, val_data

def format_example(example: dict) -> str:
    """Format example for training."""
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

def prepare_data(data: list, output_path: str):
    """Prepare data in MLX format."""
    formatted = []
    for ex in data:
        formatted.append({
            "text": format_example(ex)
        })

    with open(output_path, 'w') as f:
        for item in formatted:
            f.write(json.dumps(item) + '\n')

    print(f"Prepared {len(formatted)} examples to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare data and finetune with MLX')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare data, do not train')
    parser.add_argument('--model', default='mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit',
                        help='Base model to finetune')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    args = parser.parse_args()

    # Paths
    train_path = 'dataset/train.jsonl'
    val_path = 'dataset/validation.jsonl'
    mlx_train_path = 'dataset/mlx-train.jsonl'
    mlx_val_path = 'dataset/mlx-valid.jsonl'

    # Load and prepare data
    print("Loading dataset...")
    train_data, val_data = load_dataset(train_path, val_path)
    print(f"Loaded {len(train_data)} training, {len(val_data)} validation examples")

    print("\nPreparing MLX format...")
    prepare_data(train_data, mlx_train_path)
    prepare_data(val_data, mlx_val_path)

    if args.prepare_only:
        print("\nData prepared. Run finetuning with:")
        print(f"  mlx_lm.lora --model {args.model} \\")
        print(f"    --train --data dataset \\")
        print(f"    --iters 500 --batch-size {args.batch_size} \\")
        print(f"    --learning-rate {args.learning_rate}")
        return

    # Run finetuning
    print("\nStarting finetuning...")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")

    import subprocess
    import sys

    # Calculate iterations (examples * epochs / batch_size)
    iters = (len(train_data) * args.epochs) // args.batch_size

    cmd = [
        sys.executable, '-m', 'mlx_lm.lora',
        '--model', args.model,
        '--train',
        '--data', 'dataset',
        '--iters', str(iters),
        '--batch-size', str(args.batch_size),
        '--learning-rate', str(args.learning_rate),
        '--adapter-path', 'joco-lora',
        '--save-every', '100'
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd)

    print("\nFinetuning complete!")
    print("Adapter saved to: joco-lora/")
    print("\nTo test:")
    print("  python -m mlx_lm.generate --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit \\")
    print("    --adapter-path joco-lora --prompt 'Generate commit for: ...'")

if __name__ == '__main__':
    main()
