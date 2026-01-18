#!/usr/bin/env python3
"""
Evaluate the fine-tuned LoRA model on multiple datasets.

This script evaluates the joco-lora-cpu LoRA adapter with Qwen2.5-Coder-0.5B-Instruct
base model on:
1. validation.jsonl (29 examples)
2. validation_extended.jsonl (208 examples)
3. Angular benchmark (100 examples)

Metrics measured:
- Format compliance (% valid conventional commits)
- Type accuracy (% correct type)
- Overall quality score (0-100)
- Inference speed (tokens/sec)
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Conventional commit types
VALID_TYPES = {
    'feat', 'fix', 'docs', 'style', 'refactor', 'perf',
    'test', 'build', 'ci', 'chore'
}

# Conventional commit pattern
CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r'^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)'
    r'(\([a-z0-9/_-]+\))?'
    r'(!)?'
    r': '
    r'.+'
)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL dataset file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_commit_type(commit_message: str) -> str:
    """Extract commit type from conventional commit message."""
    commit_message = commit_message.strip()
    match = re.match(r'^([a-z]+)(\([^)]+\))?:', commit_message)
    if match:
        return match.group(1)
    return 'unknown'


def is_valid_conventional_commit(message: str) -> bool:
    """Check if message follows conventional commit format."""
    message = message.strip()
    return bool(CONVENTIONAL_COMMIT_PATTERN.match(message))


def load_model_and_tokenizer(base_model_name: str, lora_path: str):
    """Load base model with LoRA adapter."""
    print(f"Loading base model: {base_model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,  # CPU inference
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    # Load LoRA adapter
    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    return model, tokenizer


def generate_commit_message(
    model,
    tokenizer,
    diff_text: str,
    instruction: str,
    max_new_tokens: int = 100,
    temperature: float = 0.3,
) -> Tuple[str, float]:
    """
    Generate commit message from diff using fine-tuned model.
    Returns (generated_message, inference_time_seconds).
    """
    # Format input using chat template
    messages = [
        {
            "role": "system",
            "content": instruction
        },
        {
            "role": "user",
            "content": diff_text
        }
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_length = inputs['input_ids'].shape[1]

    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - start_time

    # Extract only the newly generated tokens
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Clean up response
    response = response.strip()

    # Take only first line if multi-line
    response = response.split('\n')[0].strip()

    return response, inference_time


def evaluate_dataset(
    model,
    tokenizer,
    dataset: List[Dict],
    dataset_name: str,
    format_type: str = "training"
) -> Dict:
    """
    Evaluate model on a dataset.

    format_type: "training" for instruction/input/output format,
                 "benchmark" for diff/expectedMessage format
    """
    print(f"\n{'='*80}")
    print(f"Evaluating on: {dataset_name}")
    print(f"{'='*80}")
    print(f"Total examples: {len(dataset)}")

    results = {
        'dataset': dataset_name,
        'total': len(dataset),
        'format_correct': 0,
        'type_correct': 0,
        'exact_match': 0,
        'predictions': [],
        'inference_times': [],
    }

    for i, example in enumerate(dataset):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(dataset)}", flush=True)

        # Extract input and expected output based on format
        if format_type == "training":
            instruction = example.get('instruction', '')
            diff_text = example.get('input', '')
            expected = example.get('output', '').strip()
        else:  # benchmark format
            instruction = (
                "Generate a conventional commit message for the following git diff.\n\n"
                "Follow the conventional commit format: type(scope): description\n\n"
                "Types: feat, fix, docs, style, refactor, test, chore, ci, build, perf\n\n"
                "Rules:\n"
                "- Use lowercase for description\n"
                "- Use imperative mood (add, fix, update)\n"
                "- No period at end\n"
                "- Keep under 72 characters"
            )
            diff_text = example.get('diff', '')
            expected = example.get('expectedMessage', '').strip()

        # Generate prediction
        try:
            predicted, inference_time = generate_commit_message(
                model, tokenizer, diff_text, instruction
            )
            results['inference_times'].append(inference_time)
        except Exception as e:
            print(f"  Error on example {i}: {e}")
            predicted = ""
            results['inference_times'].append(0)

        # Extract types
        expected_type = extract_commit_type(expected)
        predicted_type = extract_commit_type(predicted)

        # Check format compliance
        is_valid_format = is_valid_conventional_commit(predicted)
        if is_valid_format:
            results['format_correct'] += 1

        # Check type accuracy
        if predicted_type == expected_type:
            results['type_correct'] += 1

        # Check exact match
        if predicted.strip().lower() == expected.strip().lower():
            results['exact_match'] += 1

        # Store prediction
        results['predictions'].append({
            'expected': expected,
            'predicted': predicted,
            'expected_type': expected_type,
            'predicted_type': predicted_type,
            'valid_format': is_valid_format,
            'type_match': predicted_type == expected_type,
            'inference_time': inference_time,
        })

    # Calculate metrics
    results['format_compliance'] = results['format_correct'] / results['total']
    results['type_accuracy'] = results['type_correct'] / results['total']
    results['exact_match_rate'] = results['exact_match'] / results['total']

    # Quality score (weighted combination)
    # Format compliance: 40%, Type accuracy: 40%, Exact match: 20%
    results['quality_score'] = (
        results['format_compliance'] * 0.4 +
        results['type_accuracy'] * 0.4 +
        results['exact_match_rate'] * 0.2
    ) * 100

    # Inference speed
    if results['inference_times']:
        results['avg_inference_time'] = sum(results['inference_times']) / len(results['inference_times'])
        results['total_inference_time'] = sum(results['inference_times'])
    else:
        results['avg_inference_time'] = 0
        results['total_inference_time'] = 0

    return results


def print_results(results: Dict):
    """Print evaluation results."""
    print(f"\n{'='*80}")
    print(f"Results for: {results['dataset']}")
    print(f"{'='*80}")
    print(f"Total examples: {results['total']}")
    print(f"Format compliance: {results['format_correct']}/{results['total']} ({results['format_compliance']*100:.1f}%)")
    print(f"Type accuracy: {results['type_correct']}/{results['total']} ({results['type_accuracy']*100:.1f}%)")
    print(f"Exact match: {results['exact_match']}/{results['total']} ({results['exact_match_rate']*100:.1f}%)")
    print(f"\nQuality Score: {results['quality_score']:.1f}/100")
    print(f"\nInference Speed:")
    print(f"  Average time: {results['avg_inference_time']:.3f}s per example")
    print(f"  Total time: {results['total_inference_time']:.1f}s")
    print()


def print_sample_predictions(results: Dict, n: int = 5):
    """Print sample predictions."""
    print(f"\nSample Predictions from {results['dataset']}:")
    print("-" * 80)

    for i, pred in enumerate(results['predictions'][:n]):
        print(f"\nExample {i+1}:")
        print(f"  Expected: {pred['expected']}")
        print(f"  Predicted: {pred['predicted']}")
        print(f"  Expected Type: {pred['expected_type']}")
        print(f"  Predicted Type: {pred['predicted_type']}")
        print(f"  Valid Format: {pred['valid_format']}")
        print(f"  Type Match: {pred['type_match']}")
        print(f"  Inference Time: {pred['inference_time']:.3f}s")


def save_results(results: Dict, output_path: Path):
    """Save detailed results to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_path}")


def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("LoRA Model Evaluation")
    print("="*80)
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    lora_path = project_root / "joco-lora-cpu"

    dataset_dir = project_root / "dataset"
    benchmark_dir = project_root / "benchmark"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load model
    print("Loading model and LoRA adapter...")
    model, tokenizer = load_model_and_tokenizer(base_model_name, str(lora_path))
    print("Model loaded successfully!")

    # Define datasets to evaluate
    datasets_to_eval = [
        {
            'name': 'validation.jsonl',
            'path': dataset_dir / 'validation.jsonl',
            'format': 'training',
        },
        {
            'name': 'validation_extended.jsonl',
            'path': dataset_dir / 'validation_extended.jsonl',
            'format': 'training',
        },
        {
            'name': 'angular_benchmark',
            'path': benchmark_dir / 'format-correctness' / 'angular-commits.jsonl',
            'format': 'benchmark',
        },
    ]

    # Evaluate on each dataset
    all_results = []
    for ds_config in datasets_to_eval:
        if not ds_config['path'].exists():
            print(f"Warning: Dataset not found: {ds_config['path']}")
            continue

        # Load dataset
        dataset = load_jsonl(ds_config['path'])

        # Evaluate
        results = evaluate_dataset(
            model,
            tokenizer,
            dataset,
            ds_config['name'],
            ds_config['format']
        )

        # Print results
        print_results(results)
        print_sample_predictions(results, n=5)

        # Save results
        output_file = results_dir / f"d1_lora_{ds_config['name'].replace('.jsonl', '')}.json"
        save_results(results, output_file)

        all_results.append(results)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL EVALUATIONS")
    print("="*80)
    print()
    print(f"{'Dataset':<30} {'Total':<8} {'Format':<10} {'Type Acc':<10} {'Quality':<10}")
    print("-" * 78)
    for r in all_results:
        print(f"{r['dataset']:<30} {r['total']:<8} "
              f"{r['format_compliance']*100:>6.1f}%   "
              f"{r['type_accuracy']*100:>6.1f}%   "
              f"{r['quality_score']:>6.1f}/100")

    print()
    print("="*80)
    print("COMPARISON TO BASELINE")
    print("="*80)
    print()
    print("Multi-step prompt baseline (from EXPERIMENT_LOG):")
    print("  - Format compliance: 100% (8/8)")
    print("  - Quality score: 88/100")
    print()

    # Find Angular results for comparison
    angular_results = next((r for r in all_results if 'angular' in r['dataset'].lower()), None)
    if angular_results:
        print(f"LoRA model on Angular benchmark:")
        print(f"  - Format compliance: {angular_results['format_compliance']*100:.1f}% "
              f"({angular_results['format_correct']}/{angular_results['total']})")
        print(f"  - Type accuracy: {angular_results['type_accuracy']*100:.1f}% "
              f"({angular_results['type_correct']}/{angular_results['total']})")
        print(f"  - Quality score: {angular_results['quality_score']:.1f}/100")
        print()

        # Calculate improvement
        baseline_score = 88.0
        improvement = angular_results['quality_score'] - baseline_score
        if improvement > 0:
            print(f"LoRA model is {improvement:.1f} points BETTER than baseline")
        else:
            print(f"LoRA model is {abs(improvement):.1f} points WORSE than baseline")

    print()
    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
