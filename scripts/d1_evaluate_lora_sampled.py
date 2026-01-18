#!/usr/bin/env python3
"""
Evaluate the fine-tuned LoRA model on sampled datasets for faster results.

Samples:
- validation.jsonl: All 29 examples (small dataset)
- validation_extended.jsonl: 30 random samples from 208
- angular_benchmark: 30 random samples from 100

Total: ~89 examples for reasonable evaluation time.
"""

import json
import re
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


VALID_TYPES = {
    'feat', 'fix', 'docs', 'style', 'refactor', 'perf',
    'test', 'build', 'ci', 'chore'
}

CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r'^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)'
    r'(\([a-z0-9/_-]+\))?'
    r'(!)?'
    r': '
    r'.+'
)


def load_jsonl(file_path: Path) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_commit_type(commit_message: str) -> str:
    commit_message = commit_message.strip()
    match = re.match(r'^([a-z]+)(\([^)]+\))?:', commit_message)
    if match:
        return match.group(1)
    return 'unknown'


def is_valid_conventional_commit(message: str) -> bool:
    message = message.strip()
    return bool(CONVENTIONAL_COMMIT_PATTERN.match(message))


def load_model_and_tokenizer(base_model_name: str, lora_path: str):
    print(f"Loading base model: {base_model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter: {lora_path}", flush=True)
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
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": diff_text}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_length = inputs['input_ids'].shape[1]

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

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    response = response.split('\n')[0].strip()
    return response, inference_time


def evaluate_dataset(
    model,
    tokenizer,
    dataset: List[Dict],
    dataset_name: str,
    format_type: str = "training"
) -> Dict:
    print(f"\n{'='*80}", flush=True)
    print(f"Evaluating on: {dataset_name}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total examples: {len(dataset)}", flush=True)

    results = {
        'dataset': dataset_name,
        'total': len(dataset),
        'format_correct': 0,
        'type_correct': 0,
        'exact_match': 0,
        'predictions': [],
        'inference_times': [],
    }

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

    for i, example in enumerate(dataset):
        print(f"  Progress: {i+1}/{len(dataset)} (Avg: {sum(results['inference_times'])/len(results['inference_times']):.1f}s/example)" if results['inference_times'] else f"  Progress: {i+1}/{len(dataset)}", flush=True)

        if format_type == "training":
            diff_text = example.get('input', '')
            expected = example.get('output', '').strip()
        else:  # benchmark
            diff_text = example.get('diff', '')
            expected = example.get('expectedMessage', '').strip()

        try:
            predicted, inference_time = generate_commit_message(
                model, tokenizer, diff_text, instruction
            )
            results['inference_times'].append(inference_time)
        except Exception as e:
            print(f"  Error on example {i}: {e}", flush=True)
            predicted = ""
            results['inference_times'].append(0)

        expected_type = extract_commit_type(expected)
        predicted_type = extract_commit_type(predicted)
        is_valid_format = is_valid_conventional_commit(predicted)

        if is_valid_format:
            results['format_correct'] += 1
        if predicted_type == expected_type:
            results['type_correct'] += 1
        if predicted.strip().lower() == expected.strip().lower():
            results['exact_match'] += 1

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
    results['quality_score'] = (
        results['format_compliance'] * 0.4 +
        results['type_accuracy'] * 0.4 +
        results['exact_match_rate'] * 0.2
    ) * 100

    if results['inference_times']:
        results['avg_inference_time'] = sum(results['inference_times']) / len(results['inference_times'])
        results['total_inference_time'] = sum(results['inference_times'])
    else:
        results['avg_inference_time'] = 0
        results['total_inference_time'] = 0

    return results


def print_results(results: Dict):
    print(f"\n{'='*80}", flush=True)
    print(f"Results for: {results['dataset']}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"Total examples: {results['total']}", flush=True)
    print(f"Format compliance: {results['format_correct']}/{results['total']} ({results['format_compliance']*100:.1f}%)", flush=True)
    print(f"Type accuracy: {results['type_correct']}/{results['total']} ({results['type_accuracy']*100:.1f}%)", flush=True)
    print(f"Exact match: {results['exact_match']}/{results['total']} ({results['exact_match_rate']*100:.1f}%)", flush=True)
    print(f"\nQuality Score: {results['quality_score']:.1f}/100", flush=True)
    print(f"\nInference Speed:", flush=True)
    print(f"  Average time: {results['avg_inference_time']:.3f}s per example", flush=True)
    print(f"  Total time: {results['total_inference_time']:.1f}s ({results['total_inference_time']/60:.1f}min)", flush=True)
    print(flush=True)


def print_sample_predictions(results: Dict, n: int = 3):
    print(f"\nSample Predictions from {results['dataset']}:", flush=True)
    print("-" * 80, flush=True)

    for i, pred in enumerate(results['predictions'][:n]):
        print(f"\nExample {i+1}:", flush=True)
        print(f"  Expected:  {pred['expected']}", flush=True)
        print(f"  Predicted: {pred['predicted']}", flush=True)
        print(f"  Types: {pred['expected_type']} -> {pred['predicted_type']}", flush=True)
        print(f"  Valid: {pred['valid_format']}, Match: {pred['type_match']}", flush=True)


def save_results(results: Dict, output_path: Path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to: {output_path}", flush=True)


def main():
    print("="*80, flush=True)
    print("LoRA Model Evaluation (Sampled)", flush=True)
    print("="*80, flush=True)
    print(flush=True)

    random.seed(42)  # Reproducible sampling

    project_root = Path(__file__).parent.parent
    base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    lora_path = project_root / "joco-lora-cpu"
    dataset_dir = project_root / "dataset"
    benchmark_dir = project_root / "benchmark"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading model and LoRA adapter...", flush=True)
    model, tokenizer = load_model_and_tokenizer(base_model_name, str(lora_path))
    print("Model loaded successfully!", flush=True)

    datasets_to_eval = [
        {
            'name': 'validation',
            'path': dataset_dir / 'validation.jsonl',
            'format': 'training',
            'sample_size': None,  # Use all
        },
        {
            'name': 'validation_extended',
            'path': dataset_dir / 'validation_extended.jsonl',
            'format': 'training',
            'sample_size': 30,
        },
        {
            'name': 'angular_benchmark',
            'path': benchmark_dir / 'format-correctness' / 'angular-commits.jsonl',
            'format': 'benchmark',
            'sample_size': 30,
        },
    ]

    all_results = []
    for ds_config in datasets_to_eval:
        if not ds_config['path'].exists():
            print(f"Warning: Dataset not found: {ds_config['path']}", flush=True)
            continue

        dataset = load_jsonl(ds_config['path'])
        if ds_config['sample_size'] and len(dataset) > ds_config['sample_size']:
            dataset = random.sample(dataset, ds_config['sample_size'])
            print(f"Sampled {ds_config['sample_size']} examples from {len(load_jsonl(ds_config['path']))}", flush=True)

        results = evaluate_dataset(
            model,
            tokenizer,
            dataset,
            ds_config['name'],
            ds_config['format']
        )

        print_results(results)
        print_sample_predictions(results, n=3)

        output_file = results_dir / f"d1_lora_{ds_config['name']}_sampled.json"
        save_results(results, output_file)

        all_results.append(results)

    # Summary
    print("\n" + "="*80, flush=True)
    print("SUMMARY OF ALL EVALUATIONS", flush=True)
    print("="*80, flush=True)
    print(flush=True)
    print(f"{'Dataset':<30} {'Total':<8} {'Format':<10} {'Type Acc':<10} {'Quality':<10}", flush=True)
    print("-" * 78, flush=True)
    for r in all_results:
        print(f"{r['dataset']:<30} {r['total']:<8} "
              f"{r['format_compliance']*100:>6.1f}%   "
              f"{r['type_accuracy']*100:>6.1f}%   "
              f"{r['quality_score']:>6.1f}/100", flush=True)

    print(flush=True)
    print("="*80, flush=True)
    print("COMPARISON TO BASELINE", flush=True)
    print("="*80, flush=True)
    print(flush=True)
    print("Multi-step prompt baseline (from EXPERIMENT_LOG):", flush=True)
    print("  - Format compliance: 100% (8/8)", flush=True)
    print("  - Quality score: 88/100", flush=True)
    print(flush=True)

    angular_results = next((r for r in all_results if 'angular' in r['dataset'].lower()), None)
    if angular_results:
        print(f"LoRA model on Angular benchmark (30 samples):", flush=True)
        print(f"  - Format compliance: {angular_results['format_compliance']*100:.1f}%", flush=True)
        print(f"  - Type accuracy: {angular_results['type_accuracy']*100:.1f}%", flush=True)
        print(f"  - Quality score: {angular_results['quality_score']:.1f}/100", flush=True)
        print(flush=True)

        baseline_score = 88.0
        improvement = angular_results['quality_score'] - baseline_score
        if improvement > 0:
            print(f"LoRA model is {improvement:.1f} points BETTER than baseline", flush=True)
        else:
            print(f"LoRA model is {abs(improvement):.1f} points WORSE than baseline", flush=True)

    print(flush=True)
    print("="*80, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print("="*80, flush=True)


if __name__ == '__main__':
    main()
