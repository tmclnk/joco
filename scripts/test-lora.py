#!/usr/bin/env python3
"""
Test LoRA fine-tuned model against base model for commit message generation.

Compares format compliance and output quality between:
- Base Qwen2.5-Coder-0.5B-Instruct model
- Fine-tuned LoRA adapter

Works on CPU (no GPU required).
"""

import json
import argparse
import re
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_validation_examples(path: str, num_examples: int) -> list[dict]:
    """Load examples from validation JSONL file."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break
            examples.append(json.loads(line))
    return examples


def check_format_compliance(output: str) -> tuple[bool, str]:
    """
    Check if output follows conventional commit format.
    Valid formats:
    - type(scope): description
    - type: description

    Returns (is_valid, reason)
    """
    output = output.strip()

    # Extract first line only
    first_line = output.split('\n')[0].strip()

    # Valid commit types
    valid_types = ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'ci', 'build', 'perf']

    # Pattern with scope: type(scope): description
    pattern_with_scope = r'^([a-z]+)\(([a-z0-9_/-]+)\):\s+(.+)$'
    # Pattern without scope: type: description
    pattern_no_scope = r'^([a-z]+):\s+(.+)$'

    match_scope = re.match(pattern_with_scope, first_line, re.IGNORECASE)
    match_no_scope = re.match(pattern_no_scope, first_line, re.IGNORECASE)

    if match_scope:
        commit_type = match_scope.group(1).lower()
        if commit_type not in valid_types:
            return False, f"invalid type '{commit_type}'"
        return True, "valid (with scope)"

    if match_no_scope:
        commit_type = match_no_scope.group(1).lower()
        if commit_type not in valid_types:
            return False, f"invalid type '{commit_type}'"
        return True, "valid (no scope)"

    return False, "format mismatch"


def truncate_text(text: str, max_length: int = 60) -> str:
    """Truncate text with ellipsis if too long."""
    text = text.replace('\n', ' ').strip()
    if len(text) > max_length:
        return text[:max_length - 3] + "..."
    return text


def generate_commit_message(
    model,
    tokenizer,
    instruction: str,
    diff_input: str,
    max_new_tokens: int = 100
) -> str:
    """Generate a commit message using the model."""
    # Build user content
    user_content = instruction
    if diff_input:
        user_content += "\n\n" + diff_input

    messages = [
        {"role": "user", "content": user_content}
    ]

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic for comparison
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)

    return result.strip()


def print_comparison_table(results: list[dict], show_base: bool):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        print(f"\n--- Example {i} ---")
        print(f"Input (diff):     {truncate_text(r['input'], 70)}")
        print(f"Expected:         {truncate_text(r['expected'], 70)}")

        if show_base:
            base_status = "PASS" if r['base_format_ok'] else "FAIL"
            print(f"Base model:       {truncate_text(r['base_output'], 60)} [{base_status}]")

        ft_status = "PASS" if r['finetuned_format_ok'] else "FAIL"
        print(f"Fine-tuned:       {truncate_text(r['finetuned_output'], 60)} [{ft_status}]")

        # Show match status
        if r['finetuned_output'].strip().lower() == r['expected'].strip().lower():
            print("                  ^ Exact match with expected!")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total = len(results)
    ft_pass = sum(1 for r in results if r['finetuned_format_ok'])
    ft_exact = sum(1 for r in results if r['finetuned_output'].strip().lower() == r['expected'].strip().lower())

    print(f"\nFine-tuned model:")
    print(f"  Format compliance: {ft_pass}/{total} ({100*ft_pass/total:.1f}%)")
    print(f"  Exact matches:     {ft_exact}/{total} ({100*ft_exact/total:.1f}%)")

    if show_base:
        base_pass = sum(1 for r in results if r['base_format_ok'])
        base_exact = sum(1 for r in results if r['base_output'].strip().lower() == r['expected'].strip().lower())
        print(f"\nBase model:")
        print(f"  Format compliance: {base_pass}/{total} ({100*base_pass/total:.1f}%)")
        print(f"  Exact matches:     {base_exact}/{total} ({100*base_exact/total:.1f}%)")

        if ft_pass > base_pass:
            print(f"\n>>> Fine-tuning improved format compliance by {ft_pass - base_pass} examples!")
        elif ft_pass < base_pass:
            print(f"\n>>> Base model had better format compliance by {base_pass - ft_pass} examples")
        else:
            print(f"\n>>> Both models have same format compliance rate")


def main():
    parser = argparse.ArgumentParser(
        description="Test LoRA fine-tuned model for commit message generation"
    )
    parser.add_argument(
        "--adapter",
        default="joco-lora-cpu",
        help="Path to LoRA adapter directory (default: joco-lora-cpu)"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of validation examples to test (default: 5)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both base and fine-tuned models for comparison"
    )
    parser.add_argument(
        "--validation-file",
        default="dataset/validation.jsonl",
        help="Path to validation JSONL file"
    )
    args = parser.parse_args()

    # Force CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"

    print("=" * 60)
    print("LoRA Fine-tuning Test")
    print("=" * 60)
    print(f"Adapter path: {args.adapter}")
    print(f"Num examples: {args.num_examples}")
    print(f"Compare mode: {args.compare}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load adapter config to get base model
    adapter_config_path = Path(args.adapter) / "adapter_config.json"
    if not adapter_config_path.exists():
        print(f"Error: Adapter config not found at {adapter_config_path}")
        return 1

    with open(adapter_config_path) as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    print(f"\nBase model: {base_model_name}")

    # Load validation examples
    print(f"\nLoading {args.num_examples} validation examples...")
    examples = load_validation_examples(args.validation_file, args.num_examples)
    print(f"Loaded {len(examples)} examples")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []

    # Load and test base model if comparing
    if args.compare:
        print("\nLoading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map={"": device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        base_model.eval()

        print("Generating with base model...")
        for i, ex in enumerate(examples):
            print(f"  Example {i+1}/{len(examples)}...", end=" ", flush=True)
            output = generate_commit_message(
                base_model,
                tokenizer,
                ex["instruction"],
                ex.get("input", "")
            )
            is_valid, reason = check_format_compliance(output)
            print(f"[{reason}]")

            results.append({
                "input": ex.get("input", ""),
                "expected": ex.get("output", ""),
                "base_output": output,
                "base_format_ok": is_valid,
                "base_format_reason": reason,
            })

        # Free base model memory
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        # Initialize results without base model outputs
        for ex in examples:
            results.append({
                "input": ex.get("input", ""),
                "expected": ex.get("output", ""),
                "base_output": "",
                "base_format_ok": False,
                "base_format_reason": "not tested",
            })

    # Load fine-tuned model
    print("\nLoading fine-tuned model with LoRA adapter...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        device_map={"": device},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, args.adapter)
    finetuned_model.eval()

    print("Generating with fine-tuned model...")
    for i, ex in enumerate(examples):
        print(f"  Example {i+1}/{len(examples)}...", end=" ", flush=True)
        output = generate_commit_message(
            finetuned_model,
            tokenizer,
            ex["instruction"],
            ex.get("input", "")
        )
        is_valid, reason = check_format_compliance(output)
        print(f"[{reason}]")

        results[i]["finetuned_output"] = output
        results[i]["finetuned_format_ok"] = is_valid
        results[i]["finetuned_format_reason"] = reason

    # Print comparison table
    print_comparison_table(results, show_base=args.compare)

    return 0


if __name__ == "__main__":
    exit(main())
