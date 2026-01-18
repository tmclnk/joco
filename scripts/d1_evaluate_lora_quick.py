#!/usr/bin/env python3
"""
Quick evaluation on just 5 examples from each dataset to verify it works.
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


def generate_commit_message(model, tokenizer, diff_text: str, instruction: str) -> Tuple[str, float]:
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
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - start_time

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    response = response.split('\n')[0].strip()
    return response, inference_time


def main():
    print("="*80, flush=True)
    print("LoRA Model Quick Evaluation (5 examples per dataset)", flush=True)
    print("="*80, flush=True)

    project_root = Path(__file__).parent.parent
    base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    lora_path = project_root / "joco-lora-cpu"

    model, tokenizer = load_model_and_tokenizer(base_model_name, str(lora_path))
    print("Model loaded!\n", flush=True)

    # Test on 5 examples from validation.jsonl
    print("Testing on validation.jsonl (5 examples)...", flush=True)
    val_data = load_jsonl(project_root / "dataset" / "validation.jsonl")[:5]

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

    format_correct = 0
    type_correct = 0

    for i, example in enumerate(val_data):
        diff_text = example.get('input', '')
        expected = example.get('output', '').strip()
        expected_type = extract_commit_type(expected)

        predicted, inf_time = generate_commit_message(model, tokenizer, diff_text, instruction)
        predicted_type = extract_commit_type(predicted)
        is_valid = is_valid_conventional_commit(predicted)

        if is_valid:
            format_correct += 1
        if predicted_type == expected_type:
            type_correct += 1

        print(f"\nExample {i+1}:", flush=True)
        print(f"  Expected:  {expected}", flush=True)
        print(f"  Predicted: {predicted}", flush=True)
        print(f"  Valid format: {is_valid}, Type match: {predicted_type == expected_type}", flush=True)
        print(f"  Inference: {inf_time:.2f}s", flush=True)

    print(f"\nResults:", flush=True)
    print(f"  Format compliance: {format_correct}/5 ({format_correct/5*100:.0f}%)", flush=True)
    print(f"  Type accuracy: {type_correct}/5 ({type_correct/5*100:.0f}%)", flush=True)
    print("\nQuick test complete!", flush=True)


if __name__ == '__main__':
    main()
