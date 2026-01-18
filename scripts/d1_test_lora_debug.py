#!/usr/bin/env python3
"""
Debug version to see full output.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    print("Testing LoRA model - DEBUG VERSION")
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    lora_path = project_root / "joco-lora-cpu"

    # Load
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()
    print("Model loaded!")

    # Test
    instruction = (
        "Generate a conventional commit message for the following git diff.\n\n"
        "Follow the conventional commit format: type(scope): description\n\n"
        "Types: feat, fix, docs, style, refactor, test, chore, ci, build, perf"
    )

    diff = """diff --git a/README.md b/README.md
index 1234567..abcdefg 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,2 @@
 # My Project
+Add a simple test line for testing.
"""

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": diff}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    print(f"\nInput length: {inputs['input_ids'].shape[1]} tokens")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(f"Output length: {outputs.shape[1]} tokens")
    print(f"Generated tokens: {outputs.shape[1] - inputs['input_ids'].shape[1]}")

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)

    print("\n" + "="*80)
    print("FULL OUTPUT (with special tokens):")
    print("="*80)
    print(full_output)
    print("="*80)

    # Try different extraction methods
    print("\n" + "="*80)
    print("EXTRACTION ATTEMPTS:")
    print("="*80)

    # Method 1: skip special tokens
    clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n1. Skip special tokens:\n{clean}")

    # Method 2: after assistant tag
    if "<|im_start|>assistant" in full_output:
        after_assistant = full_output.split("<|im_start|>assistant")[-1]
        print(f"\n2. After assistant tag:\n{after_assistant}")

    # Method 3: just the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print(f"\n3. Just new tokens (with special):\n{new_text}")

    new_text_clean = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"\n4. Just new tokens (without special):\n{new_text_clean}")


if __name__ == '__main__':
    main()
