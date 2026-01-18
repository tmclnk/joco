#!/usr/bin/env python3
"""
Quick test to verify LoRA model can load and generate a single prediction.
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    print("Testing LoRA model loading and inference...")
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    lora_path = project_root / "joco-lora-cpu"

    # Load tokenizer
    print(f"Loading tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print("Tokenizer loaded!")

    # Load base model
    print(f"Loading base model: {base_model_name}")
    start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"Base model loaded in {time.time()-start:.1f}s")

    # Load LoRA
    print(f"Loading LoRA adapter: {lora_path}")
    start = time.time()
    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.eval()
    print(f"LoRA adapter loaded in {time.time()-start:.1f}s")

    # Test with a simple example
    print("\nTesting inference...")
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
+A simple test project for testing commit messages.
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

    print("\nInput prompt (truncated):")
    print(text[:200] + "...")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    print("\nGenerating...")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - start

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract response
    if "<|im_start|>assistant" in generated_text:
        response = generated_text.split("<|im_start|>assistant")[-1].strip()
    else:
        response = generated_text[len(text):].strip()

    response = response.replace("<|im_end|>", "").strip()
    response = response.split('\n')[0].strip()

    print(f"\nGenerated commit message: {response}")
    print(f"Inference time: {inference_time:.3f}s")
    print("\nTest completed successfully!")


if __name__ == '__main__':
    main()
