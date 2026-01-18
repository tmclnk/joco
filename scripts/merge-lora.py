#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and optionally convert to GGUF for Ollama.

Usage:
    # Merge only (creates HuggingFace model)
    python scripts/merge-lora.py --adapter joco-lora-cpu --output joco-merged

    # Merge and convert to GGUF
    python scripts/merge-lora.py --adapter joco-lora-cpu --output joco-merged --gguf

    # Full pipeline with Ollama model creation
    python scripts/merge-lora.py --adapter joco-lora-cpu --output joco-merged --gguf --ollama joco
"""

import argparse
import subprocess
import sys
from pathlib import Path


def merge_adapter(base_model: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model."""
    print(f"Loading base model: {base_model}")

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Load base model
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging adapter into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}/")
    return output_path


def find_llama_cpp():
    """Find llama.cpp installation (source or binary)."""
    # Check common locations
    search_paths = [
        Path.home() / "llama.cpp",
        Path.cwd() / "llama.cpp",
        Path.cwd() / "llama-b7765",  # Extracted binary release
    ]

    # Also check for extracted binary directories in cwd
    for p in Path.cwd().glob("llama-b*"):
        if p.is_dir():
            search_paths.insert(0, p)

    for path in search_paths:
        if path.exists():
            return path

    return None


def convert_to_gguf(model_path: str, output_file: str, quantize: str = None):
    """Convert HuggingFace model to GGUF format."""
    print(f"\nConverting to GGUF: {output_file}")

    llama_cpp_path = find_llama_cpp()

    # Check for conversion script (only in source repo, not binary releases)
    convert_script = None
    if llama_cpp_path:
        convert_script = llama_cpp_path / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            convert_script = None

    if not convert_script:
        print("\nllama.cpp conversion script not found.")
        print("The prebuilt binaries don't include convert_hf_to_gguf.py.")
        print("\nTo convert to GGUF, you need the source repo:")
        print("  git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        print("  cd ~/llama.cpp && pip install -r requirements.txt")
        print("  Re-run this script with --gguf")
        print(f"\nOr manually convert:")
        print(f"  python ~/llama.cpp/convert_hf_to_gguf.py {model_path} --outfile {output_file}")
        return None

    # Run conversion
    cmd = [sys.executable, str(convert_script), model_path, "--outfile", output_file]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Conversion failed: {result.stderr}")
        return None

    print(f"GGUF saved to: {output_file}")

    # Quantize if requested
    if quantize:
        # Look for llama-quantize in llama.cpp path or extracted binary dir
        quantize_bin = None
        if llama_cpp_path:
            for name in ["llama-quantize", "quantize"]:
                candidate = llama_cpp_path / name
                if candidate.exists():
                    quantize_bin = candidate
                    break

        if quantize_bin:
            quantized_file = output_file.replace(".gguf", f"-{quantize.lower()}.gguf")
            cmd = [str(quantize_bin), output_file, quantized_file, quantize]
            print(f"Quantizing to {quantize}: {quantized_file}")
            subprocess.run(cmd)
            return quantized_file
        else:
            print(f"llama-quantize not found.")
            print(f"If using prebuilt binaries, extract them first:")
            print(f"  tar -xzf llama-b*-bin-*.tar.gz")

    return output_file


def create_ollama_model(gguf_path: str, model_name: str):
    """Create an Ollama model from GGUF file."""
    print(f"\nCreating Ollama model: {model_name}")

    modelfile_path = Path(gguf_path).parent / "Modelfile"

    # Create Modelfile
    modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.7
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM You are a commit message generator. Given a git diff, output only a conventional commit message in the format: type(scope): description. Output nothing else.
"""

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"Created Modelfile at: {modelfile_path}")

    # Create Ollama model
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\nOllama model created! Test with:")
        print(f"  ollama run {model_name} 'Generate commit for: README.md updated'")
    else:
        print(f"Failed to create Ollama model: {result.stderr}")
        print(f"\nManually create with:")
        print(f"  ollama create {model_name} -f {modelfile_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and convert to GGUF")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        help="Base model name/path")
    parser.add_argument("--adapter", default="joco-lora-cpu",
                        help="Path to LoRA adapter")
    parser.add_argument("--output", default="joco-merged",
                        help="Output directory for merged model")
    parser.add_argument("--gguf", action="store_true",
                        help="Convert to GGUF format")
    parser.add_argument("--quantize", choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
                        help="Quantization level for GGUF")
    parser.add_argument("--ollama", metavar="NAME",
                        help="Create Ollama model with this name")
    args = parser.parse_args()

    # Step 1: Merge adapter
    output_path = merge_adapter(args.base_model, args.adapter, args.output)

    # Step 2: Convert to GGUF (optional)
    gguf_path = None
    if args.gguf or args.ollama:
        gguf_file = f"{args.output}.gguf"
        gguf_path = convert_to_gguf(output_path, gguf_file, args.quantize)

    # Step 3: Create Ollama model (optional)
    if args.ollama and gguf_path:
        create_ollama_model(gguf_path, args.ollama)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print(f"\nMerged model: {args.output}/")
    if gguf_path:
        print(f"GGUF file: {gguf_path}")
    if args.ollama:
        print(f"Ollama model: {args.ollama}")

    print("\nNext steps:")
    print("  # Test with Python:")
    print(f"  python -c \"from transformers import pipeline; p = pipeline('text-generation', '{args.output}'); print(p('Generate commit:'))\"")
    if not args.gguf:
        print("\n  # Convert to GGUF for Ollama:")
        print(f"  python scripts/merge-lora.py --adapter {args.adapter} --output {args.output} --gguf --ollama joco")


if __name__ == "__main__":
    main()
