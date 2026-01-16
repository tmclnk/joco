# Finetuning Guide for Joco

This guide explains how to finetune a small language model on conventional commit message generation using the curated dataset.

## Dataset Overview

The dataset is located in `dataset/` and contains high-quality conventional commit examples extracted from well-known repositories.

### Current Statistics

| Metric | Value |
|--------|-------|
| Training examples | 42 |
| Validation examples | 5 |
| Source repositories | Angular |
| Average diff length | ~2,400 chars |
| Average output length | ~53 chars |

### Dataset Format

Each record uses the **instruction-tuning format**:

```json
{
  "instruction": "Generate a conventional commit message for the following git diff...",
  "input": "<git diff content>",
  "output": "refactor(http): remove redundant providedIn",
  "metadata": {
    "id": "angular-e38c1bf7",
    "repository": "angular",
    "commit_hash": "e38c1bf743be5f9dfa3cc4dc8e7dc87437cc821a",
    "author": "...",
    "type": "refactor",
    "scope": "http",
    "quality_score": 90
  }
}
```

## Regenerating the Dataset

### Add More Source Data

1. Extract commits from additional repositories:
   ```bash
   # Clone a reference repo
   git clone --depth=500 https://github.com/some/repo.git tmp/repo

   # Extract test cases
   ./mvnw exec:java -Pharness -Dexec.args="extract tmp/repo benchmark/format-correctness/repo-commits.jsonl --max=100"
   ```

2. Re-run the curator to include new data:
   ```bash
   ./scripts/curate-dataset.sh
   # or
   ./mvnw exec:java -Pdataset
   ```

### Quality Filters

The curator applies these filters:
- Diff length: 50-8,000 characters
- Validation score: 70+ out of 100
- Valid conventional commit format required
- Subject under 72 characters
- No trailing periods
- Deduplicated by commit message

## Finetuning Methods

### Option 1: Unsloth (Recommended for 8GB RAM)

[Unsloth](https://github.com/unslothai/unsloth) provides 2x faster finetuning with 60% less memory.

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load model (4-bit quantization for low memory)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Load dataset
dataset = load_dataset("json", data_files={
    "train": "dataset/train.jsonl",
    "validation": "dataset/validation.jsonl"
})

# Format for training
def format_prompt(example):
    return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    formatting_func=format_prompt,
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
    ),
)

trainer.train()

# Save LoRA adapters
model.save_pretrained("joco-lora")

# Merge and export to GGUF for Ollama
model.save_pretrained_gguf("joco-gguf", tokenizer, quantization_method="q4_k_m")
```

### Option 2: Axolotl

[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) is a configuration-driven finetuning tool.

Create `axolotl-config.yml`:

```yaml
base_model: Qwen/Qwen2.5-Coder-1.5B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

datasets:
  - path: dataset/train.jsonl
    type: alpaca

val_set_size: 0.1
sequence_len: 2048
sample_packing: true

gradient_accumulation_steps: 4
micro_batch_size: 1
num_epochs: 3
learning_rate: 2e-4
optimizer: adamw_bnb_8bit

output_dir: ./joco-finetuned
```

Run training:
```bash
accelerate launch -m axolotl.cli.train axolotl-config.yml
```

### Option 3: HuggingFace Trainer (Direct)

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Load base model
model_name = "Qwen/Qwen2.5-Coder-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# Add LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load and tokenize dataset
dataset = load_dataset("json", data_files={
    "train": "dataset/train.jsonl",
    "validation": "dataset/validation.jsonl"
})

def tokenize(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return tokenizer(prompt, truncation=True, max_length=2048)

tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# Train
trainer = Trainer(
    model=model,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    args=TrainingArguments(
        output_dir="./joco-finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_steps=50,
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("./joco-lora")
```

## Converting to Ollama

After finetuning, convert your model to GGUF format for use with Ollama:

### Using llama.cpp

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Convert to GGUF
python convert_hf_to_gguf.py ../joco-finetuned --outfile joco.gguf --outtype q4_k_m
```

### Create Ollama Model

Create a `Modelfile`:
```
FROM ./joco.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "### Instruction:"
PARAMETER stop "### Input:"

SYSTEM You generate conventional commit messages from git diffs. Output only the commit message, nothing else.
```

Import to Ollama:
```bash
ollama create joco -f Modelfile
```

## Recommended Models for 8GB RAM

| Model | Size | VRAM (4-bit) | Quality |
|-------|------|--------------|---------|
| Qwen2.5-Coder-0.5B | 0.5B | ~1GB | Basic |
| Qwen2.5-Coder-1.5B | 1.5B | ~2GB | Good |
| Qwen2.5-Coder-3B | 3B | ~3GB | Better |
| DeepSeek-Coder-1.3B | 1.3B | ~2GB | Good |

For 8GB RAM MacBook Air, use **Qwen2.5-Coder-1.5B** with 4-bit quantization.

## Benchmarking Your Finetuned Model

After finetuning and importing to Ollama:

```bash
# Run benchmark
./mvnw exec:java -Pharness -Dexec.args="run --model=joco --cases=benchmark/format-correctness/angular-commits.jsonl --max=20"

# Compare with baseline
./mvnw exec:java -Pharness -Dexec.args="compare <baseline-run-id> <finetuned-run-id>"
```

## Expanding the Dataset

To improve model quality, add more diverse examples:

1. **More repositories**: Extract from React, Vue, Rust, Go projects
2. **More authors**: Add notable developer commits from `benchmark/content-quality/`
3. **Balance types**: Current dataset is heavy on `docs` and `refactor`

```bash
# Extract from a new repo with conventional commits
./mvnw exec:java -Pharness -Dexec.args="extract /path/to/repo benchmark/format-correctness/new-repo.jsonl --max=100"

# Regenerate dataset
./scripts/curate-dataset.sh
```

## Troubleshooting

### Out of Memory

- Use 4-bit quantization (`load_in_4bit=True`)
- Reduce `max_seq_length` to 1024
- Reduce batch size to 1
- Increase `gradient_accumulation_steps`

### Poor Results

- Increase training epochs (3-5)
- Add more diverse training examples
- Check that prompt format matches inference format
- Ensure dataset quality filters aren't too strict

### Slow Training

- Use Unsloth for 2x speedup
- Enable Flash Attention if supported
- Use `sample_packing` in Axolotl

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Axolotl Documentation](https://github.com/OpenAccess-AI-Collective/axolotl)
- [PEFT (LoRA) Documentation](https://huggingface.co/docs/peft)
- [Conventional Commits Specification](https://www.conventionalcommits.org/)
- [Qwen2.5-Coder Models](https://huggingface.co/Qwen)
