# Finetuning Plan: Small Model for Commit Generation

## Goal
Create a ~600MB model achieving 90%+ format compliance for conventional commits.

## Current State
- Best small model: llama3.2:1b (60% format, 50% type accuracy)
- Dataset: 47 examples (too small)
- Bottleneck: Need more training data

---

## Step 1: Expand Dataset (Target: 300-500 examples)

### New Repositories to Add

| Language | Repo | Why |
|----------|------|-----|
| TypeScript | facebook/react | Large, well-maintained, good commits |
| TypeScript | vuejs/vue | Conventional commits |
| Go | kubernetes/kubernetes | Excellent commit hygiene |
| Go | docker/cli | Good commit messages |
| Python | fastapi/fastapi | Clean conventional commits |
| Python | django/django | Well-documented changes |
| Rust | rust-lang/rust | High-quality commits |
| Java | spring-projects/spring-boot | Enterprise patterns |

### Extraction Commands

```bash
# Clone repos (shallow)
cd tmp
git clone --depth=500 https://github.com/facebook/react.git
git clone --depth=500 https://github.com/vuejs/core.git
git clone --depth=500 https://github.com/kubernetes/kubernetes.git
git clone --depth=500 https://github.com/fastapi/fastapi.git
git clone --depth=500 https://github.com/rust-lang/rust.git

# Extract diffs (without commit messages)
./mvnw exec:java -Pharness -Dexec.args="extract tmp/react benchmark/raw/react.jsonl --max=100 --raw"
# ... repeat for each repo
```

### Label with Claude

```bash
# Use Claude to generate commit messages for raw diffs
./mvnw exec:java -Pharness -Dexec.args="label benchmark/raw/react.jsonl benchmark/labeled/react.jsonl --backend=claude"
```

### Merge into Training Set

```bash
./scripts/curate-dataset.sh
# Should produce 300-500 examples in dataset/train.jsonl
```

---

## Step 2: Training Setup

### Requirements
- 8GB RAM (tight but workable with Unsloth)
- ~10GB disk space
- Python 3.10+

### Install Dependencies

```bash
pip install unsloth transformers datasets trl peft bitsandbytes
```

### Training Script

Use the Unsloth script from `FINETUNING.md`:

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Load base model (4-bit for memory efficiency)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Load expanded dataset
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
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
        save_steps=50,
    ),
)

trainer.train()
model.save_pretrained("joco-lora")
```

---

## Step 3: Export to Ollama

### Merge LoRA and Quantize

```python
# Merge adapters and export to GGUF
model.save_pretrained_gguf("joco-gguf", tokenizer, quantization_method="q4_k_m")
```

### Create Ollama Modelfile

```dockerfile
FROM ./joco-gguf/unsloth.Q4_K_M.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "### Instruction:"
PARAMETER stop "### Input:"

SYSTEM You generate conventional commit messages from git diffs. Output only the commit message, nothing else.
```

### Import to Ollama

```bash
ollama create joco -f Modelfile
```

---

## Step 4: Benchmark

```bash
# Run harness with finetuned model
./mvnw exec:java -Pharness -Dexec.args="run --model=joco --cases=benchmark/format-correctness/angular-commits.jsonl --max=20"

# Compare with baseline
./mvnw exec:java -Pharness -Dexec.args="compare <baseline-run> <finetuned-run>"
```

---

## Expected Results

| Metric | Before (llama3.2:1b) | After (finetuned) |
|--------|---------------------|-------------------|
| Format Compliance | 60% | 90-95% |
| Type Accuracy | 50% | 80-85% |
| Score | 59.5 | 85+ |
| Model Size | 1.3GB | ~600MB |

---

## Timeline

| Phase | Time | Notes |
|-------|------|-------|
| Dataset expansion | 2-3 hours | Clone repos, extract, label with Claude |
| Training | 30-60 min | On MacBook with Unsloth |
| Export & Test | 30 min | Quantize, import to Ollama, benchmark |
| **Total** | **~4 hours** | |

---

## Checkpoints

- [ ] Clone additional repositories
- [ ] Extract raw diffs from each repo
- [ ] Label diffs with Claude backend
- [ ] Regenerate curated dataset (target: 300+ examples)
- [ ] Run finetuning with Unsloth
- [ ] Export to GGUF format
- [ ] Import to Ollama
- [ ] Run benchmark comparison
- [ ] Update EXPERIMENT_LOG.md with results
