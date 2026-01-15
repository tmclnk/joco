---
license: apache-2.0
task_categories:
  - text-generation
  - text2text-generation
language:
  - en
tags:
  - git
  - commits
  - conventional-commits
  - code
  - developer-tools
pretty_name: Joco Conventional Commits Dataset
size_categories:
  - n<1K
---

# Joco Conventional Commits Dataset

A curated dataset for finetuning language models to generate conventional commit messages from git diffs.

## Dataset Description

This dataset contains git diffs paired with high-quality conventional commit messages, curated from well-maintained open source repositories that follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Dataset Summary

- **Source**: Angular framework and related repositories
- **Format**: Instruction-tuning JSONL (compatible with Axolotl, Unsloth, LLaMA-Factory)
- **Size**: 42 examples (train) + 5 examples (validation)
- **Quality**: All examples pass structural validation for conventional commit format

### Supported Tasks

- **Commit Message Generation**: Generate a commit message from a git diff
- **Code Understanding**: Understand changes in code and summarize them appropriately

### Languages

The dataset is in English, with code samples from various programming languages.

## Dataset Structure

### Data Fields

Each record contains:

| Field | Type | Description |
|-------|------|-------------|
| `instruction` | string | System prompt describing the task |
| `input` | string | Git diff (from `git diff --staged` or `git show`) |
| `output` | string | Expected conventional commit message |
| `metadata` | object | Additional information (see below) |

#### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for the example |
| `repository` | string | Source repository name |
| `commit_hash` | string | Original commit SHA |
| `author` | string | Commit author |
| `type` | string | Commit type (feat, fix, docs, etc.) |
| `scope` | string | Commit scope (if present) |
| `quality_score` | int | Structural quality score (0-100) |

### Data Splits

| Split | Examples |
|-------|----------|
| train | 42 |
| validation | 5 |

### Example Record

```json
{
  "instruction": "Generate a conventional commit message for the following git diff.\n\nFollow the conventional commits format: type(scope): description\n\nValid types: feat, fix, docs, style, refactor, perf, test, build, ci, chore\n\nRules:\n- Keep the subject line under 72 characters\n- Start description with lowercase\n- Do not end with a period\n- Be concise and specific",
  "input": " .ng-dev/github.mjs | 1 +\n 1 file changed, 1 insertion(+)\n\ndiff --git a/.ng-dev/github.mjs b/.ng-dev/github.mjs\nindex 57b58572..8df9dcd8 100644\n--- a/.ng-dev/github.mjs\n+++ b/.ng-dev/github.mjs\n@@ -9,4 +9,5 @@ export const github = {\n   name: 'angular',\n   mainBranchName: 'main',\n   mergeMode: 'caretaker-only',\n+  requireReleaseModeForRelease: true,\n };",
  "output": "build: begin requiring usage of release mode for releases",
  "metadata": {
    "id": "angular-4d6a6aaf",
    "repository": "angular",
    "commit_hash": "4d6a6aafeecc0449025b3ad94c08223770af1b97",
    "author": "Joey Perrott <josephperrott@gmail.com>",
    "type": "build",
    "quality_score": 80
  }
}
```

## Conventional Commits Format

The dataset follows the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Valid Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Formatting, missing semicolons, etc. |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or correcting tests |
| `build` | Changes to build system or dependencies |
| `ci` | CI configuration changes |
| `chore` | Other changes that don't modify src or test files |

## Dataset Creation

### Curation Rationale

This dataset was created to finetune small language models (0.5B-3B parameters) for efficient commit message generation on resource-constrained hardware (8GB RAM).

### Source Data

The data is extracted from public git repositories that consistently follow the conventional commits specification:

- **Angular Framework**: Google's Angular project, known for strict commit message guidelines

### Quality Filtering

Records are filtered based on:

1. **Diff size**: 50-8000 characters (excludes trivial and overly complex changes)
2. **Structural validation**: Must match `type(scope): description` pattern
3. **Subject length**: Must be under 72 characters
4. **Quality score**: Minimum score of 70/100

### Deduplication

Records are deduplicated by commit message to ensure diverse examples.

## Usage

### With HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "validation": "validation.jsonl"
})

# Access a sample
print(dataset["train"][0])
```

### With Axolotl

```yaml
# axolotl config
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
datasets:
  - path: ./dataset/train.jsonl
    type: alpaca
    split: train
```

### With Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length=4096,
)

# Load dataset
from datasets import load_dataset
dataset = load_dataset("json", data_files="train.jsonl")
```

## Considerations for Using the Data

### Social Impact

This dataset enables developers to automate commit message generation, potentially improving code documentation practices and reducing cognitive load during development.

### Known Limitations

- **Small size**: Currently 47 examples total; larger datasets would improve model quality
- **Single source**: Primarily from Angular; may not generalize to all codebases
- **English only**: All commit messages are in English

### Recommendations

- Use for finetuning small models (0.5B-3B parameters)
- Combine with other conventional commit datasets for better coverage
- Consider data augmentation techniques to expand the dataset

## Citation

```bibtex
@misc{joco-dataset,
  title={Joco Conventional Commits Dataset},
  author={joco contributors},
  year={2024},
  url={https://github.com/tmclnk/joco}
}
```

## License

This dataset is released under the Apache 2.0 License.

## Dataset Card Contact

For questions or issues, please open an issue on the [joco repository](https://github.com/tmclnk/joco).
