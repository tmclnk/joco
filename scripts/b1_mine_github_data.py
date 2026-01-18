#!/usr/bin/env python3
"""
B1: Mine additional conventional commit training data from GitHub

This script mines high-quality conventional commit examples from popular repositories
that are known to follow the conventional commits standard.

Target repos: Vue.js, Electron, Babel, ESLint, Jest, Webpack, TypeScript, Redux, Next.js
Current dataset: 290 examples (261 train, 29 val)
Target: 2000+ examples for better generalization

Quality filters:
- Diff length: 50-8000 characters
- Valid conventional commit format
- Balanced class distribution (feat, fix, docs, etc.)

Expected impact: 5-15% accuracy improvement from larger, more diverse dataset
"""

import json
import re
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import random


# Target repositories known to use conventional commits
TARGET_REPOS = [
    ("https://github.com/vuejs/vue.git", "vue", 200),
    ("https://github.com/electron/electron.git", "electron", 200),
    ("https://github.com/babel/babel.git", "babel", 200),
    ("https://github.com/eslint/eslint.git", "eslint", 200),
    ("https://github.com/jestjs/jest.git", "jest", 200),
    ("https://github.com/webpack/webpack.git", "webpack", 200),
    ("https://github.com/microsoft/TypeScript.git", "typescript", 200),
    ("https://github.com/reduxjs/redux.git", "redux", 150),
    ("https://github.com/vercel/next.js.git", "nextjs", 150),
]

# Conventional commit regex pattern
CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r'^(feat|fix|docs|test|ci|chore|refactor|perf|build|style)(\(.+\))?:\s+.+'
)

# Diff size constraints (in characters)
MIN_DIFF_LENGTH = 50
MAX_DIFF_LENGTH = 8000

# Dataset split ratio
VALIDATION_RATIO = 0.15

# Prompt template (same as existing dataset)
PROMPT_TEMPLATE = """Generate a conventional commit message for the following git diff.

Follow the conventional commit format: type(scope): description

Types: feat, fix, docs, style, refactor, test, chore, ci, build, perf

Rules:
- Use lowercase for description
- Use imperative mood (add, fix, update)
- No period at end
- Keep under 72 characters

"""


def is_conventional_commit(message: str) -> bool:
    """Check if commit message follows conventional commit format."""
    first_line = message.strip().split('\n')[0]
    return bool(CONVENTIONAL_COMMIT_PATTERN.match(first_line))


def extract_commit_type(message: str) -> Optional[str]:
    """Extract commit type from conventional commit message."""
    first_line = message.strip().split('\n')[0]
    match = re.match(r'^(\w+)(?:\([^)]+\))?:', first_line)
    if match:
        return match.group(1).lower()
    return None


def get_commit_diff(repo_path: Path, commit_hash: str) -> Optional[str]:
    """Get the diff for a specific commit."""
    try:
        result = subprocess.run(
            ["git", "show", "--format=", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Error getting diff for {commit_hash}: {e}")
        return None


def get_commit_message(repo_path: Path, commit_hash: str) -> Optional[str]:
    """Get the commit message for a specific commit."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%B", commit_hash],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  Error getting message for {commit_hash}: {e}")
        return None


def get_diff_stats(diff: str) -> str:
    """Extract diff stats (file changes summary) from full diff."""
    lines = diff.split('\n')
    stats_lines = []

    for line in lines:
        # Look for lines that show file changes (before the actual diff content)
        if line.startswith('diff --git'):
            # Extract just the file path
            parts = line.split()
            if len(parts) >= 4:
                file_path = parts[2].replace('a/', '')
                stats_lines.append(f" {file_path}")
        elif line.startswith('---') or line.startswith('+++'):
            continue
        elif line.startswith('@@'):
            break

    # If we have file stats, use them; otherwise create a summary
    if stats_lines:
        return '\n'.join(stats_lines[:10])  # Limit to 10 files
    else:
        # Fallback: just count changed lines
        added = sum(1 for l in lines if l.startswith('+') and not l.startswith('+++'))
        removed = sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))
        return f" {len(lines)} lines changed (+{added}, -{removed})"


def mine_repo_commits(
    repo_url: str,
    repo_name: str,
    target_count: int,
    temp_dir: Path
) -> List[Dict]:
    """Mine conventional commits from a repository."""
    print(f"\nMining {repo_name} from {repo_url}...")

    repo_path = temp_dir / repo_name
    examples = []

    # Clone repository (shallow clone to save time and space)
    print(f"  Cloning repository...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "2000", repo_url, str(repo_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Timeout cloning {repo_name}")
        return []
    except Exception as e:
        print(f"  ERROR: Failed to clone {repo_name}: {e}")
        return []

    if not repo_path.exists():
        print(f"  ERROR: Repository not cloned")
        return []

    # Get commit history
    print(f"  Fetching commit history...")
    try:
        result = subprocess.run(
            ["git", "log", "--format=%H", "--no-merges"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30
        )
        commit_hashes = result.stdout.strip().split('\n')
        print(f"  Found {len(commit_hashes)} commits")
    except Exception as e:
        print(f"  ERROR: Failed to get commit history: {e}")
        return []

    # Process commits
    processed = 0
    skipped_format = 0
    skipped_size = 0
    type_counts = Counter()

    for commit_hash in commit_hashes:
        if len(examples) >= target_count:
            break

        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed} commits, found {len(examples)} valid examples...")

        # Get commit message
        message = get_commit_message(repo_path, commit_hash)
        if not message or not is_conventional_commit(message):
            skipped_format += 1
            continue

        # Extract commit type for balancing
        commit_type = extract_commit_type(message)
        if not commit_type:
            skipped_format += 1
            continue

        # Get diff
        diff = get_commit_diff(repo_path, commit_hash)
        if not diff:
            skipped_size += 1
            continue

        # Check diff size
        diff_length = len(diff)
        if diff_length < MIN_DIFF_LENGTH or diff_length > MAX_DIFF_LENGTH:
            skipped_size += 1
            continue

        # Extract first line of commit message (the actual conventional commit)
        commit_first_line = message.split('\n')[0].strip()

        # Get diff stats for the prompt
        diff_stats = get_diff_stats(diff)

        # Create example in the same format as existing dataset
        example = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{PROMPT_TEMPLATE}{diff_stats}\n\n{diff}"
                },
                {
                    "role": "assistant",
                    "content": commit_first_line
                }
            ]
        }

        examples.append(example)
        type_counts[commit_type] += 1

    print(f"  Results: {len(examples)} examples from {processed} commits")
    print(f"  Skipped: {skipped_format} (format), {skipped_size} (size)")
    print(f"  Type distribution: {dict(type_counts)}")

    # Cleanup
    try:
        shutil.rmtree(repo_path)
    except Exception as e:
        print(f"  Warning: Failed to cleanup {repo_path}: {e}")

    return examples


def balance_dataset(examples: List[Dict], max_per_type: int = 300) -> List[Dict]:
    """Balance dataset by limiting examples per commit type."""
    type_buckets = defaultdict(list)

    # Group by commit type
    for example in examples:
        assistant_msg = example["messages"][1]["content"]
        commit_type = extract_commit_type(assistant_msg)
        if commit_type:
            type_buckets[commit_type].append(example)

    # Sample from each type
    balanced = []
    for commit_type, type_examples in type_buckets.items():
        if len(type_examples) > max_per_type:
            sampled = random.sample(type_examples, max_per_type)
            print(f"  {commit_type}: {len(type_examples)} -> {max_per_type} (sampled)")
        else:
            sampled = type_examples
            print(f"  {commit_type}: {len(type_examples)} (kept all)")
        balanced.extend(sampled)

    return balanced


def save_jsonl(data: List[Dict], file_path: Path):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} examples to {file_path}")


def load_existing_dataset(base_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Load existing train and validation datasets."""
    train_path = base_dir / "train.jsonl"
    val_path = base_dir / "validation.jsonl"

    train_data = []
    val_data = []

    if train_path.exists():
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))

    if val_path.exists():
        with open(val_path, 'r', encoding='utf-8') as f:
            for line in f:
                val_data.append(json.loads(line.strip()))

    return train_data, val_data


def print_statistics(data: List[Dict], name: str):
    """Print dataset statistics."""
    type_counts = Counter()

    for example in data:
        # Handle both new format (messages) and potential old formats
        if "messages" in example:
            assistant_msg = example["messages"][1]["content"]
        elif "output" in example:
            assistant_msg = example["output"]
        elif "response" in example:
            assistant_msg = example["response"]
        else:
            continue

        commit_type = extract_commit_type(assistant_msg)
        if commit_type:
            type_counts[commit_type] += 1

    print(f"\n{name} Statistics:")
    print(f"  Total examples: {len(data)}")
    print(f"  Type distribution:")
    for commit_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(data)) * 100
        print(f"    {commit_type}: {count} ({percentage:.1f}%)")


def main():
    """Main mining function."""
    print("=" * 80)
    print("B1: Mining GitHub Conventional Commit Data")
    print("=" * 80)

    # Setup paths
    base_dir = Path(__file__).parent.parent / "dataset"
    base_dir.mkdir(exist_ok=True)

    # Load existing dataset
    print("\nLoading existing dataset...")
    existing_train, existing_val = load_existing_dataset(base_dir)
    print(f"  Existing train: {len(existing_train)} examples")
    print(f"  Existing validation: {len(existing_val)} examples")

    # Mine new data
    all_examples = []
    repo_stats = {}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for repo_url, repo_name, target_count in TARGET_REPOS:
            examples = mine_repo_commits(repo_url, repo_name, target_count, temp_path)
            all_examples.extend(examples)
            repo_stats[repo_name] = {
                "count": len(examples),
                "target": target_count
            }

    print(f"\n{'=' * 80}")
    print(f"Total mined examples: {len(all_examples)}")
    print(f"{'=' * 80}")

    # Balance dataset
    print("\nBalancing dataset by commit type...")
    balanced_examples = balance_dataset(all_examples, max_per_type=300)

    # Shuffle
    random.shuffle(balanced_examples)

    # Split into train/validation
    val_size = int(len(balanced_examples) * VALIDATION_RATIO)
    new_val = balanced_examples[:val_size]
    new_train = balanced_examples[val_size:]

    print(f"\nNew data split:")
    print(f"  Train: {len(new_train)} examples")
    print(f"  Validation: {len(new_val)} examples")

    # Combine with existing data
    combined_train = existing_train + new_train
    combined_val = existing_val + new_val

    # Shuffle combined datasets
    random.shuffle(combined_train)
    random.shuffle(combined_val)

    # Save extended datasets
    train_extended_path = base_dir / "train_extended.jsonl"
    val_extended_path = base_dir / "validation_extended.jsonl"

    save_jsonl(combined_train, train_extended_path)
    save_jsonl(combined_val, val_extended_path)

    # Print final statistics
    print_statistics(combined_train, "Combined Training Set")
    print_statistics(combined_val, "Combined Validation Set")

    # Print repo statistics
    print(f"\n{'=' * 80}")
    print("Repository Mining Statistics:")
    print(f"{'=' * 80}")
    for repo_name, stats in repo_stats.items():
        print(f"  {repo_name}: {stats['count']}/{stats['target']} examples")

    print(f"\n{'=' * 80}")
    print("SUCCESS: Extended datasets created!")
    print(f"{'=' * 80}")
    print(f"  {train_extended_path}")
    print(f"  {val_extended_path}")
    print(f"\nOriginal dataset: {len(existing_train) + len(existing_val)} examples")
    print(f"Extended dataset: {len(combined_train) + len(combined_val)} examples")
    print(f"Increase: {len(combined_train) + len(combined_val) - len(existing_train) - len(existing_val)} examples")


if __name__ == "__main__":
    main()
