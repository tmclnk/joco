#!/usr/bin/env python3
"""
Validate the curated dataset and print statistics.

Usage: python3 scripts/validate-dataset.py
"""

import json
import re
from pathlib import Path
from collections import Counter

CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r'^(feat|fix|chore|docs|refactor|test|style|perf|build|ci)(\([\w.-]+\))?: .+'
)

def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_record(record: dict) -> list[str]:
    """Validate a single record and return list of issues."""
    issues = []

    # Check required fields
    for field in ['instruction', 'input', 'output']:
        if field not in record:
            issues.append(f"Missing required field: {field}")

    if 'output' in record:
        output = record['output']

        # Check conventional commit format
        if not CONVENTIONAL_COMMIT_PATTERN.match(output):
            issues.append(f"Output not in conventional commit format: {output[:50]}...")

        # Check length
        if len(output) > 72:
            issues.append(f"Output too long: {len(output)} chars")

        # Check for period at end
        if output.endswith('.'):
            issues.append("Output should not end with period")

    if 'input' in record:
        diff = record['input']

        # Check diff is non-empty
        if not diff.strip():
            issues.append("Empty diff")

        # Check diff looks like a git diff
        if 'diff --git' not in diff and '@@' not in diff:
            issues.append("Input doesn't look like a git diff")

    return issues


def print_statistics(records: list[dict], split_name: str):
    """Print statistics about a dataset split."""
    print(f"\n=== {split_name} Statistics ===")
    print(f"Total records: {len(records)}")

    if not records:
        return

    # Type distribution
    types = Counter()
    scopes = Counter()
    repos = Counter()

    diff_lengths = []
    output_lengths = []

    for record in records:
        if 'metadata' in record:
            meta = record['metadata']
            if 'type' in meta:
                types[meta['type']] += 1
            if 'scope' in meta:
                scopes[meta['scope']] += 1
            if 'repository' in meta:
                repos[meta['repository']] += 1

        if 'input' in record:
            diff_lengths.append(len(record['input']))
        if 'output' in record:
            output_lengths.append(len(record['output']))

    print(f"\nType distribution:")
    for t, count in sorted(types.items()):
        pct = count / len(records) * 100
        print(f"  {t}: {count} ({pct:.1f}%)")

    if repos:
        print(f"\nSource distribution:")
        for r, count in sorted(repos.items()):
            print(f"  {r}: {count}")

    if diff_lengths:
        print(f"\nDiff lengths:")
        print(f"  Min: {min(diff_lengths)} chars")
        print(f"  Max: {max(diff_lengths)} chars")
        print(f"  Avg: {sum(diff_lengths) // len(diff_lengths)} chars")

    if output_lengths:
        print(f"\nOutput lengths:")
        print(f"  Min: {min(output_lengths)} chars")
        print(f"  Max: {max(output_lengths)} chars")
        print(f"  Avg: {sum(output_lengths) // len(output_lengths)} chars")


def main():
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "dataset"

    print("=== Joco Dataset Validator ===")
    print(f"Dataset directory: {dataset_dir}")

    train_file = dataset_dir / "train.jsonl"
    validation_file = dataset_dir / "validation.jsonl"

    all_issues = []

    for split_file, split_name in [(train_file, "Train"), (validation_file, "Validation")]:
        if not split_file.exists():
            print(f"\n{split_name} file not found: {split_file}")
            continue

        records = load_jsonl(split_file)
        print_statistics(records, split_name)

        # Validate records
        issues_count = 0
        for i, record in enumerate(records):
            issues = validate_record(record)
            if issues:
                issues_count += 1
                all_issues.append((split_name, i, issues))

    print("\n=== Validation Results ===")
    if all_issues:
        print(f"Found issues in {len(all_issues)} records:")
        for split, idx, issues in all_issues[:10]:  # Show first 10
            print(f"  [{split}][{idx}]: {', '.join(issues)}")
        if len(all_issues) > 10:
            print(f"  ... and {len(all_issues) - 10} more")
    else:
        print("All records passed validation!")

    # Check for HuggingFace compatibility
    print("\n=== HuggingFace Compatibility ===")
    if train_file.exists():
        train_records = load_jsonl(train_file)
        if train_records:
            sample = train_records[0]
            required_fields = ['instruction', 'input', 'output']
            has_all = all(f in sample for f in required_fields)
            print(f"Instruction-tuning format: {'Yes' if has_all else 'No'}")
            print(f"Fields present: {list(sample.keys())}")

    print("\nDone!")


if __name__ == "__main__":
    main()
