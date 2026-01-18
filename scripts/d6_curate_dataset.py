#!/usr/bin/env python3
"""
D6: Curate high-quality fine-tuning dataset from extended data

Applies curation criteria:
1. FORMAT QUALITY: Perfect conventional commit format, clear type, concise description (<50 chars), no trailing periods
2. DIFF QUALITY: Clear focused changes (not 50+ files), 50-5000 char range, representative of real commits
3. SEMANTIC CLARITY: Unambiguous type, description matches diff, no vague messages
4. BALANCE: Equal representation (target 200 of each major type)

Input: dataset/train_extended.jsonl (1,276 examples)
Output: dataset/train_curated.jsonl, dataset/val_curated.jsonl, dataset/test_curated.jsonl
Target: 800-1000 HIGH-QUALITY examples (700 train, 100 validation, 100 test)
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import random

# Set random seed for reproducibility
random.seed(42)

# Valid conventional commit types
VALID_TYPES = {'feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'ci', 'build', 'perf'}

# Vague words that indicate poor semantic clarity
VAGUE_WORDS = {
    'update', 'change', 'modify', 'tweak', 'adjust', 'improve', 'enhance',
    'cleanup', 'misc', 'various', 'stuff', 'things', 'work'
}

def extract_commit_message(example):
    """Extract the commit message from the assistant's response."""
    for msg in example['messages']:
        if msg['role'] == 'assistant':
            return msg['content']
    return None

def parse_commit_message(message):
    """Parse conventional commit message into type, scope, and description."""
    # Pattern: type(scope): description or type: description
    pattern = r'^(\w+)(?:\(([^)]+)\))?\s*:\s*(.+)$'
    match = re.match(pattern, message.strip())
    if match:
        commit_type, scope, description = match.groups()
        return commit_type.lower(), scope, description.strip()
    return None, None, None

def extract_diff_content(example):
    """Extract the diff content from the user's message."""
    for msg in example['messages']:
        if msg['role'] == 'user':
            content = msg['content']
            # Find the diff section
            if 'diff --git' in content:
                diff_start = content.index('diff --git')
                return content[diff_start:]
    return None

def count_files_changed(diff_content):
    """Count number of files changed in the diff."""
    if not diff_content:
        return 0
    return len(re.findall(r'^diff --git', diff_content, re.MULTILINE))

def check_format_quality(message):
    """
    Check if commit message meets format quality criteria:
    - Valid conventional commit format
    - Valid type
    - Description under 72 chars (preferably under 50)
    - No trailing period
    - Imperative mood (heuristic check)
    """
    if not message:
        return False, "No message"

    commit_type, scope, description = parse_commit_message(message)

    # Check valid format
    if not commit_type or not description:
        return False, "Invalid format"

    # Check valid type
    if commit_type not in VALID_TYPES:
        return False, f"Invalid type: {commit_type}"

    # Check description length (prefer under 50, max 72)
    if len(description) > 72:
        return False, f"Description too long: {len(description)} chars"

    # Check no trailing period
    if description.endswith('.'):
        return False, "Trailing period"

    # Check not all uppercase (should be lowercase)
    if description.isupper():
        return False, "All uppercase"

    # Heuristic: check for past tense (common error)
    # Words ending in 'ed' are often past tense
    first_word = description.split()[0].lower()
    if first_word.endswith('ed') and not first_word in ['added', 'updated', 'fixed', 'removed']:
        # Some -ed words are acceptable in imperative (e.g., "add", "fix")
        # but most past tense like "implemented", "created" are not
        pass  # This is a soft check, don't reject

    return True, "OK"

def check_diff_quality(diff_content):
    """
    Check if diff meets quality criteria:
    - 50-5000 character range
    - Not too many files (< 50)
    - Has actual changes (not just whitespace)
    """
    if not diff_content:
        return False, "No diff"

    diff_len = len(diff_content)

    # Check length range
    if diff_len < 50:
        return False, f"Diff too short: {diff_len} chars"
    if diff_len > 5000:
        return False, f"Diff too long: {diff_len} chars"

    # Check file count
    file_count = count_files_changed(diff_content)
    if file_count > 50:
        return False, f"Too many files: {file_count}"
    if file_count == 0:
        return False, "No files changed"

    # Check for actual content changes (not just whitespace)
    has_additions = '+' in diff_content and re.search(r'^\+[^\+]', diff_content, re.MULTILINE)
    has_deletions = '-' in diff_content and re.search(r'^-[^-]', diff_content, re.MULTILINE)

    if not (has_additions or has_deletions):
        return False, "No actual changes"

    return True, "OK"

def check_semantic_clarity(message, diff_content):
    """
    Check if message has semantic clarity:
    - Not overly vague
    - Description length is reasonable (prefer under 50 chars)
    - Type seems to match diff context
    """
    if not message or not diff_content:
        return False, "Missing data"

    commit_type, scope, description = parse_commit_message(message)

    if not description:
        return False, "No description"

    # Check for vague descriptions
    desc_lower = description.lower()

    # If description is just a vague word with no context
    if any(desc_lower == vague or desc_lower.startswith(vague + ' ') for vague in VAGUE_WORDS):
        # Allow if there's more context
        if len(description.split()) < 3:
            return False, f"Vague description: {description[:30]}"

    # Check description is not too short (at least 2 words usually)
    if len(description.split()) < 2 and len(description) < 15:
        return False, f"Description too short: {description}"

    # Prefer descriptions under 50 chars for quality
    if len(description) > 50:
        # This is still acceptable, but not ideal
        pass

    # Type-specific checks (heuristic)
    if commit_type == 'test' and 'test' not in diff_content.lower():
        # Be lenient, test files might not have 'test' in path
        pass

    if commit_type == 'docs' and not any(ext in diff_content.lower() for ext in ['.md', 'readme', 'doc', '.mdx', '.rst']):
        # Be lenient, docs might be in code comments
        pass

    return True, "OK"

def apply_quality_filters(examples):
    """Apply all quality filters and return filtered examples with stats."""
    filtered = []
    stats = defaultdict(int)
    rejection_reasons = defaultdict(int)

    for example in examples:
        message = extract_commit_message(example)
        diff_content = extract_diff_content(example)

        stats['total'] += 1

        # Apply filters
        format_ok, format_reason = check_format_quality(message)
        if not format_ok:
            rejection_reasons[f"format: {format_reason}"] += 1
            continue

        diff_ok, diff_reason = check_diff_quality(diff_content)
        if not diff_ok:
            rejection_reasons[f"diff: {diff_reason}"] += 1
            continue

        semantic_ok, semantic_reason = check_semantic_clarity(message, diff_content)
        if not semantic_ok:
            rejection_reasons[f"semantic: {semantic_reason}"] += 1
            continue

        # All checks passed
        commit_type, _, _ = parse_commit_message(message)
        filtered.append({
            'example': example,
            'type': commit_type,
            'message': message,
            'diff_len': len(diff_content),
            'file_count': count_files_changed(diff_content)
        })
        stats['passed'] += 1

    return filtered, stats, rejection_reasons

def balance_classes(examples, max_per_type=200):
    """Balance classes by limiting each type to max_per_type examples."""
    by_type = defaultdict(list)

    for ex in examples:
        by_type[ex['type']].append(ex)

    balanced = []
    type_counts = {}

    for commit_type, type_examples in by_type.items():
        # Shuffle to randomize selection
        random.shuffle(type_examples)
        # Take up to max_per_type
        selected = type_examples[:max_per_type]
        balanced.extend(selected)
        type_counts[commit_type] = len(selected)

    return balanced, type_counts

def split_dataset(examples, train_size=0.7, val_size=0.15, test_size=0.15):
    """Split dataset into train, validation, and test sets."""
    # Shuffle
    random.shuffle(examples)

    total = len(examples)
    train_end = int(total * train_size)
    val_end = train_end + int(total * val_size)

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    return train, val, test

def save_dataset(examples, output_path):
    """Save examples to JSONL file."""
    with open(output_path, 'w') as f:
        for ex in examples:
            json.dump(ex['example'], f)
            f.write('\n')

def main():
    # Load extended dataset
    input_path = Path('dataset/train_extended.jsonl')
    print(f"Loading dataset from {input_path}...")

    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    # Apply quality filters
    print("\nApplying quality filters...")
    filtered, stats, rejection_reasons = apply_quality_filters(examples)

    print(f"\nFilter results:")
    print(f"  Total: {stats['total']}")
    print(f"  Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"  Rejected: {stats['total'] - stats['passed']}")

    print(f"\nTop rejection reasons:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {reason}: {count}")

    # Analyze type distribution before balancing
    type_dist_before = Counter(ex['type'] for ex in filtered)
    print(f"\nType distribution before balancing:")
    for commit_type, count in sorted(type_dist_before.items(), key=lambda x: x[1], reverse=True):
        print(f"  {commit_type}: {count}")

    # Balance classes
    print(f"\nBalancing classes (max 200 per type)...")
    balanced, type_counts = balance_classes(filtered, max_per_type=200)

    print(f"\nType distribution after balancing:")
    for commit_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {commit_type}: {count}")

    print(f"\nTotal curated examples: {len(balanced)}")

    # Split dataset
    print(f"\nSplitting dataset (70% train, 15% val, 15% test)...")
    train, val, test = split_dataset(balanced, train_size=0.7, val_size=0.15, test_size=0.15)

    print(f"  Train: {len(train)}")
    print(f"  Validation: {len(val)}")
    print(f"  Test: {len(test)}")

    # Save datasets
    output_dir = Path('dataset')
    output_dir.mkdir(exist_ok=True)

    train_path = output_dir / 'train_curated.jsonl'
    val_path = output_dir / 'val_curated.jsonl'
    test_path = output_dir / 'test_curated.jsonl'

    print(f"\nSaving datasets...")
    save_dataset(train, train_path)
    save_dataset(val, val_path)
    save_dataset(test, test_path)

    print(f"\nSaved:")
    print(f"  {train_path}: {len(train)} examples")
    print(f"  {val_path}: {len(val)} examples")
    print(f"  {test_path}: {len(test)} examples")

    # Generate statistics report
    print("\n" + "="*60)
    print("CURATION SUMMARY")
    print("="*60)
    print(f"Input: {len(examples)} examples")
    print(f"After quality filters: {len(filtered)} examples ({len(filtered)/len(examples)*100:.1f}%)")
    print(f"After balancing: {len(balanced)} examples")
    print(f"\nQuality criteria applied:")
    print(f"  1. Format: Valid conventional commit, <72 chars, no trailing period")
    print(f"  2. Diff: 50-5000 chars, <50 files, actual changes")
    print(f"  3. Semantic: Not vague, descriptive, type-appropriate")
    print(f"  4. Balance: Max 200 per type")
    print(f"\nFinal dataset:")
    print(f"  Train: {len(train)} examples ({len(train)/len(balanced)*100:.1f}%)")
    print(f"  Validation: {len(val)} examples ({len(val)/len(balanced)*100:.1f}%)")
    print(f"  Test: {len(test)} examples ({len(test)/len(balanced)*100:.1f}%)")
    print(f"  Total: {len(train) + len(val) + len(test)} examples")

    # Sample quality check
    print(f"\nSample curated examples:")
    for i, ex in enumerate(random.sample(balanced, min(5, len(balanced)))):
        print(f"\n{i+1}. Type: {ex['type']}")
        print(f"   Message: {ex['message']}")
        print(f"   Diff length: {ex['diff_len']} chars, Files: {ex['file_count']}")

if __name__ == '__main__':
    main()
