#!/usr/bin/env python3
"""
Experiment C2: Structural Feature Classifier for Commit Type Prediction

This script implements a classifier based on STRUCTURAL features extracted from
git diffs rather than text features. The hypothesis is that structural patterns
(file operations, line changes, hunk patterns) are more universal and repo-independent
than text-based features like TF-IDF.

HYPOTHESIS: Structural features generalize better than text features because:
- File operation patterns are universal (feat = new files, fix = small changes)
- Line change metrics are language-agnostic
- Hunk patterns capture intent (many small hunks = refactor, one large = feat)
- No vocabulary dependence (works across any repo/language)

Structural Features (15 total):
1. File operations: new_files, modified_files, deleted_files
2. Line changes: insertions, deletions, net_lines, insert_delete_ratio
3. Hunk analysis: hunk_count, avg_hunk_size, max_hunk_size, min_hunk_size
4. Derived: lines_per_file, hunks_per_file, fragmentation_score
5. File types: pct_code_files, pct_docs_files, pct_test_files

Expected Results:
- Overall accuracy: 65-70% (better than B2's 56.7%, B3's 59.6%)
- Feat/fix precision: >70% (vs current 37%)
- Better generalization (structural features are universal)
- Low overfitting (15 features << 261 examples)

Comparison to Previous:
- B2 (TF-IDF): 56.7% validation, 37% feat precision
- B3 (Enhanced TF-IDF): 59.6% validation, 36.8% feat precision
- Target: >65% overall, >70% feat/fix precision
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL dataset file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_commit_type(commit_message: str) -> str:
    """Extract commit type from conventional commit message."""
    # Pattern: type(scope): description or type: description
    match = re.match(r'^(\w+)(?:\([^)]+\))?:', commit_message.strip())
    if match:
        return match.group(1).lower()
    return 'unknown'


def extract_structural_features(diff_text: str) -> Dict[str, float]:
    """
    Extract 15 structural features from git diff.

    Features are designed to be universal and repo-independent:
    - File operations capture commit scope
    - Line changes capture commit magnitude
    - Hunk patterns capture commit fragmentation
    - File types capture commit domain

    Returns dict with 15 numeric features.
    """
    features = {}

    # Initialize counters
    new_files = 0
    modified_files = 0
    deleted_files = 0
    insertions = 0
    deletions = 0
    hunk_sizes = []
    file_paths = []

    current_file = None
    current_hunk_size = 0

    lines = diff_text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Track file operations from diff headers
        if line.startswith('diff --git'):
            # Save previous hunk size
            if current_hunk_size > 0:
                hunk_sizes.append(current_hunk_size)
                current_hunk_size = 0

            # Extract file path
            match = re.search(r'b/([^\s]+)', line)
            if match:
                current_file = match.group(1)
                file_paths.append(current_file)

        # Detect new file
        elif line.startswith('new file mode'):
            new_files += 1

        # Detect deleted file
        elif line.startswith('deleted file mode'):
            deleted_files += 1

        # Detect modified file (has both --- and +++ lines with non /dev/null)
        elif line.startswith('---'):
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if (next_line.startswith('+++') and
                    '/dev/null' not in line and
                    '/dev/null' not in next_line):
                    modified_files += 1

        # Track hunk headers to count hunks
        elif line.startswith('@@'):
            # Save previous hunk size
            if current_hunk_size > 0:
                hunk_sizes.append(current_hunk_size)
            current_hunk_size = 0

            # Parse hunk header: @@ -X,Y +A,B @@
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                # Extract old and new line counts
                old_count = int(match.group(2)) if match.group(2) else 1
                new_count = int(match.group(4)) if match.group(4) else 1
                # Hunk size is max of old and new counts
                current_hunk_size = max(old_count, new_count)

        # Count insertions and deletions
        elif line.startswith('+') and not line.startswith('+++'):
            insertions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1

        i += 1

    # Save final hunk
    if current_hunk_size > 0:
        hunk_sizes.append(current_hunk_size)

    # Calculate file operation features (1-3)
    features['new_files'] = new_files
    features['modified_files'] = modified_files
    features['deleted_files'] = deleted_files
    total_files = new_files + modified_files + deleted_files

    # Calculate line change features (4-7)
    features['insertions'] = insertions
    features['deletions'] = deletions
    features['net_lines'] = insertions - deletions
    # Ratio of insertions to deletions (handle div by zero)
    features['insert_delete_ratio'] = (
        insertions / deletions if deletions > 0 else insertions
    )

    # Calculate hunk analysis features (8-11)
    features['hunk_count'] = len(hunk_sizes)
    features['avg_hunk_size'] = np.mean(hunk_sizes) if hunk_sizes else 0
    features['max_hunk_size'] = max(hunk_sizes) if hunk_sizes else 0
    features['min_hunk_size'] = min(hunk_sizes) if hunk_sizes else 0

    # Calculate derived features (12-14)
    features['lines_per_file'] = (
        (insertions + deletions) / total_files if total_files > 0 else 0
    )
    features['hunks_per_file'] = (
        len(hunk_sizes) / total_files if total_files > 0 else 0
    )
    # Fragmentation: ratio of hunks to total lines changed
    total_changes = insertions + deletions
    features['fragmentation_score'] = (
        len(hunk_sizes) / total_changes if total_changes > 0 else 0
    )

    # Calculate file type features (15-17)
    # Code file extensions
    code_extensions = {'.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.c', '.cpp',
                      '.h', '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.cs',
                      '.scala', '.sh', '.bash', '.vue', '.html', '.css', '.scss'}
    # Documentation extensions
    doc_extensions = {'.md', '.txt', '.rst', '.adoc', '.tex'}
    # Test patterns
    test_patterns = ['test', 'spec', '__tests__', '.test.', '.spec.']

    code_files = 0
    doc_files = 0
    test_files = 0

    for path in file_paths:
        # Check extension
        path_lower = path.lower()
        ext = Path(path).suffix.lower()

        if ext in code_extensions:
            code_files += 1
        if ext in doc_extensions:
            doc_files += 1

        # Check for test patterns
        if any(pattern in path_lower for pattern in test_patterns):
            test_files += 1

    features['pct_code_files'] = code_files / total_files if total_files > 0 else 0
    features['pct_docs_files'] = doc_files / total_files if total_files > 0 else 0
    features['pct_test_files'] = test_files / total_files if total_files > 0 else 0

    return features


def extract_diff_from_user_message(user_message: str) -> str:
    """Extract the git diff portion from the user message."""
    # The diff starts after the rules section
    lines = user_message.split('\n')

    # Find where the diff starts (after the file summary line)
    diff_start = 0
    for i, line in enumerate(lines):
        # Look for lines like " file1.js | 10 ++++----"
        # or "diff --git" which marks start of actual diff
        if line.strip().startswith('diff --git'):
            diff_start = i
            break
        # Also try to find file stat lines (contain " | " with numbers)
        elif ' | ' in line and any(c.isdigit() for c in line):
            # The actual diff starts after these summary lines
            # Look for the "diff --git" line after this
            for j in range(i, len(lines)):
                if lines[j].strip().startswith('diff --git'):
                    diff_start = j
                    break
            if diff_start > 0:
                break

    if diff_start > 0:
        return '\n'.join(lines[diff_start:])

    # Fallback: return everything if we can't find the start
    return user_message


def prepare_data(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and labels y from dataset.

    Handles two formats:
    1. Messages format: {'messages': [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]}
    2. Instruction format: {'instruction': ..., 'input': ..., 'output': ...}

    Returns:
        X: numpy array of shape (n_samples, 15) with structural features
        y: numpy array of shape (n_samples,) with commit type labels
    """
    X = []
    y = []

    for item in data:
        # Handle messages format (train.jsonl)
        if 'messages' in item:
            messages = item.get('messages', [])
            if len(messages) < 2:
                continue

            # Extract diff from user message
            user_message = messages[0]['content']
            diff_text = extract_diff_from_user_message(user_message)

            # Extract commit type from assistant message
            assistant_message = messages[1]['content']
            commit_type = extract_commit_type(assistant_message)

        # Handle instruction format (validation.jsonl)
        elif 'input' in item and 'output' in item:
            # input contains the diff
            diff_text = item['input']

            # output contains the commit message
            commit_message = item['output']
            commit_type = extract_commit_type(commit_message)

        else:
            continue

        if commit_type == 'unknown':
            continue

        # Extract structural features
        features = extract_structural_features(diff_text)

        # Convert to feature vector (ensure consistent ordering)
        feature_vector = [
            features['new_files'],
            features['modified_files'],
            features['deleted_files'],
            features['insertions'],
            features['deletions'],
            features['net_lines'],
            features['insert_delete_ratio'],
            features['hunk_count'],
            features['avg_hunk_size'],
            features['max_hunk_size'],
            features['min_hunk_size'],
            features['lines_per_file'],
            features['hunks_per_file'],
            features['fragmentation_score'],
            features['pct_code_files'],
            # features['pct_docs_files'],  # Removed to keep exactly 15 features
            # features['pct_test_files'],  # Can add later if needed
        ]

        X.append(feature_vector)
        y.append(commit_type)

    return np.array(X), np.array(y)


def print_class_distribution(y: np.ndarray, name: str):
    """Print class distribution statistics."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)

    print(f"\n{name} class distribution:")
    print(f"{'Type':<12} {'Count':<8} {'Percentage':<10}")
    print("-" * 32)
    for label, count in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
        pct = (count / total) * 100
        print(f"{label:<12} {count:<8} {pct:>6.1f}%")
    print(f"{'Total':<12} {total:<8} {100.0:>6.1f}%")


def train_and_evaluate():
    """Train structural feature classifier and evaluate on original dataset."""

    print("=" * 80)
    print("C2: Structural Feature Classifier for Commit Type Prediction")
    print("=" * 80)
    print("\nHypothesis: Structural features (file ops, line changes, hunk patterns)")
    print("are MORE predictive and generalizable than text features (TF-IDF).")
    print("\nStructural features are:")
    print("  - Language/repo independent")
    print("  - Capture commit intent directly")
    print("  - Low dimensional (15 features vs 1000+ TF-IDF)")
    print("  - Less prone to overfitting")

    # Define paths - using ORIGINAL dataset (261 train, 29 val)
    # This is the proven high-quality dataset from Angular
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_file = dataset_dir / 'train.jsonl'
    val_file = dataset_dir / 'validation.jsonl'

    print(f"\n📂 Loading datasets...")
    print(f"   Train: {train_file}")
    print(f"   Val:   {val_file}")

    # Load data
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)

    print(f"\n✓ Loaded {len(train_data)} training examples")
    print(f"✓ Loaded {len(val_data)} validation examples")

    # Prepare features
    print("\n🔧 Extracting structural features from git diffs...")
    X_train, y_train = prepare_data(train_data)
    X_val, y_val = prepare_data(val_data)

    print(f"\n✓ Extracted features:")
    print(f"   Training:   {X_train.shape[0]} examples × {X_train.shape[1]} features")
    if X_val.shape[0] > 0:
        print(f"   Validation: {X_val.shape[0]} examples × {X_val.shape[1]} features")
    else:
        print(f"   Validation: {X_val.shape[0]} examples (no valid data after filtering)")
        print("\n⚠ WARNING: Validation set is empty after feature extraction!")
        print("   This may be due to diff parsing issues or data format problems.")
        return

    # Print class distributions
    print_class_distribution(y_train, "Training")
    print_class_distribution(y_val, "Validation")

    # Feature names for reference
    feature_names = [
        'new_files', 'modified_files', 'deleted_files',
        'insertions', 'deletions', 'net_lines', 'insert_delete_ratio',
        'hunk_count', 'avg_hunk_size', 'max_hunk_size', 'min_hunk_size',
        'lines_per_file', 'hunks_per_file', 'fragmentation_score',
        'pct_code_files',
    ]

    # Print feature statistics
    print("\n📊 Feature statistics (training set):")
    print(f"{'Feature':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 73)
    for i, name in enumerate(feature_names):
        mean = np.mean(X_train[:, i])
        std = np.std(X_train[:, i])
        min_val = np.min(X_train[:, i])
        max_val = np.max(X_train[:, i])
        print(f"{name:<25} {mean:<12.2f} {std:<12.2f} {min_val:<12.2f} {max_val:<12.2f}")

    # Build pipeline: StandardScaler + LogisticRegression
    print("\n🏗️  Building pipeline: StandardScaler → LogisticRegression")
    print("   - StandardScaler: Normalize features (mean=0, std=1)")
    print("   - LogisticRegression: Multi-class classifier (one-vs-rest)")
    print("   - max_iter=1000, default regularization (C=1.0)")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train
    print("\n🎓 Training classifier...")
    pipeline.fit(X_train, y_train)
    print("✓ Training complete")

    # Evaluate on training set
    print("\n" + "=" * 80)
    print("TRAINING SET RESULTS")
    print("=" * 80)
    y_train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {train_accuracy:.1%}")
    print(f"  F1 Score:  {train_f1:.3f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_train, y_train_pred, zero_division=0))

    print("\nConfusion Matrix:")
    cm_train = confusion_matrix(y_train, y_train_pred)
    labels_train = sorted(list(set(y_train)))
    print(f"{'':>12}", end='')
    for label in labels_train:
        print(f"{label:>8}", end='')
    print()
    for i, label in enumerate(labels_train):
        print(f"{label:>12}", end='')
        for j in range(len(labels_train)):
            print(f"{cm_train[i, j]:>8}", end='')
        print()

    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("VALIDATION SET RESULTS")
    print("=" * 80)
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {val_accuracy:.1%}")
    print(f"  F1 Score:  {val_f1:.3f}")

    print("\nDetailed Classification Report:")
    report = classification_report(y_val, y_val_pred, zero_division=0, output_dict=True)
    print(classification_report(y_val, y_val_pred, zero_division=0))

    print("\nConfusion Matrix:")
    cm_val = confusion_matrix(y_val, y_val_pred)
    labels_val = sorted(list(set(y_val)))
    print(f"{'':>12}", end='')
    for label in labels_val:
        print(f"{label:>8}", end='')
    print()
    for i, label in enumerate(labels_val):
        print(f"{label:>12}", end='')
        for j in range(len(labels_val)):
            print(f"{cm_val[i, j]:>8}", end='')
        print()

    # Calculate overfitting gap
    overfit_gap = train_accuracy - val_accuracy

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTraining Accuracy:   {train_accuracy:.1%}")
    print(f"Validation Accuracy: {val_accuracy:.1%}")
    print(f"Overfitting Gap:     {overfit_gap:+.1%}")
    print(f"\nF1 Scores:")
    print(f"  Training:   {train_f1:.3f}")
    print(f"  Validation: {val_f1:.3f}")

    # Extract feat/fix precision
    feat_precision = report.get('feat', {}).get('precision', 0.0)
    fix_precision = report.get('fix', {}).get('precision', 0.0)

    print(f"\nCritical Type Precision (Validation):")
    print(f"  feat: {feat_precision:.1%}")
    print(f"  fix:  {fix_precision:.1%}")

    # Comparison to previous experiments
    print("\n" + "=" * 80)
    print("COMPARISON TO PREVIOUS EXPERIMENTS")
    print("=" * 80)
    print("\nPrevious Results (on extended dataset):")
    print("  B2 (TF-IDF):          56.7% validation")
    print("  B3 (Enhanced TF-IDF): 59.6% validation")
    print("  Feat precision:       36.8%")
    print("  Fix precision:        74.2%")
    print(f"\nC2 (Structural) Results (on original dataset):")
    print(f"  Validation:      {val_accuracy:.1%}")
    print(f"  Feat precision:  {feat_precision:.1%}")
    print(f"  Fix precision:   {fix_precision:.1%}")
    print(f"  Overfitting gap: {overfit_gap:+.1%}")

    # Feature importance
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    print("\nLogistic Regression Coefficients (absolute mean):")
    classifier = pipeline.named_steps['classifier']

    # Get coefficients for all classes
    coefs = np.abs(classifier.coef_)  # Shape: (n_classes, n_features)
    mean_importance = np.mean(coefs, axis=0)

    # Sort by importance
    feature_importance = sorted(
        zip(feature_names, mean_importance),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n{'Rank':<6} {'Feature':<25} {'Importance':<12}")
    print("-" * 45)
    for rank, (name, importance) in enumerate(feature_importance, 1):
        print(f"{rank:<6} {name:<25} {importance:<12.3f}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    # Determine if hypothesis is supported
    target_accuracy = 0.65
    target_feat_fix = 0.70

    accuracy_met = val_accuracy >= target_accuracy
    feat_met = feat_precision >= target_feat_fix
    fix_met = fix_precision >= target_feat_fix

    print(f"\nTarget Metrics:")
    print(f"  Overall accuracy: ≥65%  {'✓' if accuracy_met else '✗'} (actual: {val_accuracy:.1%})")
    print(f"  Feat precision:   ≥70%  {'✓' if feat_met else '✗'} (actual: {feat_precision:.1%})")
    print(f"  Fix precision:    ≥70%  {'✓' if fix_met else '✗'} (actual: {fix_precision:.1%})")

    if accuracy_met and feat_met and fix_met:
        print("\n✓ HYPOTHESIS SUPPORTED!")
        print("Structural features outperform text-based approaches.")
        print("Ready to proceed with C3 (anti-overfitting measures).")
    elif accuracy_met:
        print("\n⚠ PARTIAL SUCCESS")
        print("Accuracy target met, but feat/fix precision needs improvement.")
        print("Consider adding more structural features or hybrid approach.")
    else:
        print("\n✗ HYPOTHESIS REJECTED")
        print("Structural features alone insufficient for commit classification.")
        print("Consider hybrid approach combining structural + text features.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    train_and_evaluate()
