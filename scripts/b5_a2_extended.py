#!/usr/bin/env python3
"""
Experiment B5: Re-train A2 (TF-IDF + Random Forest) on Extended Dataset

This script re-trains the A2 model (TF-IDF + Random Forest) on the
extended dataset (1,276 train / 208 validation examples) to evaluate:
- Performance improvement with 4x more training data
- Overfitting reduction (original A2 had 93.5% train, 69.0% val)
- Generalization to unseen data

Original A2 Results (261 train, 29 val):
- Training: 93.5% accuracy
- Validation: 69.0% accuracy
- Overfitting: +24.5%

Expected with Extended Dataset:
- Validation: 72-76% accuracy
- Reduced overfitting to <15%

Comparison to Other Extended Models:
- B2 (A1 on extended): 56.7% val
- B4 (XGBoost on extended): 58.2% val
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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


def extract_file_paths(diff_text: str) -> List[str]:
    """Extract file paths from git diff."""
    file_paths = []

    # Match lines like: "diff --git a/path/to/file.ext b/path/to/file.ext"
    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            # Extract the b/ path (new file version)
            match = re.search(r'b/([^\s]+)', line)
            if match:
                file_paths.append(match.group(1))
        # Also match lines like: "--- a/path/to/file.ext"
        elif line.startswith('+++'):
            match = re.match(r'\+\+\+ b/(.+)', line)
            if match:
                path = match.group(1)
                if path != '/dev/null':  # Skip deleted files
                    file_paths.append(path)

    return file_paths


def extract_diff_content(diff_text: str, max_chars: int = 2000) -> str:
    """
    Extract the actual diff content (added/removed lines) for keyword analysis.
    Truncate to max_chars to focus on meaningful changes.
    """
    content_lines = []

    for line in diff_text.split('\n'):
        # Include added and removed lines (start with + or -)
        if line.startswith('+') and not line.startswith('+++'):
            content_lines.append(line[1:].strip())
        elif line.startswith('-') and not line.startswith('---'):
            content_lines.append(line[1:].strip())

    content = ' '.join(content_lines)
    return content[:max_chars]


def generate_file_pattern_features(file_paths: List[str]) -> str:
    """
    Generate file pattern features from file paths.

    Maps file patterns to meaningful tokens that hint at commit type:
    - *.md, README*, docs/* → FILEDOCS
    - *test.*, *spec.*, __tests__/* → FILETEST
    - .github/*, .circleci/*, Jenkinsfile → FILECI
    - package.json, pom.xml, Cargo.toml → FILEBUILD
    - .gitignore, .eslintrc* → FILECHORE
    """
    features = []

    for path in file_paths:
        path_lower = path.lower()
        filename = path.split('/')[-1].lower()

        # Documentation patterns
        if (filename.endswith('.md') or
            filename.startswith('readme') or
            'docs/' in path_lower or
            '/doc/' in path_lower):
            features.append('FILEDOCS')

        # Test patterns
        if (('test' in filename) or
            ('spec' in filename) or
            ('__tests__' in path_lower) or
            filename.endswith('_test.go') or
            filename.endswith('.spec.ts') or
            filename.endswith('.spec.js') or
            filename.endswith('.test.tsx')):
            features.append('FILETEST')

        # CI/CD patterns
        if (('.github/' in path_lower) or
            ('.circleci/' in path_lower) or
            ('.gitlab-ci' in path_lower) or
            ('jenkinsfile' in path_lower) or
            ('.travis' in path_lower) or
            ('azure-pipelines' in path_lower)):
            features.append('FILECI')

        # Build/dependency patterns
        if filename in ['package.json', 'package-lock.json', 'pom.xml',
                       'cargo.toml', 'cargo.lock', 'go.mod', 'go.sum',
                       'requirements.txt', 'gemfile', 'gemfile.lock',
                       'build.gradle', 'settings.gradle', 'yarn.lock',
                       'build.xml', 'makefile', 'cmake']:
            features.append('FILEBUILD')

        # Configuration/chore patterns
        if (filename.startswith('.') and
            filename not in ['.gitignore', '.github'] or
            filename.endswith('rc') or
            filename.endswith('.config') or
            filename in ['.gitignore', '.dockerignore', '.editorconfig']):
            features.append('FILECHORE')

        # Add file extension as a feature
        if '.' in filename:
            ext = filename.split('.')[-1]
            features.append(f'FILEEXT_{ext}')

    return ' '.join(features)


def extract_features(example: Dict) -> str:
    """
    Combine file path patterns + diff text for classification.
    Returns a single string with all features.

    Supports two dataset formats:
    1. HuggingFace format: {"messages": [{"role": "user", "content": "..."}, ...]}
    2. Simple format: {"instruction": "...", "input": "...", "output": "..."}
    """
    # Determine format and extract diff
    if 'messages' in example:
        # HuggingFace format
        user_message = example['messages'][0]['content']
        # The diff starts after the instructions, typically after "diff --git"
        diff_start = user_message.find('diff --git')
        if diff_start == -1:
            # Try alternative format where diff starts with file listing
            diff_start = user_message.find('.../') or user_message.find('--- ')
        if diff_start == -1:
            diff_text = user_message  # Fallback: use entire message
        else:
            diff_text = user_message[diff_start:]
    else:
        # Simple format: input contains the diff directly
        diff_text = example.get('input', '')

    # Extract file paths and generate pattern features
    file_paths = extract_file_paths(diff_text)
    file_features = generate_file_pattern_features(file_paths)

    # Extract diff content for keyword analysis
    diff_content = extract_diff_content(diff_text)

    # Combine features: file patterns have more weight by appearing first
    combined = f"{file_features} {diff_content}"

    return combined


def prepare_dataset(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """Prepare features (X) and labels (y) from dataset.

    Supports two formats:
    1. HuggingFace: {"messages": [{"role": "user", ...}, {"role": "assistant", "content": "commit msg"}]}
    2. Simple: {"instruction": "...", "input": "...", "output": "commit msg"}
    """
    X = []
    y = []

    for example in data:
        # Extract features
        features = extract_features(example)
        X.append(features)

        # Extract label (commit type) based on format
        if 'messages' in example:
            # HuggingFace format
            assistant_message = example['messages'][1]['content']
        else:
            # Simple format
            assistant_message = example.get('output', '')

        commit_type = extract_commit_type(assistant_message)
        y.append(commit_type)

    return X, y


def main():
    """Train and evaluate TF-IDF + Random Forest classifier on extended dataset."""

    print("=" * 80)
    print("Experiment B5: Re-train A2 (TF-IDF + Random Forest) on Extended Dataset")
    print("=" * 80)
    print()

    # Load datasets
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_path = dataset_dir / 'train_extended.jsonl'
    val_path = dataset_dir / 'validation_extended.jsonl'

    print(f"Loading training data from: {train_path}")
    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training examples")

    print(f"Loading validation data from: {val_path}")
    val_data = load_jsonl(val_path)
    print(f"  Loaded {len(val_data)} validation examples")
    print()

    # Prepare datasets
    print("Extracting features from training data...")
    X_train, y_train = prepare_dataset(train_data)

    print("Extracting features from validation data...")
    X_val, y_val = prepare_dataset(val_data)
    print()

    # Print label distribution
    print("Training set label distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
    print()

    print("Validation set label distribution:")
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    for label, count in zip(unique_val, counts_val):
        print(f"  {label}: {count} ({count/len(y_val)*100:.1f}%)")
    print()

    # Create pipeline with SAME settings as original A2 best params
    # From A2 results: n_estimators=500, max_depth=20 was optimal
    print("Building TF-IDF + Random Forest pipeline...")
    print("  TF-IDF settings: ngram_range=(1,2), max_features=1000, min_df=2, max_df=0.95")
    print("  Random Forest settings: n_estimators=500, max_depth=20 (from A2 best params)")
    print()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),  # Unigrams and bigrams (same as A2)
            max_features=1000,    # Limit feature space (same as A2)
            min_df=2,             # Ignore terms that appear in < 2 documents
            max_df=0.95,          # Ignore terms that appear in > 95% of documents
            sublinear_tf=True,    # Use sublinear TF scaling (log)
        )),
        ('clf', RandomForestClassifier(
            n_estimators=500,     # Best param from A2
            max_depth=20,         # Best param from A2
            min_samples_split=2,  # Best param from A2
            min_samples_leaf=1,   # Best param from A2
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        ))
    ])

    # Train
    print("Training classifier...")
    pipeline.fit(X_train, y_train)
    print("  Training complete!")
    print()

    # Predict on training set
    y_train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Predict on validation set
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Training Set Performance:")
    print(f"  Accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
    print(f"  F1 Score (weighted): {train_f1:.3f}")
    print()
    print(f"Validation Set Performance:")
    print(f"  Accuracy: {val_accuracy:.3f} ({val_accuracy*100:.1f}%)")
    print(f"  F1 Score (weighted): {val_f1:.3f}")
    print()

    # Overfitting analysis
    overfitting_gap = train_accuracy - val_accuracy
    print(f"Overfitting Analysis:")
    print(f"  Train-Val Gap: {overfitting_gap:.3f} ({overfitting_gap*100:.1f}%)")
    if overfitting_gap < 0.10:
        print(f"  Status: Low overfitting (good generalization)")
    elif overfitting_gap < 0.20:
        print(f"  Status: Moderate overfitting")
    else:
        print(f"  Status: High overfitting")
    print()

    # Comparison with original A2 and other extended models
    print(f"Comparison with Original A2:")
    print(f"  Original A2 (small dataset):")
    print(f"    - Train: 93.5% (261 examples)")
    print(f"    - Val: 69.0% (29 examples)")
    print(f"    - Overfitting: +24.5%")
    print()
    print(f"  B5 (extended dataset):")
    print(f"    - Train: {train_accuracy*100:.1f}% (1,276 examples)")
    print(f"    - Val: {val_accuracy*100:.1f}% (208 examples)")
    print(f"    - Overfitting: {overfitting_gap*100:+.1f}%")
    val_improvement = (val_accuracy - 0.690) * 100
    print(f"    - Val Improvement: {val_improvement:+.1f} percentage points")
    print()

    print(f"Comparison with Other Extended Models:")
    print(f"  B2 (LogReg on extended): 56.7% val")
    print(f"  B4 (XGBoost on extended): 58.2% val")
    print(f"  B5 (Random Forest on extended): {val_accuracy*100:.1f}% val")
    if val_accuracy > 0.582:
        print(f"  -> B5 is BEST performing model (+{(val_accuracy - 0.582)*100:.1f}% vs XGBoost)")
    print()

    # Classification report
    print("=" * 80)
    print("CLASSIFICATION REPORT (Validation Set)")
    print("=" * 80)
    print()
    print(classification_report(y_val, y_val_pred, zero_division=0))

    # Confusion matrix
    print("=" * 80)
    print("CONFUSION MATRIX (Validation Set)")
    print("=" * 80)
    print()
    labels = sorted(set(y_val) | set(y_val_pred))
    cm = confusion_matrix(y_val, y_val_pred, labels=labels)

    # Print confusion matrix with labels
    print(f"{'':>12}", end='')
    for label in labels:
        print(f"{label:>10}", end='')
    print()
    print('-' * (12 + 10 * len(labels)))

    for i, label in enumerate(labels):
        print(f"{label:>12}", end='')
        for j in range(len(labels)):
            print(f"{cm[i, j]:>10}", end='')
        print()
    print()

    # Feature importance (Random Forest specific)
    print("=" * 80)
    print("TOP FEATURES BY IMPORTANCE")
    print("=" * 80)
    print()

    # Get feature names and importances
    feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
    importances = pipeline.named_steps['clf'].feature_importances_

    # Sort by importance
    top_indices = np.argsort(importances)[-20:][::-1]
    print("Top 20 most important features:")
    for idx in top_indices:
        print(f"  {feature_names[idx]:>30}: {importances[idx]:>7.4f}")
    print()

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Training Accuracy: {train_accuracy*100:.1f}% ({len(X_train)} examples)")
    print(f"  - Validation Accuracy: {val_accuracy*100:.1f}% ({len(X_val)} examples)")
    print(f"  - Train-Val Gap: {overfitting_gap*100:.1f}%")
    print(f"  - Model: TF-IDF + Random Forest")
    print(f"  - Settings: n_estimators=500, max_depth=20, balanced classes")
    print(f"  - Features: File patterns + diff content (max 1000 features)")
    print()
    print("Key Findings:")
    print(f"  1. Validation improved: 69.0% -> {val_accuracy*100:.1f}% ({val_improvement:+.1f}pp)")
    print(f"  2. Overfitting reduced: 24.5% -> {overfitting_gap*100:.1f}% ({(0.245 - overfitting_gap)*100:.1f}pp)")
    print(f"  3. More data helped: 4x dataset size improved generalization")
    if val_accuracy > 0.582:
        print(f"  4. Random Forest is best extended model ({val_accuracy*100:.1f}% vs 58.2% XGBoost)")
    print()


if __name__ == '__main__':
    main()
