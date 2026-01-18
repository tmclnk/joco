#!/usr/bin/env python3
"""
Experiment B6: Re-train A3 (SVM RBF) on Extended Dataset

This script re-trains the A3 model (TF-IDF + SVM with RBF kernel) on the
extended dataset (1,276 train / 208 validation examples) to evaluate:
- Performance improvement with 4x more training data
- Overfitting reduction (original A3 had severe overfitting)
- Generalization to unseen data (Angular benchmark)

Original A3 Results:
- Training: 92.3% accuracy (261 examples)
- Validation: 65.5% accuracy (29 examples)
- Overfitting: +26.8% (WORST of all models)

Expected with Extended Dataset:
- Training: 85-90% accuracy (more realistic with more data)
- Validation: 70-75% accuracy (better generalization)
- Overfitting: <18% (reduced)
- Angular benchmark: Similar to validation performance
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
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


def prepare_angular_benchmark(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """Prepare Angular benchmark data which has a different format.

    Format: {"id": "...", "diff": "...", "expectedMessage": "...", ...}
    """
    X = []
    y = []

    for example in data:
        # Extract diff and create features
        diff_text = example.get('diff', '')
        file_paths = extract_file_paths(diff_text)
        file_features = generate_file_pattern_features(file_paths)
        diff_content = extract_diff_content(diff_text)
        combined = f"{file_features} {diff_content}"
        X.append(combined)

        # Extract expected commit type
        expected_message = example.get('expectedMessage', '')
        commit_type = extract_commit_type(expected_message)
        y.append(commit_type)

    return X, y


def main():
    """Train and evaluate TF-IDF + SVM with RBF kernel on extended dataset."""

    print("=" * 80)
    print("Experiment B6: Re-train A3 (SVM RBF) on Extended Dataset")
    print("=" * 80)
    print()

    # Load datasets
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_path = dataset_dir / 'train_extended.jsonl'
    val_path = dataset_dir / 'validation_extended.jsonl'
    angular_path = Path(__file__).parent.parent / 'benchmark' / 'format-correctness' / 'angular-commits.jsonl'

    print(f"Loading training data from: {train_path}")
    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training examples")

    print(f"Loading validation data from: {val_path}")
    val_data = load_jsonl(val_path)
    print(f"  Loaded {len(val_data)} validation examples")

    print(f"Loading Angular benchmark from: {angular_path}")
    angular_data = load_jsonl(angular_path)
    print(f"  Loaded {len(angular_data)} Angular examples")
    print()

    # Prepare datasets
    print("Extracting features from training data...")
    X_train, y_train = prepare_dataset(train_data)

    print("Extracting features from validation data...")
    X_val, y_val = prepare_dataset(val_data)

    print("Extracting features from Angular benchmark...")
    X_angular, y_angular = prepare_angular_benchmark(angular_data)
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

    print("Angular benchmark label distribution:")
    unique_angular, counts_angular = np.unique(y_angular, return_counts=True)
    for label, count in zip(unique_angular, counts_angular):
        print(f"  {label}: {count} ({count/len(y_angular)*100:.1f}%)")
    print()

    # Create base pipeline (SAME as A3)
    print("Building TF-IDF + SVM with RBF kernel pipeline...")
    print()

    # TF-IDF vectorizer (same as A3)
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=1000,    # Limit feature space
        min_df=2,             # Ignore terms that appear in < 2 documents
        max_df=0.95,          # Ignore terms that appear in > 95% of documents
        sublinear_tf=True,    # Use sublinear TF scaling (log)
    )

    # Create pipeline with SVM
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('svm', SVC(kernel='rbf', random_state=42, class_weight='balanced'))
    ])

    # Define hyperparameter grid (SAME as A3)
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': [0.001, 0.01, 0.1, 1, 'scale']
    }

    print("Hyperparameter search space:")
    print(f"  C: {param_grid['svm__C']}")
    print(f"  gamma: {param_grid['svm__gamma']}")
    print(f"  Total combinations: {len(param_grid['svm__C']) * len(param_grid['svm__gamma'])}")
    print()

    # Grid search with cross-validation
    print("Starting GridSearchCV (this may take several minutes)...")
    print("  Using 3-fold cross-validation on training set")
    print()

    start_time = time.time()

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1,
        return_train_score=True
    )

    grid_search.fit(X_train, y_train)

    training_time = time.time() - start_time

    print()
    print(f"  GridSearchCV complete! (took {training_time:.1f} seconds)")
    print()

    # Best parameters
    print("=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    print()
    print(f"Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print()
    print(f"Best cross-validation score: {grid_search.best_score_:.3f} ({grid_search.best_score_*100:.1f}%)")
    print()

    # Get best estimator
    best_pipeline = grid_search.best_estimator_

    # Predict on training set
    y_train_pred = best_pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Predict on validation set
    y_val_pred = best_pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    # Predict on Angular benchmark
    y_angular_pred = best_pipeline.predict(X_angular)
    angular_accuracy = accuracy_score(y_angular, y_angular_pred)
    angular_f1 = f1_score(y_angular, y_angular_pred, average='weighted')

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
    print(f"Angular Benchmark Performance:")
    print(f"  Accuracy: {angular_accuracy:.3f} ({angular_accuracy*100:.1f}%)")
    print(f"  F1 Score (weighted): {angular_f1:.3f}")
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

    # Classification report for validation
    print("=" * 80)
    print("CLASSIFICATION REPORT (Validation Set)")
    print("=" * 80)
    print()
    print(classification_report(y_val, y_val_pred, zero_division=0))

    # Classification report for Angular
    print("=" * 80)
    print("CLASSIFICATION REPORT (Angular Benchmark)")
    print("=" * 80)
    print()
    print(classification_report(y_angular, y_angular_pred, zero_division=0))

    # Confusion matrix for validation
    print("=" * 80)
    print("CONFUSION MATRIX (Validation Set)")
    print("=" * 80)
    print()
    labels_val = sorted(set(y_val) | set(y_val_pred))
    cm_val = confusion_matrix(y_val, y_val_pred, labels=labels_val)

    # Print confusion matrix with labels
    print(f"{'':>12}", end='')
    for label in labels_val:
        print(f"{label:>10}", end='')
    print()
    print('-' * (12 + 10 * len(labels_val)))

    for i, label in enumerate(labels_val):
        print(f"{label:>12}", end='')
        for j in range(len(labels_val)):
            print(f"{cm_val[i, j]:>10}", end='')
        print()
    print()

    # Confusion matrix for Angular
    print("=" * 80)
    print("CONFUSION MATRIX (Angular Benchmark)")
    print("=" * 80)
    print()
    labels_angular = sorted(set(y_angular) | set(y_angular_pred))
    cm_angular = confusion_matrix(y_angular, y_angular_pred, labels=labels_angular)

    # Print confusion matrix with labels
    print(f"{'':>12}", end='')
    for label in labels_angular:
        print(f"{label:>10}", end='')
    print()
    print('-' * (12 + 10 * len(labels_angular)))

    for i, label in enumerate(labels_angular):
        print(f"{label:>12}", end='')
        for j in range(len(labels_angular)):
            print(f"{cm_angular[i, j]:>10}", end='')
        print()
    print()

    # Analyze feat/fix confusion
    print("=" * 80)
    print("FEAT/FIX ANALYSIS")
    print("=" * 80)
    print()

    if 'feat' in labels_val and 'fix' in labels_val:
        feat_idx = labels_val.index('feat')
        fix_idx = labels_val.index('fix')

        feat_as_fix = cm_val[feat_idx, fix_idx]
        fix_as_feat = cm_val[fix_idx, feat_idx]

        feat_total = np.sum(cm_val[feat_idx, :])
        fix_total = np.sum(cm_val[fix_idx, :])

        print(f"Confusion between feat and fix (Validation):")
        print(f"  feat misclassified as fix: {feat_as_fix}/{feat_total} ({feat_as_fix/feat_total*100:.1f}%)")
        print(f"  fix misclassified as feat: {fix_as_feat}/{fix_total} ({fix_as_feat/fix_total*100:.1f}%)")
        print()

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Training Accuracy: {train_accuracy*100:.1f}% ({len(X_train)} examples)")
    print(f"  - Validation Accuracy: {val_accuracy*100:.1f}% ({len(X_val)} examples)")
    print(f"  - Angular Accuracy: {angular_accuracy*100:.1f}% ({len(X_angular)} examples)")
    print(f"  - Train-Val Gap: {overfitting_gap*100:.1f}%")
    print(f"  - Model: SVM with RBF kernel (SAME as A3)")
    print(f"  - Best C: {grid_search.best_params_['svm__C']}")
    print(f"  - Best gamma: {grid_search.best_params_['svm__gamma']}")
    print(f"  - Features: TF-IDF (file patterns + diff content, max 1000)")
    print(f"  - Training time: {training_time:.1f} seconds")
    print()
    print("Comparison to Original A3:")
    print("  Original A3 (small dataset):")
    print("    - Train: 92.3% (261 examples)")
    print("    - Val: 65.5% (29 examples)")
    print("    - Overfitting: +26.8% (WORST)")
    print()
    print("  B6 (extended dataset):")
    print(f"    - Train: {train_accuracy*100:.1f}% (1,276 examples)")
    print(f"    - Val: {val_accuracy*100:.1f}% (208 examples)")
    print(f"    - Angular: {angular_accuracy*100:.1f}% (100 examples)")
    print(f"    - Overfitting: {overfitting_gap*100:.1f}%")
    print()


if __name__ == '__main__':
    main()
