#!/usr/bin/env python3
"""
Experiment A2 Benchmark: Evaluate Random Forest on Angular dataset

This script evaluates the same Random Forest model on the Angular benchmark dataset
to compare with A1's 61% accuracy.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
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


def prepare_dataset(data: List[Dict], is_benchmark: bool = False) -> Tuple[List[str], List[str]]:
    """Prepare features (X) and labels (y) from dataset.

    Supports multiple formats:
    1. HuggingFace: {"messages": [{"role": "user", ...}, {"role": "assistant", "content": "commit msg"}]}
    2. Simple: {"instruction": "...", "input": "...", "output": "commit msg"}
    3. Benchmark: {"diff": "...", "expectedMessage": "commit msg"}
    """
    X = []
    y = []

    for example in data:
        # Extract features based on format
        if is_benchmark and 'diff' in example:
            # Benchmark format: diff is stored directly
            diff_text = example['diff']
            file_paths = extract_file_paths(diff_text)
            file_features = generate_file_pattern_features(file_paths)
            diff_content = extract_diff_content(diff_text)
            features = f"{file_features} {diff_content}"
        else:
            # Standard format
            features = extract_features(example)

        X.append(features)

        # Extract label (commit type) based on format
        if is_benchmark and 'expectedMessage' in example:
            # Benchmark format
            assistant_message = example['expectedMessage']
        elif 'messages' in example:
            # HuggingFace format
            assistant_message = example['messages'][1]['content']
        else:
            # Simple format
            assistant_message = example.get('output', '')

        commit_type = extract_commit_type(assistant_message)
        y.append(commit_type)

    return X, y


def main():
    """Train on train.jsonl and evaluate on Angular benchmark."""

    print("=" * 80)
    print("Experiment A2 Benchmark: Random Forest on Angular Dataset")
    print("=" * 80)
    print()

    # Load datasets
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_path = dataset_dir / 'train.jsonl'

    benchmark_dir = Path(__file__).parent.parent / 'benchmark' / 'format-correctness'
    angular_path = benchmark_dir / 'angular-commits.jsonl'

    print(f"Loading training data from: {train_path}")
    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training examples")

    print(f"Loading Angular benchmark from: {angular_path}")
    angular_data = load_jsonl(angular_path)
    print(f"  Loaded {len(angular_data)} Angular examples")
    print()

    # Prepare datasets
    print("Extracting features from training data...")
    X_train, y_train = prepare_dataset(train_data)

    print("Extracting features from Angular data...")
    X_angular, y_angular = prepare_dataset(angular_data, is_benchmark=True)
    print()

    # Print label distribution
    print("Training set label distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
    print()

    print("Angular set label distribution:")
    unique_angular, counts_angular = np.unique(y_angular, return_counts=True)
    for label, count in zip(unique_angular, counts_angular):
        print(f"  {label}: {count} ({count/len(y_angular)*100:.1f}%)")
    print()

    # Create base pipeline with TF-IDF (same as A1)
    print("Building TF-IDF + Random Forest pipeline...")
    print("  TF-IDF settings: ngram_range=(1,2), max_features=1000, min_df=2, max_df=0.95")

    # Base pipeline
    base_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),  # Unigrams and bigrams (same as A1)
            max_features=1000,    # Limit feature space (same as A1)
            min_df=2,             # Ignore terms that appear in < 2 documents
            max_df=0.95,          # Ignore terms that appear in > 95% of documents
            sublinear_tf=True,    # Use sublinear TF scaling (log)
        )),
        ('clf', RandomForestClassifier(random_state=42))
    ])

    # Hyperparameter grid for Random Forest
    print("  Performing hyperparameter tuning with GridSearchCV...")
    param_grid = {
        'clf__n_estimators': [100, 200, 300, 500],  # Number of trees
        'clf__max_depth': [10, 20, None],  # Maximum depth of trees
        'clf__min_samples_split': [2, 5],  # Minimum samples to split a node
        'clf__min_samples_leaf': [1, 2],   # Minimum samples in a leaf
        'clf__class_weight': ['balanced', None],  # Handle class imbalance
    }

    # Grid search with 3-fold cross-validation
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )

    # Train
    print("Training classifier with grid search (this may take a few minutes)...")
    grid_search.fit(X_train, y_train)
    print("  Training complete!")
    print()

    # Best parameters
    print("=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    print()
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.3f}")
    print(f"Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print()

    # Use best estimator
    pipeline = grid_search.best_estimator_

    # Predict on Angular dataset
    y_angular_pred = pipeline.predict(X_angular)
    angular_accuracy = accuracy_score(y_angular, y_angular_pred)
    angular_f1 = f1_score(y_angular, y_angular_pred, average='weighted')

    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Angular Benchmark Performance:")
    print(f"  Accuracy: {angular_accuracy:.3f} ({angular_accuracy*100:.1f}%)")
    print(f"  F1 Score (weighted): {angular_f1:.3f}")
    print()

    # Comparison with A1
    a1_angular_accuracy = 0.61
    improvement = (angular_accuracy - a1_angular_accuracy) * 100
    print(f"Comparison with A1 (Logistic Regression):")
    print(f"  A1 Angular Accuracy: {a1_angular_accuracy:.3f} (61.0%)")
    print(f"  A2 Angular Accuracy: {angular_accuracy:.3f} ({angular_accuracy*100:.1f}%)")
    print(f"  Improvement: {improvement:+.1f} percentage points")
    print()

    # Classification report
    print("=" * 80)
    print("CLASSIFICATION REPORT (Angular Benchmark)")
    print("=" * 80)
    print()
    print(classification_report(y_angular, y_angular_pred, zero_division=0))

    # Confusion matrix
    print("=" * 80)
    print("CONFUSION MATRIX (Angular Benchmark)")
    print("=" * 80)
    print()
    labels = sorted(set(y_angular) | set(y_angular_pred))
    cm = confusion_matrix(y_angular, y_angular_pred, labels=labels)

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

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Angular Accuracy: {angular_accuracy*100:.1f}% (A1: 61.0%)")
    print(f"  - Angular F1: {angular_f1:.3f}")
    print(f"  - Improvement over A1: {improvement:+.1f} percentage points")
    print(f"  - Model: TF-IDF + Random Forest")
    print(f"  - Best n_estimators: {grid_search.best_params_.get('clf__n_estimators', 'N/A')}")
    print(f"  - Best max_depth: {grid_search.best_params_.get('clf__max_depth', 'N/A')}")
    print(f"  - Training examples: {len(X_train)}")
    print(f"  - Angular examples: {len(X_angular)}")
    print()


if __name__ == '__main__':
    main()
