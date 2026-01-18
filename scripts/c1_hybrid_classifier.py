#!/usr/bin/env python3
"""
Experiment C1: Hybrid Rule-Based + ML Classifier

This implements a two-phase classification approach:

PHASE 1 - RULE-BASED (High precision types):
  - If *.md OR docs/ OR README* → docs (0.88 F1)
  - If package.json OR pom.xml OR *.lock → build (0.81 F1)
  - If .github/*.yml OR .circleci/ → ci (very high precision)
  - If *test* OR *spec* OR __tests__/ → test (decent precision)
  - If .gitignore OR .eslintrc* → chore

PHASE 2 - ML FALLBACK (Semantic types):
  - For remaining commits (feat, fix, refactor, perf)
  - Use BEST model from all experiments (B4 LightGBM or B3 Enhanced TF-IDF)
  - Focus ML training on just these ambiguous types

Expected: >85% overall accuracy by combining strengths

Strategy:
1. Apply rules first - if confident match, return immediately
2. If no rule match, fall back to ML model trained on semantic types
3. Evaluate on both original validation (29 ex) and extended validation (208 ex)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL dataset file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_commit_type(commit_message: str) -> str:
    """Extract commit type from conventional commit message."""
    match = re.match(r'^(\w+)(?:\([^)]+\))?:', commit_message.strip())
    if match:
        return match.group(1).lower()
    return 'unknown'


def extract_file_paths(diff_text: str) -> List[str]:
    """Extract file paths from git diff."""
    file_paths = []
    for line in diff_text.split('\n'):
        if line.startswith('diff --git'):
            match = re.search(r'b/([^\s]+)', line)
            if match:
                file_paths.append(match.group(1))
        elif line.startswith('+++'):
            match = re.match(r'\+\+\+ b/(.+)', line)
            if match:
                path = match.group(1)
                if path != '/dev/null':
                    file_paths.append(path)
    return file_paths


# ============================================================================
# PHASE 1: RULE-BASED CLASSIFIER
# ============================================================================

def classify_by_rules(file_paths: List[str]) -> Optional[str]:
    """
    Apply HIGH-PRECISION file pattern rules ONLY.
    Returns commit type if VERY confident, None otherwise.

    Conservative strategy:
    - Only apply rules when they have >80% precision
    - Use strict thresholds to minimize false positives
    - When in doubt, fall back to ML

    Based on B2/B3 results:
    - docs: 0.88 F1 (SAFE)
    - test: Only when 100% test files
    - ci/build/chore: Only when obvious
    """
    if not file_paths:
        return None

    # Count file patterns
    docs_count = 0
    test_count = 0
    ci_count = 0
    build_count = 0

    for path in file_paths:
        path_lower = path.lower()
        filename = path.split('/')[-1].lower()

        # DOCS: *.md, docs/, README*
        if (filename.endswith('.md') or
            filename.startswith('readme') or
            'docs/' in path_lower or
            '/doc/' in path_lower or
            'documentation' in path_lower):
            docs_count += 1

        # BUILD: ONLY core dependency files (high confidence)
        if filename in ['package.json', 'package-lock.json', 'pom.xml',
                       'cargo.toml', 'cargo.lock', 'go.mod', 'go.sum',
                       'requirements.txt', 'yarn.lock', 'composer.lock']:
            build_count += 1

        # CI: ONLY .github/workflows/*.yml or .circleci/
        if (('.github/workflows/' in path_lower and filename.endswith(('.yml', '.yaml'))) or
            ('.circleci/' in path_lower)):
            ci_count += 1

        # TEST: test files only
        if (('test' in filename) or
            ('spec' in filename) or
            ('__tests__' in path_lower) or
            ('__test__' in path_lower) or
            filename.endswith('_test.go') or
            filename.endswith('.spec.ts') or
            filename.endswith('.spec.js') or
            filename.endswith('.test.tsx') or
            filename.endswith('.test.ts') or
            filename.endswith('.test.js') or
            filename.endswith('_test.py')):
            test_count += 1

    total_files = len(file_paths)

    # Apply CONSERVATIVE rules with high thresholds
    # Priority: docs > ci > build > test

    # DOCS: High confidence if >= 80% of files are docs AND at least 1 doc file
    if docs_count > 0 and docs_count / total_files >= 0.8:
        return 'docs'

    # CI: ONLY if ALL files are CI files (100%)
    if ci_count > 0 and ci_count == total_files:
        return 'ci'

    # BUILD: ONLY if ALL files are build files (100%)
    if build_count > 0 and build_count == total_files:
        return 'build'

    # TEST: ONLY if ALL files are test files (100%)
    if test_count > 0 and test_count == total_files:
        return 'test'

    # No confident rule match - fall back to ML
    return None


# ============================================================================
# PHASE 2: ML CLASSIFIER (for semantic types)
# ============================================================================

def extract_diff_content(diff_text: str, max_chars: int = 2000) -> str:
    """Extract the actual diff content for keyword analysis."""
    content_lines = []
    for line in diff_text.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            content_lines.append(line[1:].strip())
        elif line.startswith('-') and not line.startswith('---'):
            content_lines.append(line[1:].strip())
    content = ' '.join(content_lines)
    return content[:max_chars]


def generate_file_pattern_features(file_paths: List[str]) -> str:
    """Generate file pattern features from file paths (from A1)."""
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

    return ' '.join(features) if features else ''


def extract_enhanced_features(diff_text: str, file_paths: List[str]) -> np.ndarray:
    """
    Extract 13 enhanced numeric features from A5 for better feat/fix distinction.

    Features:
    1. new_files - count of new files added
    2. deleted_files - count of files deleted
    3. ins_del_ratio - insertion/deletion ratio
    4. feat_keywords - count of feat-related keywords
    5. fix_keywords - count of fix-related keywords
    6. keyword_balance - feat_keywords - fix_keywords
    7. new_funcs - count of new function definitions
    8. modified_funcs - count of modified function bodies
    9. new_imports - count of new import statements
    10. file_count - total number of files changed
    11. is_single_file - 1 if exactly 1 file changed
    12. is_few_files - 1 if 2-4 files changed
    13. is_many_files - 1 if 5+ files changed
    """
    # Count new/deleted files
    new_files = diff_text.count('new file mode')
    deleted_files = diff_text.count('deleted file mode')

    # Count insertions/deletions
    insertions = sum(1 for line in diff_text.split('\n')
                    if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff_text.split('\n')
                   if line.startswith('-') and not line.startswith('---'))

    # Insertion/deletion ratio (avoid division by zero)
    if deletions > 0:
        ins_del_ratio = insertions / deletions
    elif insertions > 0:
        ins_del_ratio = 10.0  # Large ratio if only insertions
    else:
        ins_del_ratio = 1.0  # Neutral if no changes

    # Extract diff content for keyword analysis
    diff_content = extract_diff_content(diff_text).lower()

    # Feat keywords
    feat_words = ['add', 'implement', 'introduce', 'feature', 'support',
                  'enable', 'create', 'new']
    feat_keywords = sum(diff_content.count(word) for word in feat_words)

    # Fix keywords
    fix_words = ['fix', 'bug', 'issue', 'resolve', 'correct', 'error',
                'exception', 'null', 'crash', 'repair', 'patch']
    fix_keywords = sum(diff_content.count(word) for word in fix_words)

    # Keyword balance
    keyword_balance = feat_keywords - fix_keywords

    # Count function definitions and modifications
    new_funcs = diff_content.count('+ def ') + diff_content.count('+ function ')
    new_funcs += diff_content.count('+ func ') + diff_content.count('+ const ')

    modified_funcs = diff_content.count('- def ') + diff_content.count('- function ')
    modified_funcs += diff_content.count('- func ')

    # Count new imports
    new_imports = diff_content.count('+ import ') + diff_content.count('+ from ')
    new_imports += diff_content.count('+ require(')

    # File count features
    file_count = len(file_paths)
    is_single_file = 1 if file_count == 1 else 0
    is_few_files = 1 if 2 <= file_count <= 4 else 0
    is_many_files = 1 if file_count >= 5 else 0

    return np.array([
        new_files,
        deleted_files,
        ins_del_ratio,
        feat_keywords,
        fix_keywords,
        keyword_balance,
        new_funcs,
        modified_funcs,
        new_imports,
        file_count,
        is_single_file,
        is_few_files,
        is_many_files
    ])


class SimpleFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract simple TF-IDF features like B2 (best pure ML approach)."""

    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2
        )
        self.fitted = False

    def fit(self, X, y=None):
        """Fit the TF-IDF vectorizer."""
        # Extract features for fitting
        texts = []

        for diff_text in X:
            file_paths = extract_file_paths(diff_text)
            file_pattern_features = generate_file_pattern_features(file_paths)
            diff_content = extract_diff_content(diff_text)
            combined_text = f"{file_pattern_features} {diff_content}"
            texts.append(combined_text)

        # Fit TF-IDF
        self.tfidf.fit(texts)

        self.fitted = True
        return self

    def transform(self, X):
        """Transform diffs into TF-IDF feature vectors."""
        if not self.fitted:
            raise RuntimeError("Must call fit() before transform()")

        texts = []

        for diff_text in X:
            file_paths = extract_file_paths(diff_text)
            file_pattern_features = generate_file_pattern_features(file_paths)
            diff_content = extract_diff_content(diff_text)
            combined_text = f"{file_pattern_features} {diff_content}"
            texts.append(combined_text)

        # Get TF-IDF features
        tfidf_features = self.tfidf.transform(texts)

        return tfidf_features


# ============================================================================
# HYBRID CLASSIFIER
# ============================================================================

class HybridClassifier:
    """
    Two-phase hybrid classifier:
    1. Apply rule-based classification for high-precision types
    2. Fall back to ML for semantic types (feat, fix, refactor, perf)
    """

    def __init__(self):
        self.ml_model = None
        self.feature_extractor = None
        self.semantic_types = ['feat', 'fix', 'refactor', 'perf']

    def fit(self, X_diffs: List[str], y_types: List[str]):
        """
        Train the ML component on ALL types.
        Rules will override ML predictions when confident.

        Args:
            X_diffs: List of git diff texts
            y_types: List of commit types
        """
        print(f"Training ML model on {len(X_diffs)} examples (ALL types)")

        # Show type distribution
        type_counts = {}
        for t in y_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"  Type distribution: {dict(sorted(type_counts.items(), key=lambda x: -x[1]))}")

        # Train feature extractor and ML model on ALL data
        # Use simple TF-IDF features like B2 (simpler and more robust)
        self.feature_extractor = SimpleFeatureExtractor()
        self.feature_extractor.fit(X_diffs)

        X_features = self.feature_extractor.transform(X_diffs)

        self.ml_model = LogisticRegression(
            class_weight='balanced',
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        self.ml_model.fit(X_features, y_types)

        # Print ML training accuracy
        train_pred = self.ml_model.predict(X_features)
        train_acc = accuracy_score(y_types, train_pred)
        print(f"  ML training accuracy (all types): {train_acc:.1%}")

        return self

    def predict(self, X_diffs: List[str]) -> List[str]:
        """
        Predict commit types using hybrid approach.

        Strategy:
        1. Get ML prediction for ALL examples
        2. Override with rule-based classification if high confidence

        This way:
        - ML provides baseline for everything
        - Rules correct high-precision cases (docs, ci, build, test)

        Args:
            X_diffs: List of git diff texts

        Returns:
            List of predicted commit types
        """
        # Get ML predictions for ALL examples first
        X_features = self.feature_extractor.transform(X_diffs)
        ml_predictions = self.ml_model.predict(X_features)

        predictions = []
        rule_overrides = 0
        ml_kept = 0

        for diff_text, ml_pred in zip(X_diffs, ml_predictions):
            file_paths = extract_file_paths(diff_text)

            # Check if rules give a high-confidence override
            rule_type = classify_by_rules(file_paths)

            if rule_type is not None:
                # Rule override
                predictions.append(rule_type)
                rule_overrides += 1
            else:
                # Keep ML prediction
                predictions.append(ml_pred)
                ml_kept += 1

        print(f"\nHybrid classification stats:")
        print(f"  Rule overrides: {rule_overrides}/{len(X_diffs)} ({rule_overrides/len(X_diffs):.1%})")
        print(f"  ML predictions kept: {ml_kept}/{len(X_diffs)} ({ml_kept/len(X_diffs):.1%})")

        return predictions


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(y_true: List[str], y_pred: List[str], dataset_name: str):
    """Print comprehensive evaluation metrics."""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {dataset_name}")
    print(f"{'='*70}")

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  F1 (weighted): {f1:.3f}")

    print(f"\nPer-Class Performance:")
    print(classification_report(y_true, y_pred, zero_division=0))

    print(f"\nConfusion Matrix:")
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Print header
    print(f"{'':12}", end='')
    for label in labels:
        print(f"{label:>8}", end='')
    print()

    # Print rows
    for i, label in enumerate(labels):
        print(f"{label:>12}", end='')
        for j in range(len(labels)):
            print(f"{cm[i][j]:>8}", end='')
        print()

    return accuracy, f1


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("C1: Hybrid Rule-Based + ML Classifier")
    print("="*70)

    # Load datasets
    print("\nLoading datasets...")
    train_data = load_jsonl(Path('dataset/train_extended.jsonl'))
    val_data = load_jsonl(Path('dataset/validation_extended.jsonl'))

    print(f"  Train: {len(train_data)} examples")
    print(f"  Validation (extended): {len(val_data)} examples")

    # Extract features and labels
    print("\nExtracting commit types and diffs...")

    def extract_data(data):
        X = []
        y = []
        for item in data:
            # Get diff from messages
            if 'messages' in item:
                user_msg = item['messages'][0]['content']
                # Extract diff from user message (after the prompt)
                lines = user_msg.split('\n')
                diff_start = None
                for i, line in enumerate(lines):
                    if line.strip().startswith('diff --git'):
                        diff_start = i
                        break
                if diff_start is not None:
                    diff_text = '\n'.join(lines[diff_start:])
                    X.append(diff_text)

                    # Get commit type from assistant message
                    assistant_msg = item['messages'][1]['content']
                    commit_type = extract_commit_type(assistant_msg)
                    y.append(commit_type)
        return X, y

    X_train, y_train = extract_data(train_data)
    X_val, y_val = extract_data(val_data)

    print(f"  Train: {len(X_train)} examples")
    print(f"  Validation (extended): {len(X_val)} examples")

    # Show type distribution in training set
    train_type_counts = {}
    for t in y_train:
        train_type_counts[t] = train_type_counts.get(t, 0) + 1
    print(f"\nTraining type distribution:")
    for commit_type, count in sorted(train_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {commit_type}: {count} ({count/len(y_train):.1%})")

    # Train hybrid classifier
    print("\n" + "="*70)
    print("TRAINING HYBRID CLASSIFIER")
    print("="*70)

    hybrid = HybridClassifier()
    hybrid.fit(X_train, y_train)

    # Evaluate on training set
    print("\n" + "="*70)
    print("TRAINING SET PERFORMANCE")
    print("="*70)
    y_train_pred = hybrid.predict(X_train)
    train_acc, train_f1 = evaluate_model(y_train, y_train_pred, "Training Set (1,276 examples)")

    # Evaluate on extended validation set
    print("\n" + "="*70)
    print("VALIDATION SET PERFORMANCE (EXTENDED)")
    print("="*70)
    y_val_pred = hybrid.predict(X_val)
    val_acc, val_f1 = evaluate_model(y_val, y_val_pred, "Extended Validation (208 examples)")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: C1 Hybrid vs Pure Approaches")
    print("="*70)

    print("\nExtended Validation (208 examples):")
    print(f"  Pure ML (B3 Enhanced TF-IDF): 59.6% accuracy, 0.630 F1")
    print(f"  Pure ML (B4 LightGBM):       64.4% accuracy, 0.624 F1")
    print(f"  Pure ML (B2 LogReg):          56.7% accuracy")
    print(f"  C1 Hybrid:                    {val_acc:.1%} accuracy, {val_f1:.3f} F1")

    if val_acc > 0.644:
        improvement = val_acc - 0.644
        print(f"\n  ✓ C1 BEATS BEST PURE ML by {improvement:.1%}!")
    elif val_acc > 0.85:
        print(f"\n  ✓✓ C1 ACHIEVES TARGET (>85% accuracy)!")
    else:
        gap = 0.644 - val_acc
        print(f"\n  ✗ C1 is {gap:.1%} below best pure ML")

    # Overfitting analysis
    overfit = train_acc - val_acc
    print(f"\nOverfitting Analysis:")
    print(f"  Train accuracy:      {train_acc:.1%}")
    print(f"  Validation accuracy: {val_acc:.1%}")
    print(f"  Train-Val gap:       {overfit:+.1%}")

    if overfit < 0.15:
        print(f"  ✓ Overfitting is acceptable (<15%)")
    else:
        print(f"  ⚠ High overfitting (>{overfit:.1%})")

    # Save model
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)

    model_path = Path('models/c1_hybrid_classifier.pkl')
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(hybrid, f)

    print(f"  Saved to: {model_path}")
    print(f"  Model size: {model_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
