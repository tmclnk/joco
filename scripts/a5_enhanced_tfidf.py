#!/usr/bin/env python3
"""
Experiment A5: Enhanced TF-IDF with Features Targeting Feat vs Fix Distinction

This script enhances A1 baseline with additional features specifically designed to
distinguish between 'feat' and 'fix' commit types.

New features:
1. Code change patterns: new file count, insertion/deletion ratio
2. Diff keywords: 'add'/'implement' → feat, 'fix'/'bug' → fix
3. File count: many files (5+) → feat, few files (1-2) → fix
4. Code structure: new functions → feat, modified functions → fix

Target: >50% precision on feat and fix (up from A1's 31%/36%)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
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

    return ' '.join(features)


# NEW: Enhanced features for feat/fix distinction

def count_new_files(diff_text: str) -> int:
    """Count new files being added (indicator of feat)."""
    new_file_count = 0
    for line in diff_text.split('\n'):
        if line.startswith('--- /dev/null'):
            new_file_count += 1
    return new_file_count


def count_deleted_files(diff_text: str) -> int:
    """Count files being deleted."""
    deleted_file_count = 0
    for line in diff_text.split('\n'):
        if line.startswith('+++ /dev/null'):
            deleted_file_count += 1
    return deleted_file_count


def calculate_insertion_deletion_ratio(diff_text: str) -> float:
    """
    Calculate insertion/deletion ratio.
    High ratio (more insertions) → feat
    Low ratio (balanced/more deletions) → fix
    """
    insertions = 0
    deletions = 0

    for line in diff_text.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            insertions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1

    # Avoid division by zero
    if deletions == 0:
        return 10.0 if insertions > 0 else 1.0

    return insertions / deletions


def extract_feat_keywords(diff_text: str) -> int:
    """Count feat-related keywords in diff."""
    feat_keywords = [
        r'\badd\b', r'\badded\b', r'\badding\b',
        r'\bimplement\b', r'\bimplemented\b', r'\bimplementing\b',
        r'\bintroduce\b', r'\bintroduced\b', r'\bintroducing\b',
        r'\bfeature\b', r'\bfeatures\b',
        r'\bsupport\b', r'\bsupported\b', r'\bsupporting\b',
        r'\benable\b', r'\benabled\b', r'\benabling\b',
        r'\bcreate\b', r'\bcreated\b', r'\bcreating\b',
        r'\bnew\b',
    ]

    count = 0
    diff_lower = diff_text.lower()
    for pattern in feat_keywords:
        count += len(re.findall(pattern, diff_lower))

    return count


def extract_fix_keywords(diff_text: str) -> int:
    """Count fix-related keywords in diff."""
    fix_keywords = [
        r'\bfix\b', r'\bfixed\b', r'\bfixes\b', r'\bfixing\b',
        r'\bbug\b', r'\bbugs\b',
        r'\bissue\b', r'\bissues\b',
        r'\bresolve\b', r'\bresolved\b', r'\bresolves\b', r'\bresolving\b',
        r'\bcorrect\b', r'\bcorrected\b', r'\bcorrecting\b',
        r'\berror\b', r'\berrors\b',
        r'\bexception\b', r'\bexceptions\b',
        r'\bnull\b', r'\bnullpointer\b',
        r'\bcrash\b', r'\bcrashed\b', r'\bcrashing\b',
        r'\brepair\b', r'\brepaired\b', r'\brepairing\b',
        r'\bpatch\b', r'\bpatched\b', r'\bpatching\b',
    ]

    count = 0
    diff_lower = diff_text.lower()
    for pattern in fix_keywords:
        count += len(re.findall(pattern, diff_lower))

    return count


def count_new_functions(diff_text: str) -> int:
    """
    Count new function definitions in diff.
    Patterns: def, function, func, fn, public/private class methods
    """
    new_function_patterns = [
        r'^\+.*\bdef\s+\w+\s*\(',  # Python
        r'^\+.*\bfunction\s+\w+\s*\(',  # JavaScript/TypeScript
        r'^\+.*\bfn\s+\w+\s*\(',  # Rust
        r'^\+.*\bfunc\s+\w+\s*\(',  # Go
        r'^\+.*(public|private|protected)\s+\w+\s+\w+\s*\(',  # Java/C#
        r'^\+.*\w+\s+\w+\s*\([^)]*\)\s*\{',  # C/C++
    ]

    count = 0
    for line in diff_text.split('\n'):
        for pattern in new_function_patterns:
            if re.search(pattern, line):
                count += 1
                break  # Count each line once

    return count


def count_modified_functions(diff_text: str) -> int:
    """
    Count modifications to existing function bodies.
    Look for changes inside function blocks.
    """
    # This is a heuristic: count lines that modify code inside indented blocks
    modified_count = 0
    in_function = False

    for line in diff_text.split('\n'):
        # Check if we're entering a function definition
        if re.search(r'(def|function|func|fn)\s+\w+', line):
            in_function = True

        # Count modifications inside functions (lines starting with +/- and indented)
        if in_function and (line.startswith('+') or line.startswith('-')):
            if re.match(r'^[+-]\s{2,}', line):  # At least 2 spaces after +/-
                modified_count += 1

    return modified_count


def count_new_imports(diff_text: str) -> int:
    """Count new import/require statements (indicator of feat)."""
    import_patterns = [
        r'^\+.*\bimport\b',  # Python, Java, JavaScript
        r'^\+.*\brequire\b',  # Node.js
        r'^\+.*\buse\b',  # Rust, PHP
        r'^\+.*\binclude\b',  # C/C++
    ]

    count = 0
    for line in diff_text.split('\n'):
        for pattern in import_patterns:
            if re.search(pattern, line):
                count += 1
                break

    return count


class EnhancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract enhanced numeric features from diffs.
    These features are designed to distinguish feat from fix commits.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        X is a list of dicts containing raw examples.
        Returns a 2D numpy array of numeric features.
        """
        features = []

        for example in X:
            # Extract diff text
            if 'messages' in example:
                user_message = example['messages'][0]['content']
                diff_start = user_message.find('diff --git')
                if diff_start == -1:
                    diff_start = user_message.find('.../') or user_message.find('--- ')
                if diff_start == -1:
                    diff_text = user_message
                else:
                    diff_text = user_message[diff_start:]
            else:
                diff_text = example.get('input', '')

            # Extract file paths
            file_paths = extract_file_paths(diff_text)
            file_count = len(set(file_paths))  # Unique file count

            # Calculate features
            new_files = count_new_files(diff_text)
            deleted_files = count_deleted_files(diff_text)
            ins_del_ratio = calculate_insertion_deletion_ratio(diff_text)
            feat_keywords = extract_feat_keywords(diff_text)
            fix_keywords = extract_fix_keywords(diff_text)
            new_funcs = count_new_functions(diff_text)
            modified_funcs = count_modified_functions(diff_text)
            new_imports = count_new_imports(diff_text)

            # File count buckets (categorical encoded as one-hot)
            is_single_file = 1 if file_count == 1 else 0
            is_few_files = 1 if 2 <= file_count <= 4 else 0
            is_many_files = 1 if file_count >= 5 else 0

            # Keyword balance (feat_keywords - fix_keywords)
            keyword_balance = feat_keywords - fix_keywords

            # Create feature vector
            feature_vec = [
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
                is_many_files,
            ]

            features.append(feature_vec)

        return np.array(features, dtype=float)


def extract_text_features(example: Dict) -> str:
    """
    Extract text features for TF-IDF (from A1).
    Combines file path patterns + diff text for classification.
    """
    # Extract diff text
    if 'messages' in example:
        user_message = example['messages'][0]['content']
        diff_start = user_message.find('diff --git')
        if diff_start == -1:
            diff_start = user_message.find('.../') or user_message.find('--- ')
        if diff_start == -1:
            diff_text = user_message
        else:
            diff_text = user_message[diff_start:]
    else:
        diff_text = example.get('input', '')

    # Extract file paths and generate pattern features
    file_paths = extract_file_paths(diff_text)
    file_features = generate_file_pattern_features(file_paths)

    # Extract diff content for keyword analysis
    diff_content = extract_diff_content(diff_text)

    # Combine features
    combined = f"{file_features} {diff_content}"

    return combined


def prepare_dataset(data: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Prepare features and labels from dataset.
    Returns: (text_features, raw_examples, labels)
    """
    text_features = []
    raw_examples = []
    labels = []

    for example in data:
        # Extract text features for TF-IDF
        text_feat = extract_text_features(example)
        text_features.append(text_feat)

        # Keep raw example for numeric feature extraction
        raw_examples.append(example)

        # Extract label
        if 'messages' in example:
            assistant_message = example['messages'][1]['content']
        else:
            assistant_message = example.get('output', '')

        commit_type = extract_commit_type(assistant_message)
        labels.append(commit_type)

    return text_features, raw_examples, labels


# Custom wrapper to handle dual input
class DualInputWrapper:
    """Wraps a pipeline to handle both text and raw data inputs."""

    def __init__(self, text_vectorizer, numeric_extractor, classifier):
        self.text_vectorizer = text_vectorizer
        self.numeric_extractor = numeric_extractor
        self.classifier = classifier
        self.scaler = StandardScaler()

    def fit(self, X_text, X_raw, y):
        # Fit text vectorizer
        text_features = self.text_vectorizer.fit_transform(X_text)

        # Fit numeric extractor and scaler
        numeric_features = self.numeric_extractor.fit_transform(X_raw)
        numeric_features = self.scaler.fit_transform(numeric_features)

        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([text_features, numeric_features])

        # Fit classifier
        self.classifier.fit(combined_features, y)

        return self

    def predict(self, X_text, X_raw):
        # Transform text features
        text_features = self.text_vectorizer.transform(X_text)

        # Transform numeric features
        numeric_features = self.numeric_extractor.transform(X_raw)
        numeric_features = self.scaler.transform(numeric_features)

        # Combine features
        from scipy.sparse import hstack
        combined_features = hstack([text_features, numeric_features])

        # Predict
        return self.classifier.predict(combined_features)


def main():
    """Train and evaluate enhanced TF-IDF classifier."""

    print("=" * 80)
    print("Experiment A5: Enhanced TF-IDF with Feat/Fix Features")
    print("=" * 80)
    print()

    # Load datasets
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_path = dataset_dir / 'train.jsonl'
    val_path = dataset_dir / 'validation.jsonl'

    print(f"Loading training data from: {train_path}")
    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training examples")

    print(f"Loading validation data from: {val_path}")
    val_data = load_jsonl(val_path)
    print(f"  Loaded {len(val_data)} validation examples")
    print()

    # Prepare datasets
    print("Extracting features from training data...")
    X_train_text, X_train_raw, y_train = prepare_dataset(train_data)

    print("Extracting features from validation data...")
    X_val_text, X_val_raw, y_val = prepare_dataset(val_data)
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

    # Create components
    print("Building Enhanced TF-IDF + Logistic Regression pipeline...")
    print("  - TF-IDF vectorizer for text features")
    print("  - Custom numeric features for feat/fix distinction:")
    print("    * New file count, deleted file count")
    print("    * Insertion/deletion ratio")
    print("    * Feat keywords (add, implement, feature, etc.)")
    print("    * Fix keywords (fix, bug, error, etc.)")
    print("    * New function definitions")
    print("    * Modified function bodies")
    print("    * New import statements")
    print("    * File count buckets (1, 2-4, 5+)")
    print()

    text_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    numeric_extractor = EnhancedFeatureExtractor()

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        class_weight='balanced',
    )

    model = DualInputWrapper(text_vectorizer, numeric_extractor, classifier)

    # Train
    print("Training classifier...")
    model.fit(X_train_text, X_train_raw, y_train)
    print("  Training complete!")
    print()

    # Predict
    y_train_pred = model.predict(X_train_text, X_train_raw)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    y_val_pred = model.predict(X_val_text, X_val_raw)
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

    # Classification report
    print("=" * 80)
    print("CLASSIFICATION REPORT (Validation Set)")
    print("=" * 80)
    print()
    report = classification_report(y_val, y_val_pred, zero_division=0, output_dict=True)
    print(classification_report(y_val, y_val_pred, zero_division=0))

    # Extract feat/fix precision for comparison
    feat_precision = report.get('feat', {}).get('precision', 0.0)
    fix_precision = report.get('fix', {}).get('precision', 0.0)

    # Confusion matrix
    print("=" * 80)
    print("CONFUSION MATRIX (Validation Set)")
    print("=" * 80)
    print()
    labels = sorted(set(y_val) | set(y_val_pred))
    cm = confusion_matrix(y_val, y_val_pred, labels=labels)

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
    print(f"  - Validation Accuracy: {val_accuracy*100:.1f}%")
    print(f"  - Validation F1: {val_f1:.3f}")
    print(f"  - Feat Precision: {feat_precision*100:.1f}% (A1 baseline: 31%)")
    print(f"  - Fix Precision: {fix_precision*100:.1f}% (A1 baseline: 36%)")
    print(f"  - Model: Enhanced TF-IDF + Logistic Regression")
    print(f"  - Features: Text (TF-IDF) + 13 numeric feat/fix features")
    print(f"  - Training examples: {len(X_train_text)}")
    print(f"  - Validation examples: {len(X_val_text)}")
    print()

    # Comparison to A1
    print("=" * 80)
    print("COMPARISON TO A1 BASELINE")
    print("=" * 80)
    print()
    print(f"A1 Baseline:")
    print(f"  - Validation Accuracy: 75.9%")
    print(f"  - Feat Precision: 31%")
    print(f"  - Fix Precision: 36%")
    print()
    print(f"A5 Enhanced:")
    print(f"  - Validation Accuracy: {val_accuracy*100:.1f}%")
    print(f"  - Feat Precision: {feat_precision*100:.1f}%")
    print(f"  - Fix Precision: {fix_precision*100:.1f}%")
    print()
    print(f"Improvement:")
    print(f"  - Accuracy: {(val_accuracy*100 - 75.9):.1f} percentage points")
    print(f"  - Feat Precision: {(feat_precision*100 - 31):.1f} percentage points")
    print(f"  - Fix Precision: {(fix_precision*100 - 36):.1f} percentage points")
    print()

    # Check if we met target
    target_met = feat_precision >= 0.50 and fix_precision >= 0.50
    print(f"Target (>50% precision on feat and fix): {'✓ MET' if target_met else '✗ NOT MET'}")
    print()


if __name__ == '__main__':
    main()
