#!/usr/bin/env python3
"""
Run the Gradient Boosting classifiers (XGBoost + LightGBM) on all benchmark datasets.

This script trains on the full training set and evaluates on each benchmark dataset
to get a comprehensive view of performance across different repos and commit types.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import xgboost as xgb
import lightgbm as lgb

# Import feature extraction from the baseline script
sys.path.insert(0, str(Path(__file__).parent))
from a1_tfidf_baseline import (
    extract_file_paths,
    generate_file_pattern_features,
    extract_diff_content,
    extract_commit_type,
)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL dataset file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_features_benchmark(example: Dict) -> str:
    """
    Extract features from benchmark format.
    Benchmark format: {"diff": "...", "expectedMessage": "...", ...}
    """
    diff_text = example.get('diff', '')

    # Extract file paths and generate pattern features
    file_paths = extract_file_paths(diff_text)
    file_features = generate_file_pattern_features(file_paths)

    # Extract diff content for keyword analysis
    diff_content = extract_diff_content(diff_text)

    # Combine features
    combined = f"{file_features} {diff_content}"
    return combined


def prepare_benchmark_dataset(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """Prepare features (X) and labels (y) from benchmark dataset."""
    X = []
    y = []

    for example in data:
        # Extract features
        features = extract_features_benchmark(example)
        X.append(features)

        # Extract label (commit type) from expectedMessage
        expected_message = example.get('expectedMessage', '')
        commit_type = extract_commit_type(expected_message)
        y.append(commit_type)

    return X, y


def prepare_training_dataset(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """Prepare features from training dataset (supports both formats)."""
    X = []
    y = []

    for example in data:
        # Determine format
        if 'messages' in example:
            # HuggingFace format
            user_message = example['messages'][0]['content']
            diff_start = user_message.find('diff --git')
            if diff_start == -1:
                diff_start = user_message.find('.../') or user_message.find('--- ')
            if diff_start == -1:
                diff_text = user_message
            else:
                diff_text = user_message[diff_start:]
            assistant_message = example['messages'][1]['content']
        else:
            # Simple format
            diff_text = example.get('input', '')
            assistant_message = example.get('output', '')

        # Extract file paths and features
        file_paths = extract_file_paths(diff_text)
        file_features = generate_file_pattern_features(file_paths)
        diff_content = extract_diff_content(diff_text)
        combined = f"{file_features} {diff_content}"

        X.append(combined)
        y.append(extract_commit_type(assistant_message))

    return X, y


def train_best_model(X_train_tfidf, y_train_encoded, num_classes: int):
    """Train LightGBM (best model from A4) with same hyperparameters."""
    print("Training LightGBM classifier...")

    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train_tfidf, label=y_train_encoded)

    # Hyperparameters tuned for text classification
    params = {
        'objective': 'multiclass',
        'num_class': num_classes,
        'metric': 'multi_error',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
        'verbose': -1,
    }

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data],
        valid_names=['train'],
        callbacks=[lgb.log_evaluation(period=0)],
    )

    print(f"  Training complete!")

    return model


def evaluate_on_dataset(model, label_encoder, vectorizer, dataset_name: str, X_test: List[str], y_test: List[str]) -> Dict:
    """Evaluate model on a test dataset and return metrics."""
    # Vectorize test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Predict
    y_pred_proba = model.predict(X_test_tfidf, num_iteration=model.best_iteration)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Get unique labels
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        'dataset': dataset_name,
        'accuracy': accuracy,
        'f1': f1,
        'support': len(y_test),
        'labels': labels,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'true_labels': y_test,
    }


def print_evaluation_results(results: Dict):
    """Print evaluation results for a dataset."""
    print(f"\n{'=' * 80}")
    print(f"Dataset: {results['dataset']}")
    print(f"{'=' * 80}")
    print(f"Samples: {results['support']}")
    print(f"Accuracy: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
    print(f"F1 Score (weighted): {results['f1']:.3f}")
    print()

    # Print per-class metrics
    print("Per-Class Performance:")
    print(f"{'Type':<12} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print('-' * 52)

    report = results['classification_report']
    for label in results['labels']:
        if label in report:
            metrics = report[label]
            print(f"{label:<12} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} "
                  f"{metrics['f1-score']:<10.2f} {int(metrics['support']):<10}")


def main():
    """Train on training set and evaluate on all benchmark datasets."""

    print("=" * 80)
    print("Gradient Boosting (LightGBM): Benchmark Evaluation")
    print("=" * 80)
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / 'dataset'
    benchmark_dir = project_root / 'benchmark'

    # Load training data
    print("Loading training data...")
    train_path = dataset_dir / 'train.jsonl'
    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training examples")

    # Prepare training data
    print("Extracting features from training data...")
    X_train, y_train = prepare_training_dataset(train_data)

    # Print training label distribution
    print("\nTraining set label distribution:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_train, counts_train):
        print(f"  {label}: {count} ({count/len(y_train)*100:.1f}%)")
    print()

    # Build TF-IDF vectorizer
    print("Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"  TF-IDF shape: {X_train_tfidf.shape}")

    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    num_classes = len(label_encoder.classes_)
    print(f"  Number of classes: {num_classes}")
    print()

    # Train model
    print("Training classifier...")
    model = train_best_model(X_train_tfidf, y_train_encoded, num_classes)
    print()

    # Find all benchmark datasets
    benchmark_files = [
        ('Angular (format-correctness)', benchmark_dir / 'format-correctness' / 'angular-commits.jsonl'),
        ('Redis (antirez)', benchmark_dir / 'content-quality' / 'antirez' / 'redis-commits.jsonl'),
        ('Hubris (Bryan Cantrill)', benchmark_dir / 'content-quality' / 'bryan-cantrill' / 'hubris-commits.jsonl'),
        ('Go stdlib (Go team)', benchmark_dir / 'content-quality' / 'go-team' / 'go-commits.jsonl'),
        ('OpenJDK (Java team)', benchmark_dir / 'content-quality' / 'java-team' / 'jdk-commits.jsonl'),
        ('Git (Linus Torvalds)', benchmark_dir / 'content-quality' / 'linus-torvalds' / 'git-commits.jsonl'),
        ('Clojure (Rich Hickey)', benchmark_dir / 'content-quality' / 'rich-hickey' / 'clojure-commits.jsonl'),
    ]

    # Evaluate on each benchmark
    all_results = []
    for name, path in benchmark_files:
        if not path.exists():
            print(f"Skipping {name} (file not found)")
            continue

        print(f"Evaluating on {name}...")
        data = load_jsonl(path)
        X_test, y_test = prepare_benchmark_dataset(data)

        results = evaluate_on_dataset(model, label_encoder, vectorizer, name, X_test, y_test)
        all_results.append(results)
        print_evaluation_results(results)

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS ACROSS ALL BENCHMARKS")
    print("=" * 80)
    print()

    total_samples = sum(r['support'] for r in all_results)
    weighted_accuracy = sum(r['accuracy'] * r['support'] for r in all_results) / total_samples
    weighted_f1 = sum(r['f1'] * r['support'] for r in all_results) / total_samples

    print(f"Total samples: {total_samples}")
    print(f"Weighted average accuracy: {weighted_accuracy:.3f} ({weighted_accuracy*100:.1f}%)")
    print(f"Weighted average F1: {weighted_f1:.3f}")
    print()

    # Summary table
    print("Per-Dataset Summary:")
    print(f"{'Dataset':<35} {'Samples':<10} {'Accuracy':<10} {'F1':<10}")
    print('-' * 65)
    for r in all_results:
        print(f"{r['dataset']:<35} {r['support']:<10} {r['accuracy']:.3f}     {r['f1']:.3f}")
    print()

    # Find best and worst performing types across all datasets
    print("=" * 80)
    print("TYPE PERFORMANCE ACROSS ALL BENCHMARKS")
    print("=" * 80)
    print()

    # Aggregate by type
    type_stats = {}
    for r in all_results:
        for label in r['labels']:
            if label not in r['classification_report']:
                continue
            metrics = r['classification_report'][label]
            support = int(metrics['support'])
            if support == 0:
                continue

            if label not in type_stats:
                type_stats[label] = {'correct': 0, 'total': 0, 'f1_sum': 0, 'f1_count': 0}

            # Calculate correct predictions
            idx = r['labels'].index(label)
            true_labels = np.array(r['true_labels'])
            predictions = np.array(r['predictions'])
            correct = np.sum((true_labels == label) & (predictions == label))

            type_stats[label]['correct'] += correct
            type_stats[label]['total'] += support
            type_stats[label]['f1_sum'] += metrics['f1-score'] * support
            type_stats[label]['f1_count'] += support

    print("Overall Type Accuracy:")
    print(f"{'Type':<12} {'Correct':<10} {'Total':<10} {'Accuracy':<10} {'Avg F1':<10}")
    print('-' * 52)

    for label in sorted(type_stats.keys()):
        stats = type_stats[label]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        avg_f1 = stats['f1_sum'] / stats['f1_count'] if stats['f1_count'] > 0 else 0
        print(f"{label:<12} {stats['correct']:<10} {stats['total']:<10} {acc:.3f}      {avg_f1:.3f}")

    print()
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
