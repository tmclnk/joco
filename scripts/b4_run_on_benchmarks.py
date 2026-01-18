#!/usr/bin/env python3
"""
Experiment B4: Angular Benchmark Evaluation for Extended XGBoost/LightGBM

Runs the B4 models (trained on 1,276 examples) on Angular benchmark to compare
against B2 results and original A4 results.
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
    f1_score,
)
import xgboost as xgb
import lightgbm as lgb

# Import feature extraction
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
            data.append(json.loads(line.strip()))
    return data


def extract_features(example: Dict) -> str:
    """Extract features from example."""
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

    file_paths = extract_file_paths(diff_text)
    file_features = generate_file_pattern_features(file_paths)
    diff_content = extract_diff_content(diff_text)
    combined = f"{file_features} {diff_content}"

    return combined


def prepare_dataset(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """Prepare features (X) and labels (y) from dataset."""
    X = []
    y = []

    for example in data:
        features = extract_features(example)
        X.append(features)

        if 'messages' in example:
            assistant_message = example['messages'][1]['content']
        else:
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


def train_xgboost(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes: int):
    """Train XGBoost classifier."""
    dtrain = xgb.DMatrix(X_train_tfidf, label=y_train_encoded)
    dval = xgb.DMatrix(X_val_tfidf, label=y_val_encoded)

    params = {
        'objective': 'multi:softmax',
        'num_class': num_classes,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': 'merror',
        'seed': 42,
        'tree_method': 'hist',
    }

    evals = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    return model


def train_lightgbm(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes: int):
    """Train LightGBM classifier."""
    train_data = lgb.Dataset(X_train_tfidf, label=y_train_encoded)
    val_data = lgb.Dataset(X_val_tfidf, label=y_val_encoded, reference=train_data)

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

    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'validation'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )

    return model


def evaluate_model(model, X_tfidf, y_true, label_encoder, is_xgboost: bool = True):
    """Evaluate a trained model."""
    if is_xgboost:
        dmatrix = xgb.DMatrix(X_tfidf)
        y_pred_encoded = model.predict(dmatrix).astype(int)
    else:
        y_pred_proba = model.predict(X_tfidf, num_iteration=model.best_iteration)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)

    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return y_pred, accuracy, f1


def main():
    """Run B4 models on Angular benchmark."""

    print("=" * 80)
    print("Experiment B4: Angular Benchmark Evaluation")
    print("=" * 80)
    print()

    # Load extended training data
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_path = dataset_dir / 'train_extended.jsonl'
    val_path = dataset_dir / 'validation_extended.jsonl'

    print(f"Loading extended training data...")
    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training examples")

    print(f"Loading extended validation data...")
    val_data = load_jsonl(val_path)
    print(f"  Loaded {len(val_data)} validation examples")
    print()

    # Prepare training data
    X_train, y_train = prepare_dataset(train_data)
    X_val, y_val = prepare_dataset(val_data)

    # TF-IDF Vectorization
    print("Building TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    num_classes = len(label_encoder.classes_)
    print(f"  Classes: {list(label_encoder.classes_)}")
    print()

    # Train models
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes)
    print("Training LightGBM...")
    lgb_model = train_lightgbm(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes)
    print()

    # Load Angular benchmark
    benchmark_path = Path(__file__).parent.parent / 'benchmark' / 'format-correctness' / 'angular-commits.jsonl'
    print(f"Loading Angular benchmark from: {benchmark_path}")
    benchmark_data = load_jsonl(benchmark_path)
    print(f"  Loaded {len(benchmark_data)} benchmark examples")
    print()

    # Prepare benchmark data
    X_bench, y_bench = prepare_angular_benchmark(benchmark_data)
    X_bench_tfidf = vectorizer.transform(X_bench)

    # Evaluate on Angular
    print("=" * 80)
    print("ANGULAR BENCHMARK RESULTS")
    print("=" * 80)
    print()

    # XGBoost
    y_pred_xgb, acc_xgb, f1_xgb = evaluate_model(xgb_model, X_bench_tfidf, y_bench, label_encoder, is_xgboost=True)

    print("XGBoost Results:")
    print(f"  Accuracy: {acc_xgb:.3f} ({acc_xgb*100:.1f}%)")
    print(f"  F1 Score (weighted): {f1_xgb:.3f}")
    print()
    print("Classification Report:")
    print(classification_report(y_bench, y_pred_xgb, zero_division=0))

    # LightGBM
    y_pred_lgb, acc_lgb, f1_lgb = evaluate_model(lgb_model, X_bench_tfidf, y_bench, label_encoder, is_xgboost=False)

    print("LightGBM Results:")
    print(f"  Accuracy: {acc_lgb:.3f} ({acc_lgb*100:.1f}%)")
    print(f"  F1 Score (weighted): {f1_lgb:.3f}")
    print()
    print("Classification Report:")
    print(classification_report(y_bench, y_pred_lgb, zero_division=0))

    # Comparison
    print("=" * 80)
    print("COMPARISON TO BASELINES")
    print("=" * 80)
    print()
    print(f"{'Model':<35} {'Angular Acc':<15} {'Angular F1':<15}")
    print('-' * 65)
    print(f"{'B4: XGBoost (1,276 train)':<35} {acc_xgb*100:.1f}%           {f1_xgb:.3f}")
    print(f"{'B4: LightGBM (1,276 train)':<35} {acc_lgb*100:.1f}%           {f1_lgb:.3f}")
    print()
    print("Previous Results:")
    print(f"{'A4: XGBoost (261 train)':<35} {'66.0%':<15} {'0.671':<15}")
    print(f"{'A4: LightGBM (261 train)':<35} {'66.0%':<15} {'0.671':<15}")
    print(f"{'A1: Logistic Regression (261)':<35} {'61.0%':<15} {'0.639':<15}")
    print()

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
