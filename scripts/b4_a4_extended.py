#!/usr/bin/env python3
"""
Experiment B4: Re-train A4 (XGBoost/LightGBM) on Extended Dataset

This script re-trains the gradient boosting models from A4 on the extended dataset
(1,276 training examples vs original 261).

Original A4 Results:
- XGBoost: 72.4% validation (91.6% train, +19.2% overfitting)
- LightGBM: 72.4% validation (82.0% train, +9.6% overfitting)
- Both UNDERPERFORMED A1 logistic regression (75.9% validation)

Hypothesis: With 5x more training data (1,276 examples), gradient boosting should:
1. Achieve target 78-82% validation accuracy
2. Reduce overfitting to <6%
3. Finally outperform simple logistic regression

This tests if model complexity helps when data is sufficient.
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
            data.append(json.loads(line.strip()))
    return data


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
        diff_start = user_message.find('diff --git')
        if diff_start == -1:
            diff_start = user_message.find('.../') or user_message.find('--- ')
        if diff_start == -1:
            diff_text = user_message
        else:
            diff_text = user_message[diff_start:]
    else:
        # Simple format
        diff_text = example.get('input', '')

    # Extract file paths and generate pattern features
    file_paths = extract_file_paths(diff_text)
    file_features = generate_file_pattern_features(file_paths)

    # Extract diff content for keyword analysis
    diff_content = extract_diff_content(diff_text)

    # Combine features
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
            assistant_message = example['messages'][1]['content']
        else:
            assistant_message = example.get('output', '')

        commit_type = extract_commit_type(assistant_message)
        y.append(commit_type)

    return X, y


def train_xgboost(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes: int):
    """Train XGBoost classifier with early stopping (SAME params as A4)."""
    print("Training XGBoost classifier...")

    # Create DMatrix for efficient training
    dtrain = xgb.DMatrix(X_train_tfidf, label=y_train_encoded)
    dval = xgb.DMatrix(X_val_tfidf, label=y_val_encoded)

    # EXACT SAME hyperparameters as A4 (no changes)
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

    # Train with early stopping (30 rounds as requested, was 20 in A4)
    evals = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best validation error: {model.best_score:.4f}")
    print(f"  Training complete!")

    return model


def train_lightgbm(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes: int):
    """Train LightGBM classifier with early stopping (SAME params as A4)."""
    print("Training LightGBM classifier...")

    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train_tfidf, label=y_train_encoded)
    val_data = lgb.Dataset(X_val_tfidf, label=y_val_encoded, reference=train_data)

    # EXACT SAME hyperparameters as A4 (no changes)
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

    # Train with early stopping (30 rounds as requested, was 20 in A4)
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

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best validation error: {evals_result['validation']['multi_error'][model.best_iteration-1]:.4f}")
    print(f"  Training complete!")

    return model


def evaluate_model(model, model_name: str, X_tfidf, y_encoded, y_true, label_encoder, is_xgboost: bool = True):
    """Evaluate a trained model and return predictions and metrics."""
    if is_xgboost:
        dmatrix = xgb.DMatrix(X_tfidf)
        y_pred_encoded = model.predict(dmatrix).astype(int)
    else:
        # LightGBM returns probabilities by default, need to get class predictions
        y_pred_proba = model.predict(X_tfidf, num_iteration=model.best_iteration)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)

    # Decode predictions back to original labels
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    return y_pred, accuracy, f1


def main():
    """Train and evaluate XGBoost and LightGBM classifiers on extended dataset."""

    print("=" * 80)
    print("Experiment B4: Re-train A4 (XGBoost + LightGBM) on Extended Dataset")
    print("=" * 80)
    print()
    print("Original A4 (261 training examples):")
    print("  XGBoost:  72.4% validation (91.6% train, +19.2% overfitting)")
    print("  LightGBM: 72.4% validation (82.0% train, +9.6% overfitting)")
    print()
    print("Expected B4 (1,276 training examples):")
    print("  Target: 78-82% validation with <6% overfitting")
    print("  Test: Can gradient boosting finally beat simple LogReg?")
    print()

    # Load EXTENDED datasets
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_path = dataset_dir / 'train_extended.jsonl'
    val_path = dataset_dir / 'validation_extended.jsonl'

    print(f"Loading training data from: {train_path}")
    train_data = load_jsonl(train_path)
    print(f"  Loaded {len(train_data)} training examples (5x original 261)")

    print(f"Loading validation data from: {val_path}")
    val_data = load_jsonl(val_path)
    print(f"  Loaded {len(val_data)} validation examples (7x original 29)")
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

    # TF-IDF Vectorization (same as A1)
    print("Building TF-IDF vectorizer (same features as A1/A4)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=1000,    # Limit feature space
        min_df=2,             # Ignore terms that appear in < 2 documents
        max_df=0.95,          # Ignore terms that appear in > 95% of documents
        sublinear_tf=True,    # Use sublinear TF scaling (log)
    )

    print("Vectorizing training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"  TF-IDF shape: {X_train_tfidf.shape}")

    print("Vectorizing validation data...")
    X_val_tfidf = vectorizer.transform(X_val)
    print(f"  TF-IDF shape: {X_val_tfidf.shape}")
    print()

    # Encode labels to integers (required for XGBoost and LightGBM)
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    num_classes = len(label_encoder.classes_)
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {list(label_encoder.classes_)}")
    print()

    # Train XGBoost
    print("=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)
    print()
    xgb_model = train_xgboost(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes)
    print()

    # Train LightGBM
    print("=" * 80)
    print("TRAINING LIGHTGBM")
    print("=" * 80)
    print()
    lgb_model = train_lightgbm(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes)
    print()

    # Evaluate XGBoost
    print("=" * 80)
    print("XGBOOST RESULTS")
    print("=" * 80)
    print()

    # Training set
    y_train_pred_xgb, train_acc_xgb, train_f1_xgb = evaluate_model(
        xgb_model, "XGBoost", X_train_tfidf, y_train_encoded, y_train, label_encoder, is_xgboost=True
    )

    # Validation set
    y_val_pred_xgb, val_acc_xgb, val_f1_xgb = evaluate_model(
        xgb_model, "XGBoost", X_val_tfidf, y_val_encoded, y_val, label_encoder, is_xgboost=True
    )

    print(f"Training Set Performance:")
    print(f"  Accuracy: {train_acc_xgb:.3f} ({train_acc_xgb*100:.1f}%)")
    print(f"  F1 Score (weighted): {train_f1_xgb:.3f}")
    print()
    print(f"Validation Set Performance:")
    print(f"  Accuracy: {val_acc_xgb:.3f} ({val_acc_xgb*100:.1f}%)")
    print(f"  F1 Score (weighted): {val_f1_xgb:.3f}")
    print()

    print("Classification Report (Validation Set):")
    print(classification_report(y_val, y_val_pred_xgb, zero_division=0))

    # Evaluate LightGBM
    print("=" * 80)
    print("LIGHTGBM RESULTS")
    print("=" * 80)
    print()

    # Training set
    y_train_pred_lgb, train_acc_lgb, train_f1_lgb = evaluate_model(
        lgb_model, "LightGBM", X_train_tfidf, y_train_encoded, y_train, label_encoder, is_xgboost=False
    )

    # Validation set
    y_val_pred_lgb, val_acc_lgb, val_f1_lgb = evaluate_model(
        lgb_model, "LightGBM", X_val_tfidf, y_val_encoded, y_val, label_encoder, is_xgboost=False
    )

    print(f"Training Set Performance:")
    print(f"  Accuracy: {train_acc_lgb:.3f} ({train_acc_lgb*100:.1f}%)")
    print(f"  F1 Score (weighted): {train_f1_lgb:.3f}")
    print()
    print(f"Validation Set Performance:")
    print(f"  Accuracy: {val_acc_lgb:.3f} ({val_acc_lgb*100:.1f}%)")
    print(f"  F1 Score (weighted): {val_f1_lgb:.3f}")
    print()

    print("Classification Report (Validation Set):")
    print(classification_report(y_val, y_val_pred_lgb, zero_division=0))

    # Compare models
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Model':<15} {'Train Acc':<12} {'Val Acc':<12} {'Val F1':<12} {'Overfitting':<12}")
    print('-' * 63)

    overfit_xgb = train_acc_xgb - val_acc_xgb
    overfit_lgb = train_acc_lgb - val_acc_lgb

    print(f"{'XGBoost':<15} {train_acc_xgb:.3f}        {val_acc_xgb:.3f}        {val_f1_xgb:.3f}        {overfit_xgb:+.3f}")
    print(f"{'LightGBM':<15} {train_acc_lgb:.3f}        {val_acc_lgb:.3f}        {val_f1_lgb:.3f}        {overfit_lgb:+.3f}")
    print()

    # Determine best model
    if val_acc_xgb > val_acc_lgb:
        best_model = "XGBoost"
        best_acc = val_acc_xgb
        best_f1 = val_f1_xgb
        best_predictions = y_val_pred_xgb
    else:
        best_model = "LightGBM"
        best_acc = val_acc_lgb
        best_f1 = val_f1_lgb
        best_predictions = y_val_pred_lgb

    print(f"Best Model: {best_model}")
    print(f"  Validation Accuracy: {best_acc:.3f} ({best_acc*100:.1f}%)")
    print(f"  Validation F1: {best_f1:.3f}")
    print()

    # Confusion matrix for best model
    print("=" * 80)
    print(f"CONFUSION MATRIX ({best_model} - Validation Set)")
    print("=" * 80)
    print()
    labels = sorted(set(y_val) | set(best_predictions))
    cm = confusion_matrix(y_val, best_predictions, labels=labels)

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

    # Comparison to original A4 and baselines
    print("=" * 80)
    print("COMPARISON: B4 vs A4 (Original) vs Baselines")
    print("=" * 80)
    print()
    print("Extended Dataset Performance (B4):")
    print(f"  Training examples: {len(X_train)} (5x original)")
    print(f"  Validation examples: {len(X_val)} (7x original)")
    print()
    print(f"{'Model':<30} {'Train':<12} {'Val Acc':<12} {'Val F1':<12} {'Overfit':<12}")
    print('-' * 78)
    print(f"{'B4: XGBoost (extended)':<30} {train_acc_xgb*100:.1f}%        {val_acc_xgb*100:.1f}%        {val_f1_xgb:.3f}        {overfit_xgb*100:+.1f}%")
    print(f"{'B4: LightGBM (extended)':<30} {train_acc_lgb*100:.1f}%        {val_acc_lgb*100:.1f}%        {val_f1_lgb:.3f}        {overfit_lgb*100:+.1f}%")
    print()
    print("Original A4 Performance (261 training examples):")
    print(f"{'A4: XGBoost (original)':<30} {'91.6%':<12} {'72.4%':<12} {'0.672':<12} {'+19.2%':<12}")
    print(f"{'A4: LightGBM (original)':<30} {'82.0%':<12} {'72.4%':<12} {'0.664':<12} {'+9.6%':<12}")
    print()
    print("Baseline Comparison:")
    print(f"{'A1: Logistic Regression':<30} {'87.0%':<12} {'75.9%':<12} {'0.768':<12} {'+11.1%':<12}")
    print()

    # Calculate improvements
    print("=" * 80)
    print("KEY METRICS: Did Extended Data Fix Gradient Boosting?")
    print("=" * 80)
    print()

    xgb_val_improvement = (val_acc_xgb - 0.724) * 100
    lgb_val_improvement = (val_acc_lgb - 0.724) * 100
    xgb_overfit_improvement = (19.2 - overfit_xgb * 100)
    lgb_overfit_improvement = (9.6 - overfit_lgb * 100)

    print(f"XGBoost Improvements:")
    print(f"  Validation accuracy: {val_acc_xgb*100:.1f}% (A4: 72.4%) = {xgb_val_improvement:+.1f}%")
    print(f"  Overfitting reduction: {overfit_xgb*100:.1f}% (A4: 19.2%) = {xgb_overfit_improvement:+.1f}%")
    print(f"  Target met: {'✅ YES' if val_acc_xgb >= 0.78 and overfit_xgb <= 0.06 else '❌ NO'}")
    print()

    print(f"LightGBM Improvements:")
    print(f"  Validation accuracy: {val_acc_lgb*100:.1f}% (A4: 72.4%) = {lgb_val_improvement:+.1f}%")
    print(f"  Overfitting reduction: {overfit_lgb*100:.1f}% (A4: 9.6%) = {lgb_overfit_improvement:+.1f}%")
    print(f"  Target met: {'✅ YES' if val_acc_lgb >= 0.78 and overfit_lgb <= 0.06 else '❌ NO'}")
    print()

    # Compare to LogReg baseline (A1)
    best_beats_logreg = best_acc > 0.759
    print(f"Best Model vs A1 Logistic Regression:")
    print(f"  {best_model}: {best_acc*100:.1f}% vs A1: 75.9%")
    print(f"  Gradient boosting beats LogReg: {'✅ YES' if best_beats_logreg else '❌ NO'}")
    print()

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Best Model: {best_model}")
    print(f"  - Validation Accuracy: {best_acc*100:.1f}%")
    print(f"  - Validation F1: {best_f1:.3f}")
    print(f"  - Training examples: {len(X_train)} (5x A4)")
    print(f"  - Validation examples: {len(X_val)} (7x A4)")
    print(f"  - Overfitting: {(train_acc_lgb if best_model == 'LightGBM' else train_acc_xgb) - best_acc:.1%}")
    print()
    print("Next Steps:")
    print("  - Run on Angular benchmark to compare to B2 results")
    print("  - Use /log skill to document findings in EXPERIMENT_LOG.md")
    print("  - Update beads issue joco-lz4 with results")
    print()


if __name__ == '__main__':
    main()
