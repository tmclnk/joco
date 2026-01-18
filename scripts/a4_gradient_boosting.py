#!/usr/bin/env python3
"""
Experiment A4: Gradient Boosting (XGBoost + LightGBM) for Commit Type Classification

This script implements gradient boosting approaches to commit type classification using:
- Same TF-IDF vectorization as A1 baseline (file patterns + diff content)
- XGBoost classifier with early stopping
- LightGBM classifier with early stopping
- Hyperparameter tuning for optimal performance

Hypothesis: Gradient boosting can achieve best accuracy by iteratively correcting errors
and handling feature interactions better than linear models or random forests.

Expected outcome: 78-82% validation accuracy (best of classical ML approaches)
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
    """Train XGBoost classifier with early stopping."""
    print("Training XGBoost classifier...")

    # Create DMatrix for efficient training
    dtrain = xgb.DMatrix(X_train_tfidf, label=y_train_encoded)
    dval = xgb.DMatrix(X_val_tfidf, label=y_val_encoded)

    # Hyperparameters tuned for text classification
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

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best validation error: {model.best_score:.4f}")
    print(f"  Training complete!")

    return model


def train_lightgbm(X_train_tfidf, y_train_encoded, X_val_tfidf, y_val_encoded, num_classes: int):
    """Train LightGBM classifier with early stopping."""
    print("Training LightGBM classifier...")

    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train_tfidf, label=y_train_encoded)
    val_data = lgb.Dataset(X_val_tfidf, label=y_val_encoded, reference=train_data)

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

    # Train with early stopping
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'validation'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
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
    """Train and evaluate XGBoost and LightGBM classifiers."""

    print("=" * 80)
    print("Experiment A4: Gradient Boosting (XGBoost + LightGBM)")
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
    print("Building TF-IDF vectorizer (same features as A1)...")
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

    # Comparison to baselines
    print("=" * 80)
    print("COMPARISON TO BASELINES")
    print("=" * 80)
    print()
    print("Model Performance Summary:")
    print(f"{'Model':<25} {'Val Accuracy':<15} {'Notes':<30}")
    print('-' * 70)
    print(f"{'A1: Logistic Regression':<25} {'75.9%':<15} {'Linear baseline':<30}")
    print(f"{'A2: Random Forest':<25} {'TBD':<15} {'Ensemble baseline':<30}")
    print(f"{'A4: XGBoost':<25} {f'{val_acc_xgb*100:.1f}%':<15} {'Gradient boosting':<30}")
    print(f"{'A4: LightGBM':<25} {f'{val_acc_lgb*100:.1f}%':<15} {'Gradient boosting':<30}")
    print()

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Best Model: {best_model}")
    print(f"  - Validation Accuracy: {best_acc*100:.1f}%")
    print(f"  - Validation F1: {best_f1:.3f}")
    print(f"  - Features: TF-IDF (same as A1)")
    print(f"  - Training examples: {len(X_train)}")
    print(f"  - Validation examples: {len(X_val)}")
    print(f"  - Feature dimensions: {X_train_tfidf.shape[1]}")
    print()


if __name__ == '__main__':
    main()
