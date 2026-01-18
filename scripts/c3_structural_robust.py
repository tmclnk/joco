#!/usr/bin/env python3
"""
Experiment C3: Robust Structural Feature Classifier with Anti-Overfitting Measures

This script enhances C2 (69.0% validation accuracy) by applying rigorous anti-overfitting
techniques to ensure production-ready performance and reliable generalization.

CONTEXT: C2 showed -13% "overfitting" gap (train: 56%, val: 69%), indicating UNDERFITTING.
This is unusual and suggests the model has room to fit the training data better while
maintaining or improving validation performance.

HYPOTHESIS: With proper validation methodology and regularization tuning, we can:
1. Increase training accuracy (currently only 56%)
2. Maintain or improve validation accuracy (currently 69%)
3. Get more reliable performance estimates via cross-validation
4. Achieve 70-75% accuracy with low variance

Anti-Overfitting Measures:

A. 5-FOLD CROSS-VALIDATION (MOST IMPORTANT):
   - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   - Every example used for validation exactly once
   - Reports mean ± std accuracy across folds
   - More reliable than single train-val split
   - Shows variance in performance

B. REGULARIZATION TUNING (C parameter):
   - Try C values: [0.01, 0.1, 0.5, 1.0, 10.0]
   - GridSearchCV to find optimal C
   - Current: C=1.0 (default)
   - Lower C = stronger regularization (may be too strong given underfitting)
   - Higher C = weaker regularization (may help model fit better)

C. THREE-WAY SPLIT (60/20/20):
   - 60% train / 20% validation / 20% test
   - Test set NEVER used during training/tuning
   - Provides unbiased final evaluation
   - Validation used for hyperparameter tuning only

D. FEATURE IMPORTANCE ANALYSIS:
   - Rank features by coefficient magnitude
   - Try removing weakest features (keep top 10 out of 15)
   - Test if simpler model generalizes better
   - Identify most predictive structural patterns

E. DEPLOYMENT CRITERIA:
   - Train-val gap < 10% (currently -13%, so PASS but inverted)
   - CV std < 5% (ensures consistency)
   - Val ≈ test (within 3%, ensures no overfitting to val set)
   - Only deploy if all criteria met

Expected Results:
- Cross-validation: 70-75% ± 3%
- Training accuracy: 65-70% (up from 56%)
- Validation accuracy: 70-75% (up from 69%)
- Test accuracy: 70-75% (within 3% of validation)
- Low variance across folds (std < 5%)

Comparison to C2:
- C2: 56% train / 69% val (-13% gap = underfitting)
- C3 target: 70% train / 72% val (±2% gap = good fit)
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
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    train_test_split,
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
    Same as C2 implementation.
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

    # Calculate file type features (15)
    # Code file extensions
    code_extensions = {'.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.c', '.cpp',
                      '.h', '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.cs',
                      '.scala', '.sh', '.bash', '.vue', '.html', '.css', '.scss'}

    code_files = 0
    for path in file_paths:
        ext = Path(path).suffix.lower()
        if ext in code_extensions:
            code_files += 1

    features['pct_code_files'] = code_files / total_files if total_files > 0 else 0

    return features


def extract_diff_from_user_message(user_message: str) -> str:
    """Extract the git diff portion from the user message."""
    lines = user_message.split('\n')

    # Find where the diff starts
    diff_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('diff --git'):
            diff_start = i
            break
        elif ' | ' in line and any(c.isdigit() for c in line):
            for j in range(i, len(lines)):
                if lines[j].strip().startswith('diff --git'):
                    diff_start = j
                    break
            if diff_start > 0:
                break

    if diff_start > 0:
        return '\n'.join(lines[diff_start:])

    return user_message


def prepare_data(data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix X and labels y from dataset.
    Same as C2 implementation.
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
    """Train robust structural feature classifier with anti-overfitting measures."""

    print("=" * 80)
    print("C3: Robust Structural Feature Classifier with Anti-Overfitting Measures")
    print("=" * 80)
    print("\nBUILDING ON C2 RESULTS:")
    print("  C2 Training:   56.0%")
    print("  C2 Validation: 69.0%")
    print("  Overfitting:   -13.0% (UNDERFITTING!)")
    print("\nC3 IMPROVEMENTS:")
    print("  A. 5-fold cross-validation for reliable estimates")
    print("  B. GridSearchCV to tune regularization (C parameter)")
    print("  C. Three-way split (60/20/20) with held-out test set")
    print("  D. Feature importance analysis and selection")
    print("  E. Deployment criteria validation")

    # Define paths
    dataset_dir = Path(__file__).parent.parent / 'dataset'
    train_file = dataset_dir / 'train.jsonl'
    val_file = dataset_dir / 'validation.jsonl'

    print(f"\n>> Loading datasets...")
    print(f"   Train: {train_file}")
    print(f"   Val:   {val_file}")

    # Load data
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)

    print(f"\n>> Loaded {len(train_data)} training examples")
    print(f">> Loaded {len(val_data)} validation examples")

    # Prepare features
    print("\n>> Extracting structural features...")
    X_train, y_train = prepare_data(train_data)
    X_val, y_val = prepare_data(val_data)

    print(f"\n>> Feature matrix shapes:")
    print(f"   Training:   {X_train.shape[0]} examples x {X_train.shape[1]} features")
    print(f"   Validation: {X_val.shape[0]} examples x {X_val.shape[1]} features")

    # Combine for three-way split
    print("\n>> Creating three-way split (60% train / 20% val / 20% test)...")
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    # First split: 80% (train+val) / 20% (test)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
    )

    # Second split: 75% of trainval = train (60% of total), 25% of trainval = val (20% of total)
    X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    print(f"\n>> Three-way split created:")
    print(f"   Train: {len(y_train_new)} examples ({len(y_train_new)/len(y_combined)*100:.1f}%)")
    print(f"   Val:   {len(y_val_new)} examples ({len(y_val_new)/len(y_combined)*100:.1f}%)")
    print(f"   Test:  {len(y_test)} examples ({len(y_test)/len(y_combined)*100:.1f}%)")

    # Print class distributions
    print_class_distribution(y_train_new, "Training")
    print_class_distribution(y_val_new, "Validation")
    print_class_distribution(y_test, "Test")

    # Feature names
    feature_names = [
        'new_files', 'modified_files', 'deleted_files',
        'insertions', 'deletions', 'net_lines', 'insert_delete_ratio',
        'hunk_count', 'avg_hunk_size', 'max_hunk_size', 'min_hunk_size',
        'lines_per_file', 'hunks_per_file', 'fragmentation_score',
        'pct_code_files',
    ]

    # ========================================================================
    # A. 5-FOLD CROSS-VALIDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("A. 5-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print("\nUsing StratifiedKFold to ensure balanced class distribution in each fold.")
    print("This provides more reliable performance estimates than a single split.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_trainval, y_trainval), 1):
        X_fold_train = X_trainval[train_idx]
        y_fold_train = y_trainval[train_idx]
        X_fold_val = X_trainval[val_idx]
        y_fold_val = y_trainval[val_idx]

        # Train pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        pipeline.fit(X_fold_train, y_fold_train)

        # Evaluate
        train_acc = pipeline.score(X_fold_train, y_fold_train)
        val_acc = pipeline.score(X_fold_val, y_fold_val)
        cv_scores.append(val_acc)

        fold_results.append({
            'fold': fold_idx,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': train_acc - val_acc,
        })

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {train_acc:.1%}  Val: {val_acc:.1%}  Gap: {train_acc - val_acc:+.1%}")

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    print(f"\n>> Cross-Validation Results:")
    print(f"   Mean accuracy: {cv_mean:.1%}")
    print(f"   Std deviation: {cv_std:.1%} ({cv_std:.4f})")
    print(f"   95% CI: [{cv_mean - 1.96*cv_std:.1%}, {cv_mean + 1.96*cv_std:.1%}]")
    print(f"\n>> CV Criteria: Std < 5%  {'PASS' if cv_std < 0.05 else 'FAIL'}")

    # ========================================================================
    # B. REGULARIZATION TUNING (GridSearchCV)
    # ========================================================================
    print("\n" + "=" * 80)
    print("B. REGULARIZATION TUNING (GridSearchCV)")
    print("=" * 80)
    print("\nTesting C values: [0.01, 0.1, 0.5, 1.0, 10.0]")
    print("Lower C = stronger regularization, Higher C = weaker regularization")
    print("C2 used default C=1.0, which may be suboptimal given underfitting.")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    param_grid = {
        'classifier__C': [0.01, 0.1, 0.5, 1.0, 10.0],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    print("\n>> Running GridSearchCV (this may take a minute)...")
    grid_search.fit(X_trainval, y_trainval)

    print(f"\n>> Best parameters: {grid_search.best_params_}")
    print(f">> Best CV score: {grid_search.best_score_:.1%}")

    print("\n>> All tested configurations:")
    print(f"{'C':<10} {'Mean CV Accuracy':<20} {'Std':<10}")
    print("-" * 40)
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        c_val = params['classifier__C']
        print(f"{c_val:<10} {mean_score:<20.1%} {std_score:<10.4f}")

    best_pipeline = grid_search.best_estimator_

    # ========================================================================
    # C. THREE-WAY SPLIT EVALUATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("C. THREE-WAY SPLIT EVALUATION")
    print("=" * 80)
    print("\nTraining final model with best hyperparameters on train set.")
    print("Validating on validation set (used for tuning).")
    print("Testing on held-out test set (NEVER seen during training/tuning).")

    # Train on train set
    print("\n>> Training on train set...")
    best_pipeline.fit(X_train_new, y_train_new)

    # Evaluate on all three sets
    train_acc = best_pipeline.score(X_train_new, y_train_new)
    val_acc = best_pipeline.score(X_val_new, y_val_new)
    test_acc = best_pipeline.score(X_test, y_test)

    print(f"\n>> Results:")
    print(f"   Training:   {train_acc:.1%}")
    print(f"   Validation: {val_acc:.1%}")
    print(f"   Test:       {test_acc:.1%}")
    print(f"\n>> Gaps:")
    print(f"   Train-Val:  {train_acc - val_acc:+.1%}")
    print(f"   Val-Test:   {val_acc - test_acc:+.1%}")

    # Detailed metrics
    print("\n>> Training Set Classification Report:")
    y_train_pred = best_pipeline.predict(X_train_new)
    print(classification_report(y_train_new, y_train_pred, zero_division=0))

    print("\n>> Validation Set Classification Report:")
    y_val_pred = best_pipeline.predict(X_val_new)
    print(classification_report(y_val_new, y_val_pred, zero_division=0))

    print("\n>> Test Set Classification Report:")
    y_test_pred = best_pipeline.predict(X_test)
    test_report = classification_report(y_test, y_test_pred, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # Confusion matrix for test set
    print("\n>> Test Set Confusion Matrix:")
    cm_test = confusion_matrix(y_test, y_test_pred)
    labels_test = sorted(list(set(y_test)))
    print(f"{'':>12}", end='')
    for label in labels_test:
        print(f"{label:>8}", end='')
    print()
    for i, label in enumerate(labels_test):
        print(f"{label:>12}", end='')
        for j in range(len(labels_test)):
            print(f"{cm_test[i, j]:>8}", end='')
        print()

    # ========================================================================
    # D. FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("D. FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    classifier = best_pipeline.named_steps['classifier']
    coefs = np.abs(classifier.coef_)
    mean_importance = np.mean(coefs, axis=0)

    feature_importance = sorted(
        zip(feature_names, mean_importance),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n>> Feature Importance Ranking:")
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12}")
    print("-" * 45)
    for rank, (name, importance) in enumerate(feature_importance, 1):
        print(f"{rank:<6} {name:<25} {importance:<12.3f}")

    # Test reduced feature set (top 10)
    print("\n>> Testing reduced feature set (top 10 features)...")
    top_10_features = [name for name, _ in feature_importance[:10]]
    top_10_indices = [feature_names.index(name) for name in top_10_features]

    X_train_reduced = X_train_new[:, top_10_indices]
    X_val_reduced = X_val_new[:, top_10_indices]
    X_test_reduced = X_test[:, top_10_indices]

    pipeline_reduced = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=grid_search.best_params_['classifier__C'],
            max_iter=1000,
            random_state=42
        ))
    ])
    pipeline_reduced.fit(X_train_reduced, y_train_new)

    reduced_train_acc = pipeline_reduced.score(X_train_reduced, y_train_new)
    reduced_val_acc = pipeline_reduced.score(X_val_reduced, y_val_new)
    reduced_test_acc = pipeline_reduced.score(X_test_reduced, y_test)

    print(f"\n>> Reduced model (10 features) results:")
    print(f"   Training:   {reduced_train_acc:.1%} (full: {train_acc:.1%})")
    print(f"   Validation: {reduced_val_acc:.1%} (full: {val_acc:.1%})")
    print(f"   Test:       {reduced_test_acc:.1%} (full: {test_acc:.1%})")

    # ========================================================================
    # E. DEPLOYMENT CRITERIA
    # ========================================================================
    print("\n" + "=" * 80)
    print("E. DEPLOYMENT CRITERIA")
    print("=" * 80)

    train_val_gap = abs(train_acc - val_acc)
    val_test_gap = abs(val_acc - test_acc)

    criteria = [
        ("Train-Val gap < 10%", train_val_gap < 0.10, f"{train_val_gap:.1%}"),
        ("CV std < 5%", cv_std < 0.05, f"{cv_std:.1%}"),
        ("Val-Test gap < 3%", val_test_gap < 0.03, f"{val_test_gap:.1%}"),
    ]

    print("\n>> Deployment Criteria:")
    all_pass = True
    for criterion, passed, value in criteria:
        status = "PASS" if passed else "FAIL"
        print(f"   [{status}] {criterion:<25} (actual: {value})")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n>> DEPLOYMENT: APPROVED")
        print("   All criteria met. Model is production-ready.")
    else:
        print("\n>> DEPLOYMENT: NOT APPROVED")
        print("   Some criteria not met. Further tuning recommended.")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n>> C2 (Baseline) Results:")
    print("   Training:   56.0%")
    print("   Validation: 69.0%")
    print("   Gap:        -13.0% (underfitting)")

    print(f"\n>> C3 (Robust) Results:")
    print(f"   Training:   {train_acc:.1%}")
    print(f"   Validation: {val_acc:.1%}")
    print(f"   Test:       {test_acc:.1%}")
    print(f"   Train-Val:  {train_acc - val_acc:+.1%}")
    print(f"   Val-Test:   {val_acc - test_acc:+.1%}")

    print(f"\n>> Cross-Validation:")
    print(f"   Mean: {cv_mean:.1%} ± {cv_std:.1%}")
    print(f"   Range: {min(cv_scores):.1%} - {max(cv_scores):.1%}")

    print(f"\n>> Best Hyperparameters:")
    print(f"   C = {grid_search.best_params_['classifier__C']}")

    print(f"\n>> Improvements over C2:")
    print(f"   Training accuracy:   {train_acc - 0.56:+.1%}")
    print(f"   Validation accuracy: {val_acc - 0.69:+.1%}")
    print(f"   More reliable estimates via CV")
    print(f"   Unbiased test set evaluation: {test_acc:.1%}")

    # Final verdict
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    target_accuracy = 0.70
    accuracy_met = test_acc >= target_accuracy
    criteria_met = all_pass

    print(f"\n>> Target: Test accuracy ≥ 70%")
    print(f"   Actual: {test_acc:.1%}  {'PASS' if accuracy_met else 'FAIL'}")

    if accuracy_met and criteria_met:
        print("\n>> SUCCESS!")
        print("   C3 achieves robust performance with proper validation methodology.")
        print("   Model is production-ready for commit message classification.")
    elif accuracy_met:
        print("\n>> PARTIAL SUCCESS")
        print("   Accuracy target met, but some deployment criteria not satisfied.")
        print("   Consider additional tuning or data collection.")
    else:
        print("\n>> NEEDS IMPROVEMENT")
        print("   Target accuracy not reached with current approach.")
        print("   Consider: More data, additional features, or ensemble methods.")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    train_and_evaluate()
