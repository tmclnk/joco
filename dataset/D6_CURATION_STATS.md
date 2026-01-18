# D6: Dataset Curation Statistics

## Overview

Successfully curated a high-quality fine-tuning dataset from the extended dataset (1,276 examples).

## Input Dataset

- **Source**: `dataset/train_extended.jsonl`
- **Total Examples**: 1,276

## Curation Criteria Applied

### 1. Format Quality
- Valid conventional commit format: `type(scope): description` or `type: description`
- Valid commit types: feat, fix, docs, style, refactor, test, chore, ci, build, perf
- Description under 72 characters
- No trailing periods
- Proper case (lowercase description)

### 2. Diff Quality
- Character range: 50-5,000 characters
- File count: < 50 files changed
- Contains actual code changes (not just whitespace)

### 3. Semantic Clarity
- Not overly vague (filters out descriptions like "update README" with no context)
- Descriptive and meaningful
- Type-appropriate for the changes

### 4. Class Balance
- Maximum 200 examples per commit type
- Ensures balanced representation for training

## Results

### Filtering Results

| Metric | Count | Percentage |
|--------|-------|------------|
| Total Input | 1,276 | 100.0% |
| Passed Quality Filters | 942 | 73.8% |
| Rejected | 334 | 26.2% |
| After Balancing | 875 | 68.6% of input |

### Top Rejection Reasons

1. **No diff** (23 examples) - Missing or empty diff content
2. **Vague descriptions** (16 examples) - e.g., "Update README" with no context
3. **Description too long** (36+ examples) - Over 72 characters in various lengths
4. **No actual changes** (4 examples) - Only whitespace or formatting

### Type Distribution

#### Before Balancing (942 examples)
| Type | Count |
|------|-------|
| docs | 232 |
| fix | 220 |
| chore | 215 |
| ci | 66 |
| build | 61 |
| refactor | 53 |
| test | 47 |
| feat | 35 |
| perf | 12 |
| style | 1 |

#### After Balancing (875 examples)
| Type | Count |
|------|-------|
| docs | 200 |
| fix | 200 |
| chore | 200 |
| ci | 66 |
| build | 61 |
| refactor | 53 |
| test | 47 |
| feat | 35 |
| perf | 12 |
| style | 1 |

**Note**: The top 3 types (docs, fix, chore) were capped at 200 examples each. Other types had fewer than 200 high-quality examples available.

## Output Datasets

### Final Split (70% train, 15% val, 15% test)

| Dataset | File | Examples | Percentage |
|---------|------|----------|------------|
| Training | `train_curated.jsonl` | 612 | 69.9% |
| Validation | `val_curated.jsonl` | 131 | 15.0% |
| Test | `test_curated.jsonl` | 132 | 15.1% |
| **Total** | | **875** | **100.0%** |

### File Sizes

- `train_curated.jsonl`: 1.4 MB
- `val_curated.jsonl`: 302 KB
- `test_curated.jsonl`: 325 KB

## Quality Examples

### Sample Curated Examples

1. **Type: test**
   - Message: `test: test case for prefixIdentifiers w/ bindings`
   - Diff: 1,477 chars, 1 file

2. **Type: chore**
   - Message: `chore(deps): bump tar from 7.5.2 to 7.5.3 (#20322)`
   - Diff: 832 chars, 1 file

3. **Type: fix**
   - Message: `fix(cgo): remove doc field to prevent binary artifacts`
   - Diff: 684 chars, 1 file

4. **Type: docs**
   - Message: `docs: add a note that cache components is opt-in near the top (#85245)`
   - Diff: ~500 chars, 1 file

5. **Type: build**
   - Message: `build: use extended tsconfig`
   - Diff: ~3,000 chars, 5 files

## Key Improvements Over Extended Dataset

1. **Higher Quality**: Removed 26.2% of examples that didn't meet quality standards
2. **Better Balance**: Top 3 types capped at 200 examples each (down from 215-232)
3. **Format Consistency**: All examples follow strict conventional commit format
4. **Semantic Clarity**: Removed vague or ambiguous commit messages
5. **Optimal Diff Size**: All diffs in 50-5,000 character range for meaningful context

## Recommendations for Future Work

1. **Augment Underrepresented Types**: Consider mining more examples for `feat`, `test`, `perf`, and `style` types
2. **Fine-tune Class Balance**: Current balance is reasonable but could target exactly 100-150 per type
3. **Evaluate Model Performance**: Use this curated dataset to train models and compare against extended dataset
4. **Iterative Refinement**: Based on model errors, refine curation criteria further

## Curation Script

- **Location**: `scripts/d6_curate_dataset.py`
- **Reproducibility**: Uses `random.seed(42)` for consistent splits
- **Runtime**: ~1-2 seconds for full curation pipeline
