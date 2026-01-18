# B1: GitHub Conventional Commit Data Mining Statistics

## Overview

Successfully mined 1,561 new conventional commit examples from 9 popular open-source repositories known for using conventional commits.

## Dataset Growth

- **Original Dataset**: 290 examples (261 train, 29 validation)
- **Extended Dataset**: 1,484 examples (1,276 train, 208 validation)
- **Growth**: +1,194 examples (412% increase)

## Mining Results by Repository

| Repository | Target | Actual | Success Rate | Commits Processed | Notes |
|-----------|--------|--------|--------------|-------------------|-------|
| Vue.js | 200 | 200 | 100% | 308 | Excellent conventional commit adoption |
| Electron | 200 | 200 | 100% | 284 | Very consistent format |
| Babel | 200 | 200 | 100% | 1,762 | Lower adoption rate (many non-conventional) |
| ESLint | 200 | 200 | 100% | 283 | High quality examples |
| Jest | 200 | 200 | 100% | 486 | Good mix of types |
| Webpack | 200 | 200 | 100% | 349 | Excellent examples |
| TypeScript | 200 | 108 | 54% | 2,051 | Very low adoption rate |
| Redux | 150 | 103 | 69% | 3,191 | Low adoption rate |
| Next.js | 150 | 150 | 100% | 1,509 | Good documentation commits |
| **Total** | **1,700** | **1,561** | **92%** | **10,223** | |

## Quality Filters Applied

1. **Conventional Commit Format**: `^(feat|fix|docs|test|ci|chore|refactor|perf|build|style)(\(.+\))?:`
2. **Diff Size**: 50-8,000 characters
3. **Class Balancing**: Maximum 300 examples per commit type

## Raw Data Distribution (Before Balancing)

| Commit Type | Count | Percentage |
|------------|-------|------------|
| fix | 601 | 38.5% |
| chore | 366 | 23.4% |
| docs | 266 | 17.0% |
| build | 72 | 4.6% |
| feat | 68 | 4.4% |
| ci | 64 | 4.1% |
| refactor | 56 | 3.6% |
| test | 46 | 2.9% |
| perf | 21 | 1.3% |
| style | 1 | 0.1% |

## Balanced Dataset Distribution

After applying max 300 examples per type and train/validation split:

### Combined Training Set (1,276 examples)

| Commit Type | Count | Percentage |
|------------|-------|------------|
| fix | 315 | 24.7% |
| docs | 293 | 23.0% |
| chore | 265 | 20.8% |
| refactor | 96 | 7.5% |
| build | 78 | 6.1% |
| feat | 72 | 5.6% |
| ci | 70 | 5.5% |
| test | 68 | 5.3% |
| perf | 18 | 1.4% |
| style | 1 | 0.1% |

### Combined Validation Set (208 examples)

Similar distribution maintained through random sampling with 15% validation split.

## Repository-Specific Type Distributions

### Vue.js (200 examples)
- chore: 55, fix: 108, refactor: 4, test: 5, perf: 1, feat: 14, ci: 4, build: 9

### Electron (200 examples)
- fix: 50, build: 57, ci: 16, chore: 15, refactor: 14, feat: 9, docs: 34, test: 4, perf: 1

### Babel (200 examples)
- chore: 44, fix: 125, docs: 4, perf: 7, test: 6, ci: 5, build: 4, feat: 3, refactor: 2

### ESLint (200 examples)
- test: 11, docs: 62, fix: 36, chore: 58, refactor: 10, ci: 14, build: 2, feat: 7

### Jest (200 examples)
- fix: 56, chore: 104, docs: 19, feat: 19, refactor: 2

### Webpack (200 examples)
- chore: 71, fix: 49, refactor: 21, docs: 16, ci: 22, test: 13, feat: 4, perf: 4

### TypeScript (108 examples)
- fix: 98, perf: 2, docs: 1, feat: 6, refactor: 1

### Redux (103 examples)
- docs: 51, chore: 5, fix: 38, refactor: 2, feat: 5, test: 1, style: 1

### Next.js (150 examples)
- fix: 41, docs: 79, perf: 6, ci: 3, feat: 1, chore: 14, test: 6

## Filtering Statistics

- **Total Commits Processed**: 10,223
- **Skipped (format)**: ~6,959 (68%) - Did not match conventional commit pattern
- **Skipped (size)**: ~703 (7%) - Diff too small (<50 chars) or too large (>8000 chars)
- **Accepted**: 1,561 (15%)

## Dataset Quality Improvements

1. **Increased Diversity**: Examples from 9 different codebases with varied coding styles
2. **Better Type Balance**: Capped at 300 examples per type to prevent over-representation
3. **Real-World Examples**: All commits from production repositories
4. **Size Constraints**: All diffs between 50-8,000 characters for optimal training

## File Outputs

- `/workspaces/joco/dataset/train_extended.jsonl` - 1,276 examples (4.4MB)
- `/workspaces/joco/dataset/validation_extended.jsonl` - 208 examples (677KB)

## Expected Impact

- **Target**: 5-15% accuracy improvement in commit type classification
- **Rationale**: Larger dataset (4x growth) with more diverse examples should improve model generalization
- **Next Steps**: Train models on extended dataset and compare performance against baseline

## Notes

- TypeScript and Redux had lower conventional commit adoption rates
- "fix" and "chore" are the most common commit types
- "style" commits are extremely rare (only 1 example found)
- Git clone approach with shallow depth (2000 commits) worked well
- No GitHub API token required - used direct git operations
- Total mining time: ~10-15 minutes

## Script Location

`/workspaces/joco/scripts/b1_mine_github_data.py`
