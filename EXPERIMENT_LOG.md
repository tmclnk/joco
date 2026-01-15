# Joco Experiment Log

> **Instructions for AI assistants**: When running experiments on prompts, models, or configurations, update this log with results. Use the `/log` skill to add new entries. Keep entries factual and include metrics. This log helps track what has been tried and what worked.

---

## Overview

This log tracks experiments to improve joco's commit message generation quality. The goal is to get small local models (1.5B-3B params) to generate valid conventional commits.

**Key Metrics:**
- **Format Compliance**: % of outputs that are valid conventional commits
- **Type Accuracy**: % where generated type matches expected
- **Score**: Overall quality score (0-100)

---

## Baseline Measurements

### 2026-01-15: Claude vs Ollama Comparison

Tested same 10 Angular commits with both backends.

| Backend | Model | Format Compliance | Type Accuracy | Score |
|---------|-------|-------------------|---------------|-------|
| Claude | claude-opus | 100% | 50% | 92/100 |
| Ollama | qwen2.5-coder:1.5b | 40% | 25% | 49/100 |

**Finding**: Claude follows instructions perfectly. Small Ollama model outputs explanations instead of commits ~60% of the time.

---

## Prompt Engineering Experiments

### Experiment 1: Terse File-Pattern Template

**Date**: 2026-01-15
**Hypothesis**: Shorter, more structured prompt would improve compliance
**Template Style**: Terse with glob patterns (`*.md -> docs`)

```
OUTPUT: type(scope): description
NOTHING ELSE. One line only.

FILE PATTERN -> TYPE:
*.md, docs/, README* -> docs
...
```

**Results**:
- Format Compliance: 10% (worse than baseline)
- Score: 30/100

**Why it failed**: Model copied the template structure into output instead of generating a commit message.

---

### Experiment 2: Explicit Instructions with Wrong Examples

**Date**: 2026-01-15
**Hypothesis**: Showing "WRONG" examples would prevent bad outputs
**Template Style**: Explicit with negative examples

```
WRONG (do not output these):
- "This commit updates..." (explanation)
- Full sentences describing the change
```

**Results**:
- Format Compliance: 50% (+10% vs baseline)
- Score: 56/100

**Partial success**: Reduced explanations but model still confused.

---

### Experiment 3: Sentence Completion Pattern

**Date**: 2026-01-15
**Hypothesis**: "Complete the sentence" format would constrain output
**Template Style**: Few-shot with diff/commit pairs, ending with "Commit:"

```
Diff: README.md changed
Commit: docs: update readme

Now your turn...

Diff:
%s

Commit:
```

**Results**:
- Format Compliance: 50% (same as Exp 2)
- Type Accuracy: 20%
- Score: 60/100 (+11 vs baseline)
- Tokens: 28 (down from 49.6)

**Best prompt so far**: More token-efficient, slightly better scores, but still 50% failure rate.

---

## Key Learnings

### What Works
1. Sentence completion pattern ("Commit:") helps constrain output
2. Explicit file-to-type mapping improves type selection
3. Few-shot examples with actual diffs help
4. Shorter prompts = fewer tokens without quality loss

### What Doesn't Work
1. Terse/abbreviated instructions confuse small models
2. Negative examples ("WRONG: ...") have limited effect
3. Complex type decision trees are ignored
4. Small models (1.5B) fundamentally struggle with instruction-following

### Bottleneck Identified

**Prompt engineering ceiling: ~50% format compliance**

The small model doesn't reliably follow instructions. Outputs include:
- Explanations: "This commit updates..."
- JSON: `{"type": "commit", ...}`
- Template copying: "type(scope): description NOTHING ELSE..."
- Garbage: `obj['commit']`

**Next steps require**:
1. Finetuning on curated dataset (47 examples ready in `dataset/`)
2. Trying larger models (3B, 7B)
3. Using Claude for production (100% compliance)

---

## Test Run Reference

| Run ID | Template | Backend | Format | Score | Notes |
|--------|----------|---------|--------|-------|-------|
| run-1768514842531 | baseline-v1 | claude | 100% | 92 | Gold standard |
| run-1768515314933 | baseline-v1 | ollama | 40% | 49 | Baseline |
| run-1768516637573 | file-pattern-v1 (terse) | ollama | 10% | 30 | Template copying |
| run-1768516747792 | file-pattern-v1 (explicit) | ollama | 50% | 56 | Improved |
| run-1768516846919 | file-pattern-v1 (completion) | ollama | 50% | 60 | Best prompt |

---

## Future Experiments to Try

- [ ] Lower temperature (0.3-0.5) for more deterministic output
- [ ] Larger model (qwen2.5-coder:3b or 7b)
- [ ] System prompt vs user prompt separation
- [ ] JSON mode / structured output
- [ ] Finetuned model on curated dataset
- [ ] Different base models (DeepSeek-Coder, CodeLlama)
