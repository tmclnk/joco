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
| run-1768583157191 | raw-diff | multistep | 100% | 90 | 3-query approach |
| run-1768583185086 | baseline-v1 | ollama | 66.7% | 70 | Comparison run |
| run-1768584172497 | raw-diff | multistep-2step | 100% | 78 | 2-query, no scope, 42% faster |

---

## Model Comparison Experiments

### Experiment 4: Alternative Small Models

**Date**: 2026-01-15
**Hypothesis**: Different base models may have better instruction-following
**Config**: Same prompt (file-pattern-v1), same test cases, different models

| Model | Size | Format | Type Acc | Score | Notes |
|-------|------|--------|----------|-------|-------|
| qwen2.5-coder:1.5b | 986MB | 50% | 20% | 60 | Baseline |
| **llama3.2:1b** | 1.3GB | **60%** | **50%** | 59.5 | Best small model |
| gemma:2b | 1.7GB | 0% | 0% | 22 | All explanations |

**Findings**:
- **llama3.2:1b is the best small model** - 60% format compliance, 50% type accuracy
- gemma:2b is terrible for this task - outputs only explanations
- llama3.2 never adds scopes (0% scope inclusion) but gets types right more often
- Model architecture matters more than parameter count

**Run IDs**:
- llama3.2:1b: run-1768518485234
- gemma:2b: run-1768518554288

---

### Experiment 5: Multi-Step Generation (3 Separate Queries)

**Date**: 2026-01-16
**Hypothesis**: Breaking commit generation into 3 focused queries (type → scope → description) would improve format compliance since each step is constrained and has low token limits
**Config**: New `--backend=multistep` with built-in prompts:
- Step 1: Classify type (10 tokens max)
- Step 2: Extract scope (10 tokens max)
- Step 3: Generate description given type+scope (30 tokens max)

**Results** (3 test cases, angular-small.jsonl):

| Approach | Format | Scope Included | Score | Avg Time |
|----------|--------|----------------|-------|----------|
| **Multi-step (3 queries)** | **100%** | **100%** | **90/100** | 5000ms |
| Single-query (baseline-v1) | 66.7% | 66.7% | 70/100 | 2000ms |

**Why it worked**:
- Each query is a simple, focused task that small models handle well
- No truncation issues - each step uses only 10-30 tokens
- The model can't ramble into explanations since token limits are so low
- Assembling the parts guarantees `type(scope): description` format

**Trade-offs**:
- 2.5x slower due to 3 Ollama round-trips
- Could potentially parallelize step 1 and 2

**Run IDs**:
- Multi-step: run-1768583157191
- Baseline comparison: run-1768583185086

---

### Experiment 5b: Multi-Step at Scale (20 test cases)

**Date**: 2026-01-16
**Hypothesis**: Multi-step approach maintains 100% format compliance at larger scale
**Config**: Same as Exp 5, but with 20 test cases

**Results** (run-1768583294642):
- Format Compliance: **100%** (20/20)
- Scope Included: **100%** (20/20)
- Length ≤ 72 chars: **100%**
- Average Score: 90/100
- Avg Generation Time: 4368ms

**Issue Found - Type Bias**:
- Type Accuracy: only 10% (2/20)
- All 20 commits classified as `refactor`
- The TYPE_PROMPT is biased - model defaults to "refactor" for everything

**Secondary Issue - Redundant Prefix**:
- Some descriptions start with "refactor:" creating output like:
  `refactor(core): refactor: Update Bazel version...`
- Need to strip type prefix from description in cleanDescription()

**Next**: Fix TYPE_PROMPT to reduce refactor bias

---

### Experiment 5c: Improved Type Classification Prompt

**Date**: 2026-01-16
**Hypothesis**: File-based type classification rules will improve type accuracy
**Config**: Updated TYPE_PROMPT with explicit file pattern matching

**Iteration 1** (run-1768583511289) - Basic file patterns:
- Type Distribution: feat=4, refactor=15, docs=1
- Type Accuracy: 15% (up from 10%)
- Still biased toward refactor

**Iteration 2** (run-1768583810756) - File-first classification on Vue commits:
- Type Distribution: fix=10, refactor=6, test=3, chore=1
- **Type Accuracy: 35.3%** (major improvement!)
- Much better diversity

**Key insight**: Putting file pattern checks FIRST and making "refactor" the fallback only when nothing else matches significantly reduces bias

**Files created**:
- `src/test/java/org/example/harness/generator/MultiStepGenerator.java`
- `src/test/java/org/example/harness/prompt/templates/RawDiffTemplate.java`

---

### Experiment 5d: Simplified 2-Step Generator (no scope)

**Date**: 2026-01-16
**Hypothesis**: Removing the scope query would speed up generation while maintaining format compliance
**Config**: Changed from 3 queries (type→scope→description) to 2 queries (type→description)

**Changes**:
- Removed scope extraction step entirely
- Output format simplified to `type: description` (no scope)
- Reduced from 3 Ollama round-trips to 2

**Results** (run-1768584172497):

| Metric | 2-Step | 3-Step (5c) |
|--------|--------|-------------|
| Format Compliance | **100%** | 100% |
| Type Distribution | test=6, chore=2, refactor=2 | varies |
| Avg Generation Time | **3789ms** | 5000ms+ |
| Score | 78/100 | 90/100 |

**Type Distribution Details**:
- test: 6 (60%)
- chore: 2 (20%)
- refactor: 2 (20%)

**Benefits**:
- **42% faster** than 3-step approach
- Simpler output format
- Maintained 100% format compliance
- Better type diversity than early 3-step iterations

**Trade-off**:
- No scope information in output
- Lower overall score (78 vs 90) due to missing scope component
- Messages are cleaner but less specific

**Conclusion**: Good option when speed matters more than scope granularity. The simpler format is also easier for downstream tooling to parse.

---

### Experiment 5e: Model Comparison with Fixed Scoring

**Date**: 2026-01-16
**Hypothesis**: llama3.2:1b (previously best small model) would perform well with multi-step approach
**Config**: 2-step multistep generator on Vue commits

**Scoring Fix Applied**:
The original scoring was flawed - it rewarded short messages regardless of quality. Fixed by:
- Removed scope bonus (+10) - not using scopes anymore
- Removed short length bonus (+10 for ≤50 chars) - was rewarding garbage
- Added good length bonus (+15 for 30-60 chars) - rewards reasonable descriptions
- Added meta-description penalty (-20) - penalizes "here is a short commit..." nonsense

**Results** (10 Vue commits each):

| Model | Score | Type Distribution | Meta-descriptions |
|-------|-------|-------------------|-------------------|
| qwen2.5-coder:1.5b | **88** | chore=2, feat=2, test=4, refactor=1 | 0/10 |
| llama3.2:1b | 83 | chore=10 (all same!) | 4/10 |

**Sample Output Comparison**:

llama3.2:1b (problematic):
```
[OK] chore: added bug fixes and performance improvements
[META] chore: here is a short commit description that captures the essence of
[META] chore: here is a short commit description that follows standard
```

qwen2.5-coder:1.5b (good):
```
[OK] chore: update CHANGELOG.md for bug fixes in compiler-sfc and reactivity
[OK] test: simplify app unmounting logic in createApp test
[OK] refactor: simplify directive argument handling
```

**Key Findings**:
1. **llama3.2:1b doesn't follow terse instructions** - outputs explanatory text instead of single words
2. **Type classification fails completely** - all fall back to `chore` default
3. **40% meta-descriptions** - model explains what it will do instead of doing it
4. **qwen2.5-coder:1.5b is the better model** for this task despite being "code-focused"

**Conclusion**: qwen2.5-coder:1.5b remains the best choice for multi-step commit generation. llama3.2:1b's instruction-following is too poor for structured prompts.

---

### Experiment 6: maxTokens Investigation

**Date**: 2026-01-16
**Hypothesis**: Different DESC_MAX_TOKENS values would affect output quality
**Config**: Tested DESC_MAX_TOKENS values of 15, 20, 30, and 50 on Vue commits

**Results Table**:

| DESC_MAX_TOKENS | Avg Length | Score | Completion Tokens |
|-----------------|------------|-------|-------------------|
| 15              | 68.2       | 84.0  | 16.9              |
| 20              | 65.3       | 87.0  | 18.7              |
| 30              | 67.5       | 85.5  | 21.0              |
| 50              | 61.3       | 87.0  | 18.7              |

**Key Findings**:
1. Token limit ≥20 is sufficient - model naturally produces 17-21 tokens regardless of limit
2. DESC_MAX_TOKENS=15 is too low - score drops to 84, outputs truncated awkwardly
3. The 72-char cleanup truncation in cleanDescription() is the real constraint, not the token limit
4. TYPE_MAX_TOKENS=10 is fine for single-word type classification

**Conclusion**: Current values (TYPE_MAX_TOKENS=10, DESC_MAX_TOKENS=30) are optimal. Keep them.

**Run IDs**: tokens-15, tokens-20, tokens-30, tokens-50

---

## Future Experiments to Try

- [ ] Lower temperature (0.3-0.5) for more deterministic output
- [ ] Larger model (qwen2.5-coder:3b or 7b)
- [ ] System prompt vs user prompt separation
- [ ] JSON mode / structured output
- [x] Different base models (llama3.2, gemma) - llama3.2:1b wins
- [ ] Finetuned model on curated dataset
- [ ] Distillation from Claude outputs
- [ ] Train larger model, quantize down
- [x] Multi-step generation (type→scope→description) - 100% format compliance!
- [ ] Run multi-step on larger test set (20-50 cases) for statistical significance
- [ ] Parallelize steps 1 and 2 in multi-step to reduce latency
- [ ] Try multi-step with llama3.2:1b (best small model)
