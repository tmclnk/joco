# Prompt Evaluation Results: run-1768581288796

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 50 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 100.0% |
| Scope Inclusion Rate | 76.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 46.0% |
| Average Score | 92.2/100 |
| Avg Generation Time | 4897 ms |

## Type Distribution

| Type | Count |
|------|-------|
| test | 1 |
| fix | 23 |
| docs | 4 |
| build | 8 |
| ci | 2 |
| chore | 4 |
| refactor | 6 |
| feat | 2 |

## Component Accuracy

| Metric | Value |
|--------|-------|
| Type Accuracy | 62.5% (25/40) |
| Scope Match (when expected) | 65.7% (23/35) |
| Scope Presence Match | 90.0% |
| Avg Description Similarity | 0.38 |
| Description Similarity Range | 0.18 - 0.65 |

## Sample Results

### Successful Examples

#### vue-623bfb29

**Expected:** `chore: fix typo`

**Generated:** `docs: fix typo in changelog`

- Type: expected=chore, got=docs
- Description Similarity: 0.52
- Score: 90/100

---

#### vue-aac7e189

**Expected:** `docs: fix typos in comments in the code (#14253)`

**Generated:** `docs: fix typos in comments`

- Type: MATCH (docs)
- Description Similarity: 0.55
- Score: 90/100

---

#### vue-c68bebfa

**Expected:** `release: v3.5.26`

**Generated:** `chore: release v3.5.26`

- Type: expected=release, got=chore
- Score: 90/100

---

#### vue-e857e12c

**Expected:** `fix(runtime-core): pass component instance to flushPreFlushCbs on unmount (#14221)`

**Generated:** `fix: prevent unmount from triggering other app's watchers`

- Type: MATCH (fix)
- Scope: expected=runtime-core, got=none
- Description Similarity: 0.24
- Score: 80/100

---

#### vue-f33b3081

**Expected:** `types(defineProps): avoid never props becoming boolean flags (#14059)`

**Generated:** `fix(types): exclude never props from boolean key detection`

- Type: expected=types, got=fix
- Score: 90/100

---

### Examples with Issues

## Review Instructions

Please analyze these results and provide:
1. **Quality Assessment**: Are the generated messages capturing the essence of the changes?
2. **Common Failure Patterns**: What types of commits is the model struggling with?
3. **Prompt Improvement Suggestions**: What changes to the prompt might improve results?
4. **Type Accuracy**: Is the model choosing appropriate commit types (feat, fix, etc.)?
5. **Scope Detection**: How well is the model identifying the affected component/module?
