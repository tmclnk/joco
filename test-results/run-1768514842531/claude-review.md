# Prompt Evaluation Results: run-1768514842531

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 100.0% |
| Scope Inclusion Rate | 70.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 50.0% |
| Average Score | 92.0/100 |
| Avg Generation Time | 4372 ms |

## Type Distribution

| Type | Count |
|------|-------|
| fix | 2 |
| docs | 2 |
| build | 1 |
| ci | 2 |
| feat | 1 |
| refactor | 2 |

## Component Accuracy

| Metric | Value |
|--------|-------|
| Type Accuracy | 50.0% (5/10) |
| Scope Match (when expected) | 40.0% (2/5) |
| Scope Presence Match | 80.0% |
| Avg Description Similarity | 0.40 |
| Description Similarity Range | 0.20 - 0.93 |

## Sample Results

### Successful Examples

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `docs(forms): rename field directive to formField`

- Type: MATCH (docs)
- Scope: generated=forms (none expected)
- Description Similarity: 0.32
- Score: 100/100

---

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `refactor(signals-visualizer): rename dagre types and simplify type`

- Type: MATCH (refactor)
- Scope: expected=devtools, got=signals-visualizer
- Description Similarity: 0.37
- Score: 90/100

---

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `fix(devtools): improve signals visualizer viewport handling`

- Type: expected=refactor, got=fix
- Scope: MATCH (devtools)
- Description Similarity: 0.20
- Score: 90/100

---

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `docs: clarify FESM may depend on shared chunks`

- Type: MATCH (docs)
- Description Similarity: 0.93
- Score: 90/100

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `feat(forms): add manual registration for FormField bindings with custom`

- Type: MATCH (feat)
- Scope: MATCH (forms)
- Description Similarity: 0.30
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
