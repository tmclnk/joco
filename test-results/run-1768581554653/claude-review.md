# Prompt Evaluation Results: run-1768581554653

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 50 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 100.0% |
| Scope Inclusion Rate | 8.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 76.0% |
| Average Score | 88.4/100 |
| Avg Generation Time | 4345 ms |

## Type Distribution

| Type | Count |
|------|-------|
| fix | 2 |
| docs | 39 |
| ci | 6 |
| refactor | 1 |
| feat | 2 |

## Sample Results

### Successful Examples

#### fastapi-a456e92a

**Expected:** `📝 Update release notes`

**Generated:** `docs: add contribution instructions for LLM generated code`

- Type: expected=?, got=docs
- Score: 80/100

---

#### fastapi-1be80f48

**Expected:** `📝 Add contribution instructions about LLM generated code and comments and automated tools for PRs (#14706)`

**Generated:** `docs: add AI contribution policy and automated PR handling`

- Type: expected=📝 Add contribution instructions about LLM generated code and comments and automated tools for PRs , got=docs
- Score: 80/100

---

#### fastapi-e63f382b

**Expected:** `📝 Update release notes`

**Generated:** `docs: update release notes`

- Type: expected=?, got=docs
- Score: 90/100

---

#### fastapi-7b864acf

**Expected:** `📝 Update docs for management tasks (#14705)`

**Generated:** `docs: simplify management tasks documentation`

- Type: expected=📝 Update docs for management tasks , got=docs
- Score: 90/100

---

#### fastapi-e9e0419e

**Expected:** `📝 Update release notes`

**Generated:** `docs: update release notes`

- Type: expected=?, got=docs
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
