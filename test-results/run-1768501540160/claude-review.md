# Prompt Evaluation Results: run-1768501540160

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 15 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 93.3% |
| Scope Inclusion Rate | 93.3% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 53.3% |
| Average Score | 90.7/100 |
| Avg Generation Time | 1832 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 1 |
| fix | 6 |
| build | 7 |

## Sample Results

### Successful Examples

#### angular-d12e6f09

**Expected:** `docs: release notes for the v21.0.9 release`

**Generated:** `fix(CHANGELOG.md): update release notes for v21.0.9`

- Type: expected=docs, got=fix
- Score: 90/100

---

#### angular-3f13db87

**Expected:** `ci: update algolia synonym updater to use node directly rather than tsx`

**Generated:** `build(deps): update algoliasearch to 4.23.0`

- Type: expected=ci, got=build
- Score: 100/100

---

#### angular-c9f584b9

**Expected:** `docs: add IMPORTANT prefix to pipe usage warning and`

**Generated:** `build(deps): update typescript to 5.0`

- Type: expected=docs, got=build
- Score: 100/100

---

#### angular-074666b2

**Expected:** `docs(docs-infra): use Signals Forms focusBoundControl`

**Generated:** `fix(core): update search dialog component to handle focus bound control`

- Type: expected=docs, got=fix
- Score: 90/100

---

#### angular-551f1205

**Expected:** `build: update dependency bazel to v8.5.1`

**Generated:** `build(deps): update typescript to 5.1`

- Type: MATCH
- Score: 100/100

---

### Examples with Issues

#### angular-724e917f

**Expected:** `docs(docs-infra): Exempts animation-related symbols from linking`

**Generated:** `docs(shared-docs/pipeline/shared/linking.mts): update linking logic`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

## Review Instructions

Please analyze these results and provide:
1. **Quality Assessment**: Are the generated messages capturing the essence of the changes?
2. **Common Failure Patterns**: What types of commits is the model struggling with?
3. **Prompt Improvement Suggestions**: What changes to the prompt might improve results?
4. **Type Accuracy**: Is the model choosing appropriate commit types (feat, fix, etc.)?
5. **Scope Detection**: How well is the model identifying the affected component/module?
