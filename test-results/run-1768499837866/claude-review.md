# Prompt Evaluation Results: run-1768499837866

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 5 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 40.0% |
| Scope Inclusion Rate | 40.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 60.0% |
| Average Score | 54.0/100 |
| Avg Generation Time | 6867 ms |

## Type Distribution

| Type | Count |
|------|-------|
| fix | 1 |
| feat | 1 |

## Sample Results

### Successful Examples

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `feat(docs-card): update docs-card extension`

- Type: expected=docs, got=feat
- Score: 100/100

---

#### angular-d12e6f09

**Expected:** `docs: release notes for the v21.0.9 release`

**Generated:** `fix(changelog): update changelog for version 21.0.9`

- Type: expected=docs, got=fix
- Score: 90/100

---

### Examples with Issues

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `{

"version": "1.0.0",
"dependencies": {
"@actions/core": "^2.0.1",
"@google-cloud/spanner": 8.0.0(supports-color@10.2.2)
}
}
`

- Type: expected=build, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-01592689

**Expected:** `docs: release notes for the v21.1.0 release`

**Generated:** `{

"version": "20.3.14",
"changelog": [
{
"date": "2025-11-25",
"description": "Resolved JIT resources in `createApplication`."
},
{
"date": "2025-11-25",
"description": "Added `context` to `createApplication`."
},
{
"date": "2
`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-3f13db87

**Expected:** `ci: update algolia synonym updater to use node directly rather than tsx`

**Generated:** `fix(scripts/synonyms): update tsconfig.json for algolia synonym script`

- Type: expected=ci, got=null
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
