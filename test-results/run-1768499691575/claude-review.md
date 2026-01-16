# Prompt Evaluation Results: run-1768499691575

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 5 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 60.0% |
| Scope Inclusion Rate | 60.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 20.0% |
| Average Score | 64.0/100 |
| Avg Generation Time | 7634 ms |

## Type Distribution

| Type | Count |
|------|-------|
| feat | 2 |
| refactor | 1 |

## Sample Results

### Successful Examples

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `refactor(docs-card): update heading transformation to handle nested`

- Type: expected=docs, got=refactor
- Score: 90/100

---

#### angular-d12e6f09

**Expected:** `docs: release notes for the v21.0.9 release`

**Generated:** `feat(changelog): add migration guide for common-to-standalone refactor`

- Type: expected=docs, got=feat
- Score: 90/100

---

#### angular-3f13db87

**Expected:** `ci: update algolia synonym updater to use node directly rather than tsx`

**Generated:** `feat(synonyms): add TypeScript configuration and update synonym script`

- Type: expected=ci, got=feat
- Score: 90/100

---

### Examples with Issues

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `The provided patch updates the `package.json` file to use a different`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-01592689

**Expected:** `docs: release notes for the v21.1.0 release`

**Generated:** `{

"title": "Angular 21.x Release Notes",
"version": "21.0.3",
"description": "Announcement of Angular 21, including new features and
deprecations.",
"release_notes": [
{
"type": "feature",
"message": "Add support for dynamic `[field]` bindings in FormUiControl
signals."
},
{
"type": "fix",
"message":
`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

## Review Instructions

Please analyze these results and provide:
1. **Quality Assessment**: Are the generated messages capturing the essence of the changes?
2. **Common Failure Patterns**: What types of commits is the model struggling with?
3. **Prompt Improvement Suggestions**: What changes to the prompt might improve results?
4. **Type Accuracy**: Is the model choosing appropriate commit types (feat, fix, etc.)?
5. **Scope Detection**: How well is the model identifying the affected component/module?
