# Prompt Evaluation Results: run-1768516846919

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 50.0% |
| Scope Inclusion Rate | 50.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 50.0% |
| Average Score | 60.0/100 |
| Avg Generation Time | 7332 ms |

## Type Distribution

| Type | Count |
|------|-------|
| fix | 1 |
| ci | 1 |
| refactor | 1 |
| feat | 2 |

## Component Accuracy

| Metric | Value |
|--------|-------|
| Type Accuracy | 20.0% (1/5) |
| Scope Match (when expected) | 0.0% (0/3) |
| Scope Presence Match | 60.0% |
| Avg Description Similarity | 0.24 |
| Description Similarity Range | 0.18 - 0.29 |

## Sample Results

### Successful Examples

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `feat(forms): add form field binding support`

- Type: expected=docs, got=feat
- Scope: generated=forms (none expected)
- Description Similarity: 0.18
- Score: 100/100

---

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `feat(signals-view): update signal graph visualizer types`

- Type: expected=refactor, got=feat
- Scope: expected=devtools, got=signals-view
- Description Similarity: 0.29
- Score: 90/100

---

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `refactor(signals-visualizer): add snapToRootNode method`

- Type: MATCH (refactor)
- Scope: expected=devtools, got=signals-visualizer
- Description Similarity: 0.21
- Score: 90/100

---

#### angular-849b8401

**Expected:** `build: update github/codeql-action action to v4.31.10`

**Generated:** `ci(scorecard.yml): update workflow`

- Type: expected=build, got=ci
- Scope: generated=scorecard.yml (none expected)
- Description Similarity: 0.22
- Score: 100/100

---

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `fix(docs): update heading transformation to handle nested links`

- Type: expected=docs, got=fix
- Scope: expected=docs-infra, got=docs
- Description Similarity: 0.29
- Score: 90/100

---

### Examples with Issues

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `feat(content/tools/libraries/angular-package-format.md): add note about`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `obj['commit']`

- Type: expected=feat, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-f2cf96b5

**Expected:** `fix(docs-infra): use public folder instead of assets folder for static files in adev tutorials`

**Generated:** `{

"message": "Added new commit for the learn-angular tutorial",
"author": {
"name": "John Doe",
"email": "john.doe@example.com"
},
"date": "2023-07-15T14:30:00Z"
}
`

- Type: expected=fix, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-22fefe78

**Expected:** `build: update bazel dependencies`

**Generated:** `{

"type": "commit",
"message": "Update transitive digests for extensions"
}
`

- Type: expected=build, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `The commit you've provided appears to be a series of changes related to

Here's a brief overview of the changes:

1. MODULE.bazel:
- The URL for fetching the `@
`

- Type: expected=build, got=null
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
