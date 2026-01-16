# Prompt Evaluation Results: run-1768516747792

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 50.0% |
| Scope Inclusion Rate | 20.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 40.0% |
| Average Score | 56.0/100 |
| Avg Generation Time | 7689 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 1 |
| build | 2 |
| ci | 2 |

## Component Accuracy

| Metric | Value |
|--------|-------|
| Type Accuracy | 20.0% (1/5) |
| Scope Match (when expected) | 0.0% (0/3) |
| Scope Presence Match | 80.0% |
| Avg Description Similarity | 0.18 |
| Description Similarity Range | 0.14 - 0.25 |

## Sample Results

### Successful Examples

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `ci: update signals-visualizer build dependencies`

- Type: expected=refactor, got=ci
- Scope: expected=devtools, got=none
- Description Similarity: 0.14
- Score: 90/100

---

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `build(deps): update dagre dependency`

- Type: expected=refactor, got=build
- Scope: expected=devtools, got=deps
- Description Similarity: 0.16
- Score: 100/100

---

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `docs: update angular-package-format.md`

- Type: MATCH (docs)
- Description Similarity: 0.15
- Score: 90/100

---

#### angular-849b8401

**Expected:** `build: update github/codeql-action action to v4.31.10`

**Generated:** `ci: update scorecard.yml`

- Type: expected=build, got=ci
- Description Similarity: 0.25
- Score: 90/100

---

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `build(deps): update marked library to version 15.10.2`

- Type: expected=docs, got=build
- Scope: expected=docs-infra, got=deps
- Description Similarity: 0.20
- Score: 90/100

---

### Examples with Issues

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `docs(introduction/essentials/signal-forms.md): update documentation for`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `This commit adds a new method to the `FormField` class called

The `registerAsBinding` method takes an optional `bindingOptions`
parameter that allows you to specify additional options for the binding.
The `focus
`

- Type: expected=feat, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-f2cf96b5

**Expected:** `fix(docs-infra): use public folder instead of assets folder for static files in adev tutorials`

**Generated:** `The commit message provides information about optimizing images in

Here's a breakdown of the content:

- Introduction: The message starts by stating that the optimization
process is crucial for improving the performance of web applications,
especially those with images.
- Usage of NgOptimizedImage Directive: It explains that the
`NgOptimizedImage` directive can be
`

- Type: expected=fix, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-22fefe78

**Expected:** `build: update bazel dependencies`

**Generated:** `The commit message appears to be a description of changes made to the`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `Merge pull request #396 from modelcontextprotocol/fix-dev-deps

Fix dev dependencies to use the latest versions.
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
