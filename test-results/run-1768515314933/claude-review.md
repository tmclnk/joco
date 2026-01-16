# Prompt Evaluation Results: run-1768515314933

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 40.0% |
| Scope Inclusion Rate | 40.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 10.0% |
| Average Score | 49.0/100 |
| Avg Generation Time | 7921 ms |

## Type Distribution

| Type | Count |
|------|-------|
| fix | 1 |
| build | 2 |
| refactor | 1 |

## Component Accuracy

| Metric | Value |
|--------|-------|
| Type Accuracy | 25.0% (1/4) |
| Scope Match (when expected) | 0.0% (0/2) |
| Scope Presence Match | 50.0% |
| Avg Description Similarity | 0.37 |
| Description Similarity Range | 0.21 - 0.65 |

## Sample Results

### Successful Examples

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `fix(signals-visualizer): optimize rendering and snap to root node`

- Type: expected=refactor, got=fix
- Scope: expected=devtools, got=signals-visualizer
- Description Similarity: 0.42
- Score: 90/100

---

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `build(deps): update angular-package-format dependency to latest version`

- Type: expected=docs, got=build
- Scope: generated=deps (none expected)
- Description Similarity: 0.21
- Score: 90/100

---

#### angular-849b8401

**Expected:** `build: update github/codeql-action action to v4.31.10`

**Generated:** `build(deps): update codeql-action to version 4.31.10`

- Type: MATCH (build)
- Scope: generated=deps (none expected)
- Description Similarity: 0.65
- Score: 90/100

---

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `refactor(docs): improve rendering of `docs-card` and link`

- Type: expected=docs, got=refactor
- Scope: expected=docs-infra, got=docs
- Description Similarity: 0.23
- Score: 90/100

---

### Examples with Issues

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `docs:introduction/essentials/signal-forms.md`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `fix(signals-view/signals-visualizer): update dagre-d3-types to match`

- Type: expected=refactor, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `The provided code snippet is a part of the Signal Forms library, which

In this context, the `FormFieldBindingOptions` interface is used to
define additional options that can be passed to the `registerAsBinding`
method of the `FormField` directive. These options include a `focus`
method that defines how the control should be focused when requested
`

- Type: expected=feat, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-f2cf96b5

**Expected:** `fix(docs-infra): use public folder instead of assets folder for static files in adev tutorials`

**Generated:** `The provided code snippets demonstrate various aspects of Angular

1. Image Optimization with `NgOptimizedImage` Directive

Description: The `NgOptimizedImage` directive is used in Angular
applications to dynamically load images based on user interaction or
conditions. This helps in improving the performance of the application
by reducing the number of HTTP requests.

Usage Example:
<img ng
`

- Type: expected=fix, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-22fefe78

**Expected:** `build: update bazel dependencies`

**Generated:** `The changes made to the `package.json` file and the `extensions.bzl`

Changes in `package.json`
- Changes for TypeScript:
- The version of `aspectrulests` is updated from `3.8.1` to `3.8.3`.
- The digest for the `bzlTransitiveDigest
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `The provided patch updates the version of `@angular/ng-dev` from`

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
