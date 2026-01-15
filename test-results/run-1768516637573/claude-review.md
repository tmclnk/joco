# Prompt Evaluation Results: run-1768516637573

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 10.0% |
| Scope Inclusion Rate | 0.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 50.0% |
| Average Score | 30.0/100 |
| Avg Generation Time | 8383 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 1 |

## Component Accuracy

| Metric | Value |
|--------|-------|
| Type Accuracy | 0.0% (0/1) |
| Scope Match (when expected) | 0.0% (0/1) |
| Scope Presence Match | 0.0% |
| Avg Description Similarity | 0.19 |
| Description Similarity Range | 0.19 - 0.19 |

## Sample Results

### Successful Examples

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `docs: README.md -> docs

ci: .github/workflows/ci.yml -> ci
build(deps): update lodash to 4.17.21 -> build(deps)
test/, _test., .spec. -> test
.config., .env, .rc -> chore

feat(signals-view/signals-visualizer/BUILD.bazel): add
signals-view/signals-visualizer.ts
fix(signals-view/signals-visualizer/BUILD.bazel
`

- Type: expected=refactor, got=docs
- Scope: expected=devtools, got=none
- Description Similarity: 0.19
- Score: 80/100

---

### Examples with Issues

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `type(scope): description

NOTHING ELSE. One line only.

FILE PATTERN -> TYPE:
.md, docs/, README -> docs
.github/, .yml, ci/* -> ci
package.json, pom.xml, *.lock, deps/ -> build
test/, _test., .spec. -> test
.config., .env, .rc -> chore

CODE CHANGES -> TYPE:
New export/function/class -> feat
Error handling, null check -> fix
Rename,
`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `code changes`

- Type: expected=refactor, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `type(scope): description

FILE PATTERN -> TYPE:
.md, docs/, README -> docs
.github/, .yml, ci/* -> ci
package.json, pom.xml, *.lock, deps/ -> build
test/, _test., .spec. -> test
.config., .env, .rc -> chore
`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `The provided code snippet is a TypeScript file for testing the

Here's a breakdown of the main components and functionalities:

1. FormField Directive: This component manages the state of a form field
and its bindings. It provides methods to focus on the host element,
register as a binding, and manage disabled
`

- Type: expected=feat, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-f2cf96b5

**Expected:** `fix(docs-infra): use public folder instead of assets folder for static files in adev tutorials`

**Generated:** `The provided code snippets are for various Angular projects, including

1. Image Optimization Example in Learn Angular Tutorial

This example demonstrates how to optimize images using
`NgOptimizedImage` in Angular.

HTML:
<!-- user.component.html -->
<ul>
<li>
Static Image:
<img ngSrc="/assets/logo.svg" alt="Angular logo" width="
`

- Type: expected=fix, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-849b8401

**Expected:** `build: update github/codeql-action action to v4.31.10`

**Generated:** `{"type":"docs","description":"Add installation instructions"}`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-22fefe78

**Expected:** `build: update bazel dependencies`

**Generated:** `The provided text is a JSON structure that defines various aspects and

Here's a breakdown of the key components:

1. Package Definitions:
- `@aspectrulesesbuild+//esbuild:extensions.bzl%esbuild`: This defines
extensions for the Esbuild build tool.
- `@aspectrulesjs+//npm:extensions.bzl
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `docs -> docs

ci -> ci
build(deps) -> build(deps)
test/ _test. -> test
.config., .env, .rc -> chore
`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `The changes in the `MODULE.bazel`, `package.json`, and `pnpm-lock.yaml``

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
