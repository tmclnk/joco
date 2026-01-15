# Prompt Evaluation Results: angular-format

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 20 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 60.0% |
| Scope Inclusion Rate | 60.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 50.0% |
| Average Score | 67.0/100 |
| Avg Generation Time | 6701 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 1 |
| fix | 2 |
| build | 8 |
| refactor | 1 |

## Sample Results

### Successful Examples

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `fix(dagre-d3-types): update to visualizer-types`

- Type: expected=refactor, got=fix
- Score: 100/100

---

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `build(deps): update @types/dagre to 0.14.2`

- Type: expected=refactor, got=build
- Score: 100/100

---

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `refactor(http): move module id logic to its own function`

- Type: expected=docs, got=refactor
- Score: 90/100

---

#### angular-849b8401

**Expected:** `build: update github/codeql-action action to v4.31.10`

**Generated:** `build(deps): update codeql-action to 4.31.10`

- Type: MATCH
- Score: 100/100

---

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `docs(docs-card): fix typo in docs-card transformation function`

- Type: MATCH
- Score: 90/100

---

### Examples with Issues

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `docs:introduction/essentials/signal-forms.md: fix typo in documentation`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `The provided code snippet is a part of a library called Signal Forms,

Overview

- FormField Component: This component is responsible for rendering a
form field and managing its state.
- FormUiControl Interface: This interface defines a contract for a UI
control that can be used with Signal Forms.
`

- Type: expected=feat, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-f2cf96b5

**Expected:** `fix(docs-infra): use public folder instead of assets folder for static files in adev tutorials`

**Generated:** `It looks like you're referring to various tutorials and steps related to

1. Optimizing Images with `ngOptimizedImage`

- Description: The `ngOptimizedImage` directive is a part of Angular
that allows you to optimize image loading by serving different-sized
versions based on the user's device width.
- Steps:
- Install `@angular/common`: If
`

- Type: expected=fix, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-22fefe78

**Expected:** `build: update bazel dependencies`

**Generated:** `The provided code snippet is a manifest for the `aspectrulesnodejs`

Here's a breakdown of some key components and their functionalities:

Manifest Structure

The manifest starts with the `package.json` file, which contains
information about the package itself, such as its name, version, and
description.
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `The changes in the `MODULE.bazel` and `package.json` files are related

`MODULE.bazel`

git_override(
module_name = "devinfra",
-    commit = "1ce250a660eca5f613794a32cbae6d0f7
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-01592689

**Expected:** `docs: release notes for the v21.1.0 release`

**Generated:** `Changelog for Angular Version 21

Release Notes

Breaking Changes:
- Angular CLI: Deprecation of the `VERSION` export in the
`@angular/core` package.

General Improvements:
- Enhancements to the animation module.
- Improved performance optimizations in certain scenarios.
- Enhanced support for dynamic `[field]` bindings and signals in forms.

Features Added

1. Animation Module: Added new features such as advanced animation hooks
and improved support
`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-724e917f

**Expected:** `docs(docs-infra): Exempts animation-related symbols from linking`

**Generated:** `docs(pipeline/shared/linking.mts): update linking documentation`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-4c2d860c

**Expected:** `build: update dependency node to v22.22.0`

**Generated:** `The provided code snippet appears to be a configuration file for Node.js

1. `node_version`: This variable specifies the version of Node.js that
will be built. In this case, it is set to "22.21.1".

2. `node_repositories`: This dictionary contains various repository URLs
for different versions
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
