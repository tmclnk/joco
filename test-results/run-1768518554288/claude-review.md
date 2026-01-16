# Prompt Evaluation Results: run-1768518554288

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 0.0% |
| Scope Inclusion Rate | 0.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 40.0% |
| Average Score | 22.0/100 |
| Avg Generation Time | 10690 ms |

## Sample Results

### Successful Examples

_No successful examples with valid conventional commit format._

### Examples with Issues

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `Changes made to the README.md file.`

- Type: expected=docs, got=null
- Score: 25/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject should not end with period

---

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `Updated visualizer types to improve performance and maintainability.`

- Type: expected=refactor, got=null
- Score: 15/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters, Subject should not end with period

---

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `Changes to the `signals-visualizer.component.ts` file.`

- Type: expected=refactor, got=null
- Score: 15/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters, Subject should not end with period

---

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `Sure, here is the commit message:

Adev: Update Angular package format to FESM

This commit updates the Angular package format from a traditional ES
module (ESM) format to a Flattened ES Module (FESM) format. This change
allows for improved code organization and performance.
`

- Type: expected=docs, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `This is a detailed breakdown of the changes made to the two files:

`errors.ts`

* Added an `SignalFormsErrorCode` enum with a constant for
`BINDINGALREADYREGISTERED`.
* Removed the `implements` clause for `ɵFormFieldDirective` in the
`Field` class.
* Added a `focus` method to the `FormState` class to handle focus logic.
* Added a `describe` block for the `FieldState
`

- Type: expected=feat, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-f2cf96b5

**Expected:** `fix(docs-infra): use public folder instead of assets folder for static files in adev tutorials`

**Generated:** `commit:

Author: your_name
Date: 2023-04-12

Subjects:
- Learn Angular tutorial: Image optimization

Changes:
- content/tutorials/learn-angular/steps/11-optimizing-images/README.md
- content/tutorials/learn-angular/steps/11-optimizing-images/answer/src/
app/user.ts
- content
`

- Type: expected=fix, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-849b8401

**Expected:** `build: update github/codeql-action action to v4.31.10`

**Generated:** `Sure. The commit message is:

Commit: Update scorecard.yml
`

- Type: expected=build, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-22fefe78

**Expected:** `build: update bazel dependencies`

**Generated:** `The commit message is informative and provides a clear summary of the

Here are some of the key points:

* New functionality: The commit introduces support for telemetry
extensions written in TypeScript.
* Extension of existing functionality: The `rulesnodejs` and
`rulespython` extensions have been extended to support telemetry.
* Transitive dependencies: The commit fixes some transitive dependencies
between extensions.
* Bug fixes: The commit fixes a few bugs related to
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `The commit message contains the requested information.`

- Type: expected=docs, got=null
- Score: 15/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters, Subject should not end with period

---

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `This log shows the changes that were made to the `package.json` file for

Here's a summary:

* The `@angular/ng-dev` package was originally specified with a local
NPM resolution.
* However, the build process now uses a Git URL to specify the package
version.
* The version in `package.json` was changed from `7.0.0` to `7
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
