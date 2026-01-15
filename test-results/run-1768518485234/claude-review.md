# Prompt Evaluation Results: run-1768518485234

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 60.0% |
| Scope Inclusion Rate | 0.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 40.0% |
| Average Score | 59.5/100 |
| Avg Generation Time | 5783 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 5 |
| chore | 1 |

## Component Accuracy

| Metric | Value |
|--------|-------|
| Type Accuracy | 50.0% (3/6) |
| Scope Match (when expected) | 0.0% (0/3) |
| Scope Presence Match | 50.0% |
| Avg Description Similarity | 0.20 |
| Description Similarity Range | 0.15 - 0.27 |

## Sample Results

### Successful Examples

#### angular-87f47973

**Expected:** `refactor(devtools): drop dagre graph type`

**Generated:** `docs: update visualizer types documentation`

- Type: expected=refactor, got=docs
- Scope: expected=devtools, got=none
- Description Similarity: 0.15
- Score: 90/100

---

#### angular-72109478

**Expected:** `refactor(devtools): snap to root node on signal graph render`

**Generated:** `chore: update signals visualizer to use zoom controller for better`

- Type: expected=refactor, got=chore
- Scope: expected=devtools, got=none
- Description Similarity: 0.20
- Score: 80/100

---

#### angular-c7eee830

**Expected:** `docs: clarify that FESM can depend on shared chunks`

**Generated:** `docs: update content-tool-library-format.md guidelines`

- Type: MATCH (docs)
- Description Similarity: 0.18
- Score: 80/100

---

#### angular-849b8401

**Expected:** `build: update github/codeql-action action to v4.31.10`

**Generated:** `docs: update scorecard workflow YAML`

- Type: expected=build, got=docs
- Description Similarity: 0.27
- Score: 90/100

---

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `docs: update test file to include new feature`

- Type: MATCH (docs)
- Scope: expected=docs-infra, got=none
- Description Similarity: 0.21
- Score: 90/100

---

### Examples with Issues

#### angular-417c385d

**Expected:** `docs: update directive name from `[field]` to `[formField]` in signal-forms documentation`

**Generated:** `docs: updated field directives for better usability.`

- Type: MATCH (docs)
- Description Similarity: 0.21
- Score: 75/100
- **Issues:** Subject exceeds recommended 50 characters, Subject should not end with period

---

#### angular-5974cd0a

**Expected:** `feat(forms): Ability to manually register a form field binding in signal forms`

**Generated:** `It looks like you've posted a large code snippet, and I'll do my best to

After reviewing the code, I noticed two potential problems:

1. The `registerAsBinding` method in the `FormField` class is not
properly checking if the field state has already been bound as a form
field binding.
2. In the `focusBoundControl` test, the `nativeElement.focus()` call may
throw an error if the input element does not have focus.
`

- Type: expected=feat, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-f2cf96b5

**Expected:** `fix(docs-infra): use public folder instead of assets folder for static files in adev tutorials`

**Generated:** `The commit message is:

`Optimize images for Angular applications`

This commit message suggests that the author has optimized the images
used in their application, likely by replacing static images with
dynamic ones using the `NgOptimizedImage` component from Angular. The
changes are reflected in the code snippets provided.

However, if you'd like to provide more context or details about what
specific issues were resolved or improved during this commit, I'd be
happy to help!
`

- Type: expected=fix, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-22fefe78

**Expected:** `build: update bazel dependencies`

**Generated:** `This is a code snippet from the official Bazel build system

To provide more specific help, I'd like to know:

1. Which language are you interested in learning about (e.g.,
JavaScript, Python, Rust)?
2. Do you have any specific questions or topics related to those
languages that you'd like me to address?
3. Are there any particular features or concepts from the documentation
snippet you're looking
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `The commit message seems to be describing the changes made to a module

To make sense of this commit message, you would typically need more
context such as:

* What is the purpose of this commit (e.g., feature, bug fix,
performance improvement)?
* Who made these changes?
* When did they happen?

Without additional information, it's difficult to provide a specific
answer about what changes were made in this
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
