# Prompt Evaluation Results: run-1768501206994

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 70.0% |
| Scope Inclusion Rate | 60.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 60.0% |
| Average Score | 74.0/100 |
| Avg Generation Time | 6375 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 1 |
| fix | 2 |
| build | 3 |
| feat | 1 |

## Sample Results

### Successful Examples

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `feat(docs): update docs-card extension to support SVG illustrations and`

- Type: expected=docs, got=feat
- Score: 90/100

---

#### angular-d12e6f09

**Expected:** `docs: release notes for the v21.0.9 release`

**Generated:** `fix(CHANGELOG.md): update release notes for v21.0.9`

- Type: expected=docs, got=fix
- Score: 90/100

---

#### angular-3f13db87

**Expected:** `ci: update algolia synonym updater to use node directly rather than tsx`

**Generated:** `build(deps): update algoliasearch to 4.2.0`

- Type: expected=ci, got=build
- Score: 100/100

---

#### angular-724e917f

**Expected:** `docs(docs-infra): Exempts animation-related symbols from linking`

**Generated:** `build(deps): update typescript to 5.0`

- Type: expected=docs, got=build
- Score: 100/100

---

#### angular-c9f584b9

**Expected:** `docs: add IMPORTANT prefix to pipe usage warning and`

**Generated:** `docs: update guide/templates/pipes.md`

- Type: MATCH
- Score: 90/100

---

### Examples with Issues

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `The commit you provided appears to be related to updating the `devinfra`

1. Updating the `MODULE.bazel` file to specify the new version of
`@angular/ng-dev`.
2. Updating the `package.json` file to use the new version of
`@angular/ng-dev`.
3. Updating the `pnpm-lock.yaml` file to reflect the new version of
`@angular/ng-dev`.

Here's a brief overview
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-01592689

**Expected:** `docs: release notes for the v21.1.0 release`

**Generated:** `This commit appears to be related to the Angular framework, specifically

1. Platform-Browser Module:
- A new method `createApplication` is added to the `BrowserModule`. This
method allows for creating an application context with additional
options.
- An optional parameter `context` is introduced, which can be used to
pass additional information or configurations to the application.

2. Router Module
`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-4c2d860c

**Expected:** `build: update dependency node to v22.22.0`

**Generated:** `{

"message": "Update Node.js version to 22.22.0",
"author": {
"name": "John Doe",
"email": "john.doe@example.com"
},
"date": "2023-10-05T14:30:00Z",
"files_changed": [
"package.json",
"src/main.js"
]
}
`

- Type: expected=build, got=null
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
