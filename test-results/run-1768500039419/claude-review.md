# Prompt Evaluation Results: run-1768500039419

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 10 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 40.0% |
| Scope Inclusion Rate | 40.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 20.0% |
| Average Score | 50.0/100 |
| Avg Generation Time | 6611 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 1 |
| feat | 3 |

## Sample Results

### Successful Examples

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `feat(docs): update anchor target links in headings`

- Type: expected=docs, got=feat
- Score: 100/100

---

#### angular-d12e6f09

**Expected:** `docs: release notes for the v21.0.9 release`

**Generated:** `docs(readme): update instructions for adding a new feature`

- Type: MATCH
- Score: 90/100

---

#### angular-a904d9f7

**Expected:** `fix(compiler-cli): support nested component declaration`

**Generated:** `feat(compiler-cli): add nested component definition test case`

- Type: expected=fix, got=feat
- Scope: MATCH
- Score: 90/100

---

#### angular-074666b2

**Expected:** `docs(docs-infra): use Signals Forms focusBoundControl`

**Generated:** `feat(api-reference-list): update search dialog form for better user`

- Type: expected=docs, got=feat
- Score: 90/100

---

### Examples with Issues

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `Merge pull request #493 from modelcontextprotocol/patch-1

Update dev dependencies to use the latest commit from
private-ng-dev-builds repository.

Commit Message

This merge requests updates the `devDependencies` in the `package.json`
file to use the latest commit from the private-ng-dev-builds repository.
The commit hash used is `0b9654158d79ed9bf68866d619ea3
`

- Type: expected=build, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-01592689

**Expected:** `docs: release notes for the v21.1.0 release`

**Generated:** `The commit message provided does not contain any significant changes or

To provide more accurate information, you would need to:

1. Identify the upcoming version of Angular (e.g., v23).
2. Determine if there are any new features or improvements expected in
this version.
3. Provide additional context if available about the
`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-3f13db87

**Expected:** `ci: update algolia synonym updater to use node directly rather than tsx`

**Generated:** `refactor(adev/scripts/synonyms): update synonym record upload script`

- Type: expected=ci, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-724e917f

**Expected:** `docs(docs-infra): Exempts animation-related symbols from linking`

**Generated:** `docs(shared-docs/pipeline/shared/linking.mts): add animation,`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-c9f584b9

**Expected:** `docs: add IMPORTANT prefix to pipe usage warning and`

**Generated:** `feat(guide/templates/pipes): update doc for pipe detection in arrays and`

- Type: expected=docs, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### angular-4c2d860c

**Expected:** `build: update dependency node to v22.22.0`

**Generated:** `Merge pull request #3 from user/repo

Update Node.js version to 22.22.0

* Update `.github/workflows/ci.yml` to use Node.js 22.22.0
* Update `package.json` and `pom.xml` to use Node.js 22.22.0
* Update `README.md` to reflect the new version number
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
