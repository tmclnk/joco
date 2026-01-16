# Prompt Evaluation Results: run-1768499787932

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 3 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 33.3% |
| Scope Inclusion Rate | 33.3% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 66.7% |
| Average Score | 50.0/100 |
| Avg Generation Time | 10588 ms |

## Type Distribution

| Type | Count |
|------|-------|
| feat | 1 |

## Sample Results

### Successful Examples

#### angular-a792315f

**Expected:** `docs(docs-infra): prevent heading from linking symbols`

**Generated:** `feat(docs-card): add support for anchor links in headings`

- Type: expected=docs, got=feat
- Score: 90/100

---

### Examples with Issues

#### angular-fa4bcf12

**Expected:** `build: update cross-repo angular dependencies`

**Generated:** `{

"name": "@angular/ng-dev",
"version": "0.0.0-b816685b6260ab760f1a24fc6f79db9bd1cb4921",
"description": "Private ng dev builds",
"main": "index.js",
"scripts": {
"build": "ng build --prod",
"test": "ng test"
`

- Type: expected=build, got=null
- Score: 30/100
- **Issues:** Does not match conventional commit format: type(scope): description

---

#### angular-01592689

**Expected:** `docs: release notes for the v21.1.0 release`

**Generated:** `# Changelog

[v21.1.0-next.0](https://github.com/angular/angular/compare/v21.1.0-next
.0...v21.1.0-next.0)

Platform-Browser

Added:
- `context` to `createApplication`

Fixed:
- Resolve JIT resources in `createApplication`

Router

Added:
- Add scroll behavior controls on router navigation
- Execute Run
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
