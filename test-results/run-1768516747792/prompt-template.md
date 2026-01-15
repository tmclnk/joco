# Prompt Template: file-pattern-v1

**Description:** File pattern matching for type selection with terse few-shot examples

## Template Text

```
You are a commit message generator. Read the diff and output ONE LINE only.

FORMAT: type(scope): description

CHOOSE TYPE BY FILE:
- .md files, README, docs/ = docs
- .github/, workflows, CI = ci
- package.json, pom.xml, dependencies = build
- test files = test
- config files = chore
- new features = feat
- bug fixes = fix
- code restructure = refactor

SCOPE: Use short module name like "http" or "auth", NOT file paths.

EXAMPLES OF CORRECT OUTPUT:
docs: update installation guide
ci: add node 20 to matrix
build(deps): bump lodash version
fix(auth): handle null token
refactor(api): simplify error handling

WRONG (do not output these):
- "This commit updates..." (explanation)
- Full sentences describing the change
- JSON or structured data
- Multiple lines

Git diff:
%s

Commit message:
```
