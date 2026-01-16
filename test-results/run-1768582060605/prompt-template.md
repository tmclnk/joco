# Prompt Template: file-pattern-v1

**Description:** File pattern matching for type selection with terse few-shot examples

## Template Text

```
Complete the commit message. Output ONLY the message after "Commit:".

Rules:
1. Format: type: description OR type(scope): description
2. Types: docs, ci, build, test, chore, feat, fix, refactor
3. One line, under 72 characters
4. Start with lowercase after colon

Type guide:
- docs = .md files, README, documentation
- ci = .github/, workflows, GitHub Actions
- build = package.json, pom.xml, dependencies
- test = test files
- feat = new feature
- fix = bug fix
- refactor = restructure without behavior change

Example diffs and commits:

Diff: README.md changed
Commit: docs: update readme

Diff: .github/workflows/ci.yml changed
Commit: ci: update workflow

Diff: package.json dependency updated
Commit: build(deps): update dependency

Diff: src/auth.ts error handling added
Commit: fix(auth): handle errors

Now your turn. Output ONLY the commit message, nothing else.

Diff:
%s

Commit:
```
