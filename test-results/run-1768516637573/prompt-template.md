# Prompt Template: file-pattern-v1

**Description:** File pattern matching for type selection with terse few-shot examples

## Template Text

```
OUTPUT: type(scope): description
NOTHING ELSE. One line only.

FILE PATTERN -> TYPE:
*.md, docs/, README* -> docs
.github/*, *.yml, ci/* -> ci
package.json, pom.xml, *.lock, deps/ -> build
test/, *_test.*, *.spec.* -> test
*.config.*, .env*, .*rc -> chore

CODE CHANGES -> TYPE:
New export/function/class -> feat
Error handling, null check -> fix
Rename, move, restructure -> refactor

SCOPE: module name (1 word), not path
Skip scope if unclear.

<examples>
diff: README.md +## Install
docs: add installation instructions

diff: .github/workflows/ci.yml
ci: add node 20 to test matrix

diff: package.json "lodash": "4.17.21"
build(deps): update lodash to 4.17.21

diff: src/auth/login.ts +catch (err)
fix(auth): handle login errors

diff: test/utils.test.ts +describe
test(utils): add unit tests

diff: src/api.ts rename fetchData->getData
refactor(api): rename fetchData to getData
</examples>

%s
```
