# Prompt Evaluation Results: rich-hickey

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 17 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 82.4% |
| Scope Inclusion Rate | 82.4% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 41.2% |
| Average Score | 81.8/100 |
| Avg Generation Time | 2818 ms |

## Type Distribution

| Type | Count |
|------|-------|
| fix | 5 |
| build | 4 |
| refactor | 1 |
| feat | 4 |

## Sample Results

### Successful Examples

#### clojure-hickey-f5403e9c

**Expected:** `CLJ-1472 Ensure monitor object is on stack, for verifiers`

**Generated:** `feat(core): introduce monitor-enter and monitor-exit macros for safe`

- Type: expected=?, got=feat
- Score: 90/100

---

#### clojure-hickey-989a3b98

**Expected:** `CLJ-1472 - Ensure monitor object is on stack so its easier to analyze in the face of locals clearing`

**Generated:** `build(deps): update leiningen to 3.8.1`

- Type: expected=?, got=build
- Score: 100/100

---

#### clojure-hickey-4d2a7dba

**Expected:** `add error phases to prepls`

**Generated:** `refactor(server): improve error handling and logging in server.clj`

- Type: expected=?, got=refactor
- Score: 90/100

---

#### clojure-hickey-2cc37bb5

**Expected:** `make :extend-via-metadata explicit option for defprotocol, Datafiable and Navigable opt-in`

**Generated:** `feat(core): update method implementations for Datafiable and Navigable`

- Type: expected=make , got=feat
- Score: 90/100

---

#### clojure-hickey-88eca12e

**Expected:** `Revert "CLJ-2427 CompilerException toString() throws when RT not yet initialized"`

**Generated:** `fix(compiler): handle spec errors in macro syntax check`

- Type: expected=Revert "CLJ-2427 CompilerException toString, got=fix
- Score: 90/100

---

### Examples with Issues

#### clojure-hickey-ff081f08

**Expected:** `add :name to datafied classes and nses, :class to meta of all datafied`

**Generated:** `chore(src/clj/clojure/datafy.clj): update types and docstrings for`

- Type: expected=add , got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### clojure-hickey-537d5ebd

**Expected:** `added string capture mode to LNPReader`

**Generated:** `fix(clj-lang/LineNumberingPushbackReader): improve handling of newline`

- Type: expected=?, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### clojure-hickey-c569e587

**Expected:** `don't preclude '.' in alias`

**Generated:** `fix(src/jvm/clojure/lang/LispReader.java): improve error handling`

- Type: expected=?, got=null
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
