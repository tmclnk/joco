# Prompt Evaluation Results: run-1768582060605

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 50 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 100.0% |
| Scope Inclusion Rate | 52.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 34.0% |
| Average Score | 88.6/100 |
| Avg Generation Time | 5375 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 2 |
| test | 5 |
| fix | 10 |
| build | 1 |
| ci | 1 |
| chore | 2 |
| feat | 3 |
| refactor | 26 |

## Sample Results

### Successful Examples

#### rust-b5c2a0fc

**Expected:** `Auto merge of #150982 - petrochenkov:modsplitfast2, r=JonathanBrouwer`

**Generated:** `refactor(resolve): improve identifier resolution diagnostics`

- Type: expected=Auto merge of #150982 - petrochenkov, got=refactor
- Score: 90/100

---

#### rust-7704328b

**Expected:** `Auto merge of #151158 - Zalathar:rollup-okXJcXA, r=Zalathar`

**Generated:** `refactor: restructure thir pattern matching and fix import ambiguity`

- Type: expected=Auto merge of #151158 - Zalathar, got=refactor
- Score: 80/100

---

#### rust-25672263

**Expected:** `Rollup merge of #151130 - diesel2, r=lqd`

**Generated:** `refactor(lint): rename ambiguous glob import lint`

- Type: expected=?, got=refactor
- Score: 100/100

---

#### rust-69f0a498

**Expected:** `Rollup merge of #151103 - array-pat-len, r=Nadrieril,petrochenkov`

**Generated:** `refactor(mir): restructure match pair builder`

- Type: expected=?, got=refactor
- Score: 100/100

---

#### rust-9682509b

**Expected:** `Rollup merge of #150979 - typeck-pat, r=lcnr`

**Generated:** `fix(pat): improve pattern type mismatch diagnostics`

- Type: expected=?, got=fix
- Score: 90/100

---

### Examples with Issues

## Review Instructions

Please analyze these results and provide:
1. **Quality Assessment**: Are the generated messages capturing the essence of the changes?
2. **Common Failure Patterns**: What types of commits is the model struggling with?
3. **Prompt Improvement Suggestions**: What changes to the prompt might improve results?
4. **Type Accuracy**: Is the model choosing appropriate commit types (feat, fix, etc.)?
5. **Scope Detection**: How well is the model identifying the affected component/module?
