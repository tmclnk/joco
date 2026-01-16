# Prompt Evaluation Results: run-1768576018990

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 50 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 100.0% |
| Scope Inclusion Rate | 72.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 24.0% |
| Average Score | 89.6/100 |
| Avg Generation Time | 4230 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 2 |
| test | 7 |
| fix | 20 |
| build | 2 |
| ci | 3 |
| refactor | 8 |
| feat | 8 |

## Sample Results

### Successful Examples

#### react-bb8a76c6

**Expected:** `[DevTools] Show fallback in inspected element pane when no element is selected (#35503)`

**Generated:** `feat(devtools): add empty state fallback for inspected element`

- Type: expected=[DevTools] Show fallback in inspected element pane when no element is selected , got=feat
- Score: 90/100

---

#### react-fae15df4

**Expected:** `[DevTools] Add React Element pane to browser Elements panel (#35240)`

**Generated:** `feat(devtools): add inspected element sidebar pane to Elements panel`

- Type: expected=[DevTools] Add React Element pane to browser Elements panel , got=feat
- Score: 90/100

---

#### react-53daaf5a

**Expected:** `Improve the detection of changed hooks (#35123)`

**Generated:** `fix(devtools): improve hook change detection for composite hooks`

- Type: expected=Improve the detection of changed hooks , got=fix
- Score: 90/100

---

#### react-4a3d993e

**Expected:** `Add the suffix to cancelled view transition names (#35485)`

**Generated:** `fix(reconciler): add unique suffix to view transition names for multiple`

- Type: expected=Add the suffix to cancelled view transition names , got=fix
- Score: 90/100

---

#### react-c1866240

**Expected:** `[Fiber] Correctly handle replaying when hydrating (#35494)`

**Generated:** `fix(hydration): reset cursor when replaying suspended host components`

- Type: expected=[Fiber] Correctly handle replaying when hydrating , got=fix
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
