# Prompt Template: few-shot-v1

**Description:** Few-shot template with 4 examples demonstrating different commit types

## Template Text

```
Generate a conventional commit message for the following git diff.

Format: type(scope): description
Types: feat, fix, docs, refactor, test, chore, style, perf

Examples:

Input diff:
```
+export function authenticate(user) {
+  return generateToken(user);
+}
```
Output: feat(auth): add user authentication function

Input diff:
```
-const result = data.map(x => x.value);
+const result = data?.map(x => x.value) ?? [];
```
Output: fix(data): handle null data array in mapping

Input diff:
```
-import { old } from './old';
+import { new } from './new';
```
Output: refactor(imports): update import paths

Input diff:
```
+describe('Calculator', () => {
+  it('should add numbers', () => {
+    expect(add(1, 2)).toBe(3);
+  });
+});
```
Output: test(calculator): add unit tests for add function

Now generate a commit message for this diff:

Input diff:
```
%s
```
Output:
```
