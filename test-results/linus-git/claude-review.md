# Prompt Evaluation Results: linus-git

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 20 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 90.0% |
| Scope Inclusion Rate | 60.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 35.0% |
| Average Score | 83.5/100 |
| Avg Generation Time | 3287 ms |

## Type Distribution

| Type | Count |
|------|-------|
| fix | 5 |
| docs | 4 |
| build | 3 |
| chore | 3 |
| feat | 1 |
| refactor | 2 |

## Sample Results

### Successful Examples

#### git-linus-04ede972

**Expected:** `symbolic-ref: refuse to set syntactically invalid target`

**Generated:** `docs: update documentation for the new `symbolic-ref` command`

- Type: expected=symbolic-ref, got=docs
- Score: 80/100

---

#### git-linus-acdd3776

**Expected:** `Add 'human' date format`

**Generated:** `refactor(date): improve human-readable date formatting`

- Type: expected=?, got=refactor
- Score: 90/100

---

#### git-linus-fd1062e5

**Expected:** `mailinfo: fix in-body header continuations`

**Generated:** `refactor(mailinfo): update commit message for multi-line in-body headers`

- Type: expected=mailinfo, got=refactor
- Score: 90/100

---

#### git-linus-859b7f1d

**Expected:** `pathspec: don't error out on all-exclusionary pathspec patterns`

**Generated:** `docs: update pathspec.c for new exclude pattern handling`

- Type: expected=pathspec, got=docs
- Score: 80/100

---

#### git-linus-42ebeb9d

**Expected:** `pathspec magic: add '^' as alias for '!'`

**Generated:** `docs: update README to include new features`

- Type: expected=pathspec magic, got=docs
- Score: 90/100

---

### Examples with Issues

#### git-linus-9d55b2e1

**Expected:** `mailinfo: don't require "text" mime type for attachments`

**Generated:** `fix(builtin/mailinfo.c): remove unused `message_type` and `boundary``

- Type: expected=mailinfo, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### git-linus-a3347b98

**Expected:** `fmt-merge-message: add empty line between tag and signature verification`

**Generated:** `fix(builtin/fmt-merge-msg.c): add newline before signatures`

- Type: expected=fmt-merge-message, got=null
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
