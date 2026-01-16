# Prompt Evaluation Results: run-1768581795536

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 50 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 96.0% |
| Scope Inclusion Rate | 74.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 34.0% |
| Average Score | 88.4/100 |
| Avg Generation Time | 4871 ms |

## Type Distribution

| Type | Count |
|------|-------|
| test | 13 |
| docs | 10 |
| fix | 13 |
| build | 2 |
| feat | 2 |
| refactor | 8 |

## Sample Results

### Successful Examples

#### go-5e1ad12d

**Expected:** `cmd/go/internal/work: sanitize flags before invoking 'pkg-config'`

**Generated:** `fix(security): validate pkg-config cflags and allow --static flag`

- Type: expected=cmd/go/internal/work, got=fix
- Score: 90/100

---

#### go-94a1296a

**Expected:** `cmd/go: update VCS commands to use safer flag/argument syntax`

**Generated:** `fix(modload): add ParsePathVersion to reject versions prefixed with - or`

- Type: expected=cmd/go, got=fix
- Score: 90/100

---

#### go-bba24719

**Expected:** `crypto/tls: don't copy auto-rotated session ticket keys in Config.Clone`

**Generated:** `fix(tls): check all chain certs for expiry on session resumption`

- Type: expected=crypto/tls, got=fix
- Score: 90/100

---

#### go-9ef26e96

**Expected:** `sort: improve comment readability for Stable function`

**Generated:** `docs: fix typo in sort package comments`

- Type: expected=sort, got=docs
- Score: 90/100

---

#### go-2bc4315d

**Expected:** `crypto/internal/fips140test: add ML-DSA to FIPS 140-3 functional tests`

**Generated:** `test(fips140): add ML-DSA tests and reorganize version conditionals`

- Type: expected=crypto/internal/fips140test, got=test
- Score: 90/100

---

### Examples with Issues

#### go-bb7c0c71

**Expected:** `archive/zip: reduce CPU usage in index construction`

**Generated:** `fix(archive/zip): optimize directory traversal in file list init`

- Type: expected=archive/zip, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### go-2dcaaa75

**Expected:** `net/url: add urlmaxqueryparams GODEBUG to limit the number of query parameters`

**Generated:** `feat(net/url): add query parameter limit to prevent DoS`

- Type: expected=net/url, got=null
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
