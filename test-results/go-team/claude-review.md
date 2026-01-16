# Prompt Evaluation Results: go-team

## Summary Metrics

| Metric | Value |
|--------|-------|
| Total Tests | 20 |
| Success Rate | 100.0% |
| Conventional Commit Rate | 85.0% |
| Scope Inclusion Rate | 75.0% |
| Length Compliance (72 char) | 100.0% |
| Recommended Length (50 char) | 55.0% |
| Average Score | 84.0/100 |
| Avg Generation Time | 4748 ms |

## Type Distribution

| Type | Count |
|------|-------|
| docs | 1 |
| fix | 3 |
| build | 10 |
| refactor | 3 |

## Sample Results

### Successful Examples

#### go-bb7c0c71

**Expected:** `archive/zip: reduce CPU usage in index construction`

**Generated:** `build(deps): update go modules to latest LTS versions`

- Type: expected=archive/zip, got=build
- Score: 90/100

---

#### go-2dcaaa75

**Expected:** `net/url: add urlmaxqueryparams GODEBUG to limit the number of query parameters`

**Generated:** `refactor(docs): update README for urlmaxqueryparams and urlstrictcolons`

- Type: expected=net/url, got=refactor
- Score: 90/100

---

#### go-5e1ad12d

**Expected:** `cmd/go/internal/work: sanitize flags before invoking 'pkg-config'`

**Generated:** `fix(http): handle pkg-config errors`

- Type: expected=cmd/go/internal/work, got=fix
- Score: 100/100

---

#### go-bba24719

**Expected:** `crypto/tls: don't copy auto-rotated session ticket keys in Config.Clone`

**Generated:** `build(deps): update go modules to use latest versions`

- Type: expected=crypto/tls, got=build
- Score: 90/100

---

#### go-9ef26e96

**Expected:** `sort: improve comment readability for Stable function`

**Generated:** `fix(sort): improve stability of sort algorithm`

- Type: expected=sort, got=fix
- Score: 100/100

---

### Examples with Issues

#### go-94a1296a

**Expected:** `cmd/go: update VCS commands to use safer flag/argument syntax`

**Generated:** `It looks like you've made several improvements to the Go command line

1. Addition of Error Handling:
- The `ValidateRepoRoot` function now includes an error check using
`errors.New`. This ensures that if there's an issue with the repository
root, it will be caught and reported.
- The `validateRepoSubDir` function also returns an
`

- Type: expected=cmd/go, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### go-9ef1692c

**Expected:** `simd/archsimd/_gen/simdgen: feature implications`

**Generated:** `The changes in the code are related to the addition of support for the

1. AVX512 Support:
- The `AVX512VNNI` function has been added to check if the CPU supports
the AVX512VNNI feature.
- The `AddRoundKey`, `InvMixColumns`, and other AES-related functions
have been updated
`

- Type: expected=simd/archsimd/_gen/simdgen, got=null
- Score: 20/100
- **Issues:** Does not match conventional commit format: type(scope): description, Subject exceeds recommended 50 characters

---

#### go-55ab5bba

**Expected:** `path/filepath: Match: improve doc comment`

**Generated:** `fix(path/filepath/match.go): use '\\"' for escaping backslashes on`

- Type: expected=path/filepath, got=null
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
