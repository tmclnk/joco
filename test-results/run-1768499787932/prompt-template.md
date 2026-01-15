# Prompt Template: baseline-v1

**Description:** Current joco production prompt - CommitMessagePrompt.createCompletePrompt()

## Template Text

```
Generate a concise git commit message from the diff.

Rules:
- Use conventional commit format: type(scope): description
- Types: feat, fix, chore, docs, refactor, test, style, perf
- Max 50 chars for subject line
- No body or footer
- Imperative mood (add, fix, update)
- Lowercase except proper nouns

Examples:
feat(auth): add OAuth2 login
fix(api): resolve null pointer in handler
chore(deps): update spring to 3.2.0

```
