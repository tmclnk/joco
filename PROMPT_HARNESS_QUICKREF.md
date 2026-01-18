# Prompt Harness Quick Reference

## Quick Start

```bash
# Start Ollama
ollama serve

# Test baseline
./scripts/prompt_harness.py --builtin baseline --num-examples 10 --verbose

# Compare all prompts
./scripts/prompt_harness.py --builtin all --num-examples 20
```

## Common Commands

```bash
# Test single prompt
./scripts/prompt_harness.py --prompt prompts/my-prompt.txt --num-examples 20

# Compare prompts
./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/new.txt --num-examples 20

# Test temperatures
./scripts/prompt_harness.py --builtin strict-format-v1 --temperatures 0.1,0.3,0.5 --num-examples 20

# Different model
./scripts/prompt_harness.py --builtin baseline --model qwen2.5-coder:0.5b --num-examples 20

# Save output
./scripts/prompt_harness.py --builtin all --num-examples 20 > results.txt
```

## Built-in Prompts

| Prompt | Description | Best Temp | Best Score |
|--------|-------------|-----------|------------|
| baseline | Current joco prompt | 0.3 | 64.0 |
| strict-format-v1 | Simplified (joco-8hs/s29) | 0.3 | 74.0 |
| minimal | Ultra-minimal | 0.3 | ? |
| verbose | Detailed guidance | 0.3 | ? |

## Metrics Guide

| Metric | Good | Target | Meaning |
|--------|------|--------|---------|
| Format % | >70% | >80% | Valid conventional commit format |
| Type % | >60% | >70% | Correct commit type |
| Scope % | >40% | >50% | Includes scope |
| Avg Score | >70 | >75 | Overall quality (0-100) |

## Score Breakdown

- Format compliance: 40 points
- Type accuracy: 30 points
- Description similarity: 30 points

## Workflow

```bash
# 1. Create prompt
cat > prompts/new-prompt.txt << 'EOF'
[Your prompt text]
EOF

# 2. Test it
./scripts/prompt_harness.py --prompt prompts/new-prompt.txt --num-examples 20

# 3. Compare to baseline
./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/new-prompt.txt --num-examples 20

# 4. Tune temperature
./scripts/prompt_harness.py --prompt prompts/new-prompt.txt --temperatures 0.1,0.3,0.5,0.7 --num-examples 20

# 5. Document in EXPERIMENT_LOG.md
```

## Files

- `scripts/prompt_harness.py` - Main script
- `scripts/test_prompt_harness.py` - Unit tests
- `prompts/*.txt` - Prompt variations
- `PROMPT_HARNESS_GUIDE.md` - Full guide
- `scripts/README_PROMPT_HARNESS.md` - Technical docs

## Tips

- Use temp=0.3 for best results
- Test with at least 20 examples
- Always compare against baseline
- Document everything in EXPERIMENT_LOG.md

## Help

```bash
./scripts/prompt_harness.py --help
./scripts/test_prompt_harness.py
```

## Related Issues

- joco-70n: Test harness implementation (closed)
- joco-8hs: strict-format-v1 test (65.5 score)
- joco-s29: Temperature tuning (74.0 score at temp=0.3)
