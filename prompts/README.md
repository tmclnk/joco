# Prompt Variations

This directory contains different prompt variations for testing with the prompt harness.

## Available Prompts

### baseline.txt
The current joco prompt used in production. Provides detailed guidance on commit types and scopes.

### strict-format-v1.txt
Simplified prompt that emphasizes format compliance. Tested in joco-8hs and joco-s29.

## Testing Prompts

Use the prompt harness to test variations:

```bash
# Test a single prompt
cd /workspaces/joco
./scripts/prompt_harness.py --prompt prompts/baseline.txt --num-examples 20

# Compare multiple prompts
./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/strict-format-v1.txt --num-examples 20

# Test with different temperatures
./scripts/prompt_harness.py --prompt prompts/strict-format-v1.txt --temperatures 0.1,0.3,0.5,0.7 --num-examples 20

# Use built-in prompts
./scripts/prompt_harness.py --builtin all --num-examples 20
```

## Creating New Prompts

1. Create a new `.txt` file in this directory
2. Write your prompt text
3. Test it with the harness
4. Document results in EXPERIMENT_LOG.md

## Prompt Design Guidelines

Based on experiments (see EXPERIMENT_LOG.md):

- **Lower temperature (0.3)** works better than higher temperatures
- **Explicit format instructions** reduce explanation text
- **Clear type definitions** improve type accuracy
- **Simple, direct language** works better than verbose guidance
- **Examples are helpful** but keep them concise

## Best Performing Prompts

As of 2026-01-18:

1. **strict-format-v1 @ temp=0.3**: 74.0 avg score, 70% format compliance
2. **strict-format-v1 @ temp=0.7**: 65.5 avg score
3. **baseline @ temp=0.7**: 64.0 avg score (baseline)
