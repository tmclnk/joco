# Prompt Test Harness

A comprehensive test harness for iterating on LLM prompts for commit message generation.

## Overview

The prompt harness (`prompt_harness.py`) loads test examples from the dataset, runs Ollama with different prompts and temperatures, evaluates the outputs, and generates detailed comparison reports.

## Features

- Load test examples from JSONL datasets
- Run Ollama models with different prompts and temperatures
- Evaluate outputs on multiple dimensions:
  - Format compliance (conventional commit format)
  - Type accuracy (feat, fix, docs, etc.)
  - Scope inclusion
  - Quality score (0-100)
- Support comparing multiple prompt variations in a single run
- Generate detailed reports with metrics and failure analysis

## Installation

Requires Python 3.9+ and the `requests` library:

```bash
pip install requests
```

Make sure Ollama is running:

```bash
ollama serve
```

## Quick Start

### Test the baseline prompt

```bash
cd /workspaces/joco
./scripts/prompt_harness.py --builtin baseline --num-examples 10 --verbose
```

### Compare all built-in prompts

```bash
./scripts/prompt_harness.py --builtin all --num-examples 20 --verbose
```

### Test a custom prompt file

```bash
./scripts/prompt_harness.py --prompt prompts/baseline.txt --num-examples 20
```

### Test with different temperatures

```bash
./scripts/prompt_harness.py --prompt prompts/strict-format-v1.txt --temperatures 0.1,0.3,0.5,0.7 --num-examples 20
```

### Compare multiple prompts

```bash
./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/strict-format-v1.txt --num-examples 20
```

## Usage

```
usage: prompt_harness.py [-h] [--dataset DATASET] [--num-examples NUM_EXAMPLES]
                         [--prompt PROMPT] [--prompts PROMPTS]
                         [--builtin {baseline,strict-format-v1,minimal,verbose,all}]
                         [--prompt-name PROMPT_NAME] [--model MODEL]
                         [--temperature TEMPERATURE] [--temperatures TEMPERATURES]
                         [--max-tokens MAX_TOKENS] [--ollama-url OLLAMA_URL]
                         [--verbose]

Dataset options:
  --dataset DATASET           Path to JSONL dataset (default: dataset/validation.jsonl)
  --num-examples NUM_EXAMPLES Number of examples to test (default: 10)

Prompt options:
  --prompt PROMPT             Path to prompt file to test
  --prompts PROMPTS           Comma-separated paths to multiple prompt files
  --builtin {baseline,strict-format-v1,minimal,verbose,all}
                              Use a built-in prompt variation
  --prompt-name PROMPT_NAME   Name for the prompt (used in reports)

Model options:
  --model MODEL               Ollama model to use (default: qwen2.5-coder:1.5b)
  --temperature TEMPERATURE   Temperature for generation (default: 0.3)
  --temperatures TEMPERATURES Comma-separated temperatures (e.g., 0.1,0.3,0.5)
  --max-tokens MAX_TOKENS     Maximum tokens to generate (default: 100)

Other options:
  --ollama-url OLLAMA_URL     Ollama API URL (default: http://localhost:11434)
  --verbose                   Print progress for each example
```

## Built-in Prompts

The harness includes several built-in prompt variations for quick testing:

- **baseline**: Current joco prompt with detailed guidance
- **strict-format-v1**: Simplified prompt emphasizing format compliance (from joco-8hs)
- **minimal**: Ultra-minimal prompt with bare instructions
- **verbose**: Detailed prompt with extensive guidance
- **all**: Test all built-in prompts at once

## Output

The harness generates a comprehensive report including:

### Summary Table

```
Prompt                         Format%     Type%    Scope%  Avg Score
--------------------------------------------------------------------------------
baseline                          70.0%     60.0%     40.0%       65.5
strict-format-v1_t0.3             80.0%     70.0%     50.0%       74.0
```

### Detailed Metrics

For each prompt variation:
- Configuration (model, temperature, max tokens)
- Core metrics (format compliance, type accuracy, scope inclusion, avg score)
- Performance metrics (generation time, token counts)
- Type distribution
- Failure analysis (format failures, type mismatches)

### Recommendation

The harness automatically identifies the best performing prompt based on average score.

## Evaluation Metrics

### Format Compliance

Checks if the output follows conventional commit format:
- `type: description` or `type(scope): description`
- Valid type from: feat, fix, docs, style, refactor, test, chore, ci, build, perf
- Non-empty description (>= 3 characters)

### Type Accuracy

Compares the generated commit type against the expected type from the dataset.

### Scope Inclusion

Tracks whether the generated commit includes a scope (e.g., `fix(auth): ...`).

### Quality Score (0-100)

Composite score based on:
- Format compliance: 40 points
- Type accuracy: 30 points
- Description similarity: 30 points (word overlap with expected)

### Token Statistics

Tracks prompt tokens, completion tokens, and generation time for performance analysis.

## Creating Custom Prompts

1. Create a text file with your prompt (e.g., `prompts/my-prompt.txt`)
2. Test it with the harness:
   ```bash
   ./scripts/prompt_harness.py --prompt prompts/my-prompt.txt --num-examples 20
   ```
3. Compare against baseline:
   ```bash
   ./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/my-prompt.txt --num-examples 20
   ```
4. Document results in `EXPERIMENT_LOG.md`

## Best Practices

Based on experiments documented in EXPERIMENT_LOG.md:

1. **Use lower temperatures**: 0.3 works better than 0.7 for format compliance
2. **Be explicit about format**: Clear format instructions reduce explanatory text
3. **Keep it simple**: Minimal prompts often perform as well as verbose ones
4. **Test with sufficient examples**: Use at least 20 examples for reliable metrics
5. **Compare against baseline**: Always benchmark new prompts against the baseline

## Integration with Beads Issues

When working on prompt experiments:

1. Create or reference a beads issue (e.g., `joco-70n`)
2. Update issue status: `bd update joco-70n --status=in_progress`
3. Run experiments with the harness
4. Document results in the issue or EXPERIMENT_LOG.md
5. Close issue when complete: `bd close joco-70n`

## Example Workflow

```bash
# 1. Start working on a prompt experiment
bd update joco-70n --status=in_progress

# 2. Test baseline for comparison
./scripts/prompt_harness.py --builtin baseline --num-examples 20 --verbose > results/baseline.txt

# 3. Create and test new prompt
cat > prompts/my-new-prompt.txt << 'EOF'
Generate a single-line conventional commit message.
Format: type(scope): description
Output only the commit message, nothing else.
EOF

./scripts/prompt_harness.py --prompt prompts/my-new-prompt.txt --num-examples 20 --verbose > results/my-new-prompt.txt

# 4. Test different temperatures
./scripts/prompt_harness.py --prompt prompts/my-new-prompt.txt --temperatures 0.1,0.3,0.5 --num-examples 20 > results/temperature-test.txt

# 5. Compare all variations
./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/my-new-prompt.txt --num-examples 20

# 6. Document results
./scripts/joco_log.py experiment \
  --name "Prompt experiment: my-new-prompt" \
  --hypothesis "Hypothesis about why this prompt should work better" \
  --metrics "format=80%, type=70%, score=75"

# 7. Close issue
bd close joco-70n
```

## Troubleshooting

### Ollama not running

```
ERROR: Cannot connect to Ollama at http://localhost:11434
Make sure Ollama is running: ollama serve
```

Solution: Start Ollama in another terminal with `ollama serve`

### Model not found

```
ERROR: model 'qwen2.5-coder:1.5b' not found
```

Solution: Pull the model first with `ollama pull qwen2.5-coder:1.5b`

### Dataset file not found

```
ERROR: [Errno 2] No such file or directory: 'dataset/validation.jsonl'
```

Solution: Check the path and use `--dataset` flag to specify correct location

## Related Files

- `prompts/`: Directory containing prompt variation files
- `EXPERIMENT_LOG.md`: Log of all prompt experiments and results
- `.beads/issues.jsonl`: Beads issue tracker for multi-session work

## References

- Issue joco-70n: Test harness for prompt iteration
- Issue joco-8hs: Prompt experiment with strict-format-v1
- Issue joco-s29: Temperature tuning experiment (best: temp=0.3)
