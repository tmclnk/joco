# Prompt Test Harness Guide

A comprehensive guide to using the prompt test harness for iterating on LLM prompts for commit message generation.

## Quick Start

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Test the baseline prompt
./scripts/prompt_harness.py --builtin baseline --num-examples 10 --verbose

# 3. Compare all built-in prompts
./scripts/prompt_harness.py --builtin all --num-examples 20

# 4. Test with different temperatures
./scripts/prompt_harness.py --builtin strict-format-v1 --temperatures 0.1,0.3,0.5,0.7 --num-examples 20
```

## What is the Prompt Harness?

The prompt harness is a Python tool that automates the process of testing and comparing different prompts for commit message generation. It:

1. Loads test examples (git diffs + expected commit messages) from the dataset
2. Runs Ollama with different prompts and parameters
3. Evaluates the generated commit messages on multiple dimensions
4. Generates detailed comparison reports

## Why Use It?

Manual prompt testing is time-consuming and inconsistent. The harness provides:

- **Reproducibility**: Same test cases, same metrics, comparable results
- **Comprehensive evaluation**: Format, type accuracy, scope, quality score
- **Efficiency**: Test multiple variations in one run
- **Detailed analysis**: Identifies specific failure patterns
- **Documentation**: Results are automatically organized and comparable

## Architecture

```
┌─────────────┐
│  Dataset    │  (validation.jsonl)
│  Examples   │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Prompt Harness     │
│  ───────────────    │
│  • Load examples    │
│  • Build prompts    │
│  • Call Ollama      │
│  • Evaluate outputs │
└──────┬──────────────┘
       │
       ▼
┌──────────────┐      ┌─────────────┐
│  Ollama API  │◄────►│  Model      │
│              │      │  (Qwen2.5)  │
└──────┬───────┘      └─────────────┘
       │
       ▼
┌──────────────────────┐
│  Evaluation          │
│  ───────────────     │
│  • Format check      │
│  • Type accuracy     │
│  • Scope inclusion   │
│  • Quality score     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Comparison Report   │
│  • Summary metrics   │
│  • Type distribution │
│  • Failure analysis  │
│  • Best prompt       │
└──────────────────────┘
```

## Evaluation Metrics

### Format Compliance

Checks if output follows conventional commit format:
- `type: description` or `type(scope): description`
- Valid type: feat, fix, docs, style, refactor, test, chore, ci, build, perf
- Non-empty description (≥ 3 characters)

**Why it matters**: Invalid format makes commits unparseable by tools and harder to understand.

### Type Accuracy

Compares generated type against expected type from the dataset.

**Why it matters**: Incorrect type classification makes commit history confusing and breaks filtering/analysis.

### Scope Inclusion

Tracks whether generated commit includes a scope (e.g., `fix(auth): ...`).

**Why it matters**: Scopes help organize commits by module/component.

### Quality Score (0-100)

Composite score based on:
- Format compliance: 40 points
- Type accuracy: 30 points
- Description similarity: 30 points (word overlap with expected)

**Why it matters**: Provides a single metric to compare prompt variations.

## Built-in Prompts

### baseline

The current joco production prompt. Provides detailed guidance on types, scopes, and format.

**Characteristics:**
- Verbose instructions
- Explicit type definitions
- Multiple examples
- Clear scope rules

**When to use**: As a baseline for comparison. This is the "control" in your experiments.

### strict-format-v1

Simplified prompt emphasizing format compliance. Tested in issues joco-8hs and joco-s29.

**Characteristics:**
- Minimal instructions
- Direct format specification
- Emphasis on "ONE line only"
- Fewer examples

**When to use**: When you want to test if brevity improves format compliance.

**Best results**: temp=0.3 (74.0 avg score, 70% format compliance)

### minimal

Ultra-minimal prompt with bare instructions.

**Characteristics:**
- Shortest possible prompt
- Just format and types
- No examples
- No detailed rules

**When to use**: To test lower bound - how little instruction is needed?

### verbose

Detailed prompt with extensive guidance.

**Characteristics:**
- Longest prompt
- Detailed type explanations
- Explicit scope rules
- Multiple examples and rules

**When to use**: To test upper bound - does more guidance help?

## Command Reference

### Test a single prompt

```bash
./scripts/prompt_harness.py --builtin baseline --num-examples 20 --verbose
```

### Compare multiple prompts

```bash
# Built-in prompts
./scripts/prompt_harness.py --builtin all --num-examples 20

# Custom prompt files
./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/my-prompt.txt --num-examples 20
```

### Test different temperatures

```bash
./scripts/prompt_harness.py --builtin strict-format-v1 --temperatures 0.1,0.3,0.5,0.7 --num-examples 20
```

### Test different models

```bash
./scripts/prompt_harness.py --builtin baseline --model qwen2.5-coder:0.5b --num-examples 20
./scripts/prompt_harness.py --builtin baseline --model qwen2.5-coder:3b --num-examples 20
```

### Use custom dataset

```bash
./scripts/prompt_harness.py --builtin baseline --dataset dataset/train.jsonl --num-examples 50
```

### Save output to file

```bash
./scripts/prompt_harness.py --builtin all --num-examples 20 > results/experiment-2026-01-18.txt
```

## Workflow: Creating and Testing a New Prompt

### 1. Create a prompt file

```bash
cat > prompts/my-new-prompt.txt << 'EOF'
Generate a conventional commit message.

Format: type(scope): description

Valid types: feat, fix, docs, test, ci, chore, build, refactor, style, perf

Rules:
- One line only
- Lowercase description
- Imperative mood

Output only the commit message.
EOF
```

### 2. Test it

```bash
./scripts/prompt_harness.py --prompt prompts/my-new-prompt.txt --num-examples 20 --verbose
```

### 3. Compare against baseline

```bash
./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/my-new-prompt.txt --num-examples 20
```

### 4. Tune temperature

```bash
./scripts/prompt_harness.py --prompt prompts/my-new-prompt.txt --temperatures 0.1,0.3,0.5,0.7 --num-examples 20
```

### 5. Document results

Update EXPERIMENT_LOG.md with your findings:

```markdown
## Experiment: My New Prompt Variation

**Date**: 2026-01-18
**Hypothesis**: Simplifying instructions will improve format compliance
**Approach**: Tested new prompt with 4 different temperatures

**Results**:
| Temperature | Format% | Type% | Avg Score |
|-------------|---------|-------|-----------|
| 0.1         | 75.0    | 65.0  | 72.0      |
| 0.3         | 80.0    | 70.0  | 76.0      |
| 0.5         | 70.0    | 65.0  | 68.0      |
| 0.7         | 60.0    | 55.0  | 62.0      |

**Conclusion**: Temperature 0.3 works best. New prompt improved baseline by 5%.
```

## Understanding the Report

### Summary Table

```
Prompt                         Format%     Type%    Scope%  Avg Score
--------------------------------------------------------------------------------
baseline                          70.0%     60.0%     40.0%       65.5
strict-format-v1_t0.3             80.0%     70.0%     50.0%       74.0
```

**How to read**:
- **Format%**: Percentage of outputs with valid conventional commit format
- **Type%**: Percentage where generated type matches expected type
- **Scope%**: Percentage that include a scope
- **Avg Score**: Average quality score (0-100)

**What to look for**:
- Higher is better for all metrics
- Format% should be >70% (aim for >80%)
- Type% indicates how well the prompt guides type selection
- Avg Score is the key metric for overall quality

### Detailed Results

For each prompt, you'll see:

**Configuration**: Model, temperature, max tokens

**Metrics**: Detailed breakdown with counts

**Type distribution**: Shows which types the model generates
```
fix: 5
docs: 3
build: 2
```

**Failure analysis**: Examples of format failures and type mismatches

**What to look for**:
- Type distribution should be diverse (not all feat or fix)
- Format failures reveal specific issues (too verbose? wrong format?)
- Type mismatches show confusion patterns (build vs ci?)

### Recommendation

The harness picks the best prompt based on average score.

## Tips for Prompt Engineering

Based on experiments documented in EXPERIMENT_LOG.md:

### 1. Temperature matters

- **0.1-0.3**: Best for format compliance and consistency
- **0.5-0.7**: More creative but less consistent
- **0.7+**: Too creative, often breaks format

**Recommendation**: Use temp=0.3 for commit messages

### 2. Less is often more

- Verbose prompts don't always perform better
- Clear, direct instructions work well
- Too many examples can confuse smaller models

**Recommendation**: Start minimal, add guidance only if needed

### 3. Emphasize the constraint

What works:
- "ONE line only"
- "Output ONLY the commit message"
- "No explanation"

What doesn't:
- Vague instructions
- Multiple examples without clear format
- Asking for both message and explanation

**Recommendation**: Be explicit about output format

### 4. Test systematically

- Always compare against baseline
- Test multiple temperatures
- Use enough examples (≥20) for reliable metrics
- Document everything

**Recommendation**: Follow the workflow above

### 5. Type classification is hard

- Smaller models struggle with build vs ci vs chore
- File patterns help (*.md → docs, package.json → build)
- Examples help but need to be clear

**Recommendation**: Provide clear type definitions with file patterns

## Integration with Beads Issues

Track your prompt experiments using Beads:

```bash
# Start working on prompt iteration
bd create "Test new minimal prompt variation"
bd update joco-123 --status=in_progress

# Run experiments
./scripts/prompt_harness.py --prompt prompts/minimal-v2.txt --num-examples 20 > results/minimal-v2.txt

# Document in issue
bd update joco-123 --comment "Tested minimal-v2: 72% format compliance, 68% type accuracy"

# Close when done
bd close joco-123
bd sync
```

## Troubleshooting

### Ollama not running

```
ERROR: Cannot connect to Ollama at http://localhost:11434
```

**Solution**: Start Ollama in another terminal: `ollama serve`

### Model not found

```
ERROR: model 'qwen2.5-coder:1.5b' not found
```

**Solution**: Pull the model first: `ollama pull qwen2.5-coder:1.5b`

### Slow generation

If generation is slow:
1. Use a smaller model: `--model qwen2.5-coder:0.5b`
2. Reduce examples: `--num-examples 10`
3. Check system resources (RAM, CPU)

### Inconsistent results

If results vary between runs:
1. Lower temperature: `--temperature 0.1`
2. Use more examples for better statistics
3. Test multiple times and average

## Advanced Usage

### Testing with extended dataset

```bash
./scripts/prompt_harness.py --builtin baseline --dataset dataset/train_extended.jsonl --num-examples 50
```

### Testing multiple models

```bash
for model in qwen2.5-coder:0.5b qwen2.5-coder:1.5b qwen2.5-coder:3b; do
    echo "Testing $model..."
    ./scripts/prompt_harness.py --builtin baseline --model "$model" --num-examples 20 > "results/baseline-$model.txt"
done
```

### Automated temperature sweep

```bash
./scripts/prompt_harness.py --builtin strict-format-v1 --temperatures 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --num-examples 20
```

### Batch testing all prompts

```bash
for prompt in prompts/*.txt; do
    name=$(basename "$prompt" .txt)
    echo "Testing $name..."
    ./scripts/prompt_harness.py --prompt "$prompt" --num-examples 20 > "results/$name-$(date +%Y%m%d).txt"
done
```

## Files and Directories

```
joco/
├── scripts/
│   ├── prompt_harness.py          # Main harness script
│   ├── test_prompt_harness.py     # Unit tests
│   ├── example_prompt_test.sh     # Example workflows
│   └── README_PROMPT_HARNESS.md   # Technical documentation
├── prompts/
│   ├── baseline.txt               # Current joco prompt
│   ├── strict-format-v1.txt       # Simplified prompt
│   └── README.md                  # Prompt documentation
├── dataset/
│   ├── validation.jsonl           # Validation set (29 examples)
│   ├── validation_extended.jsonl  # Extended validation (208 examples)
│   └── train.jsonl                # Training set
├── EXPERIMENT_LOG.md              # Experiment documentation
└── PROMPT_HARNESS_GUIDE.md        # This file
```

## Next Steps

1. **Run your first test**:
   ```bash
   ./scripts/prompt_harness.py --builtin all --num-examples 20 --verbose
   ```

2. **Create your own prompt**:
   - Copy `prompts/baseline.txt` as a starting point
   - Make your changes
   - Test it

3. **Compare systematically**:
   - Test against baseline
   - Try different temperatures
   - Document results

4. **Iterate**:
   - Analyze failures
   - Refine your prompt
   - Test again

## References

- **joco-70n**: Test harness for prompt iteration (this issue)
- **joco-8hs**: Prompt experiment with strict-format-v1
- **joco-s29**: Temperature tuning (best: temp=0.3, score=74.0)
- **EXPERIMENT_LOG.md**: Complete history of all experiments
- **scripts/README_PROMPT_HARNESS.md**: Technical documentation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review EXPERIMENT_LOG.md for similar experiments
3. Create a beads issue: `bd create "Prompt harness: [your issue]"`
