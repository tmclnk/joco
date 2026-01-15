# Log Experiment Results

Add a new entry to the experiment log file (`EXPERIMENT_LOG.md`).

## Instructions

1. Read the current `EXPERIMENT_LOG.md` file
2. Add a new experiment entry with the following structure:

```markdown
### Experiment N: [Title]

**Date**: [Today's date]
**Hypothesis**: [What we're testing]
**Template/Config**: [What changed]

**Results**:
- Format Compliance: X%
- Type Accuracy: X%
- Score: X/100
- [Other relevant metrics]

**Why it worked/failed**: [Brief explanation]
```

3. Update the "Test Run Reference" table if a new run was created
4. Update "Future Experiments to Try" if we learned something new

## Usage

When the user runs experiments (prompts, models, temperatures, etc.), use this skill to record:
- What was tried
- The exact configuration/prompt used
- Quantitative results (metrics from harness runs)
- Qualitative observations (why it worked or failed)

Keep entries factual and reproducible. Include run IDs so results can be verified.

## Example

User: "I tried lowering the temperature to 0.3"

You should:
1. Read EXPERIMENT_LOG.md
2. Add entry like:

```markdown
### Experiment 4: Lower Temperature

**Date**: 2026-01-15
**Hypothesis**: Lower temperature (0.3) would produce more consistent output
**Config**: `--temp=0.3` with file-pattern-v1 template

**Results**:
- Format Compliance: 55%
- Score: 62/100
- Run ID: run-XXXXX

**Finding**: Slight improvement in consistency but still ~45% failure rate.
```

3. Update the test run reference table
