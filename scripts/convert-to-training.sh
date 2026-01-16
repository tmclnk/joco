#!/bin/bash
# Convert harness results + raw diffs to training data format
# Usage: ./scripts/convert-to-training.sh <results.jsonl> <raw-commits.jsonl> <output.jsonl>

RESULTS_FILE="$1"
RAW_FILE="$2"
OUTPUT_FILE="$3"

if [ -z "$RESULTS_FILE" ] || [ -z "$RAW_FILE" ] || [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 <results.jsonl> <raw-commits.jsonl> <output.jsonl>"
    exit 1
fi

INSTRUCTION="Generate a conventional commit message for the following git diff.

Follow the conventional commit format: type(scope): description

Types: feat, fix, docs, style, refactor, test, chore, ci, build, perf

Rules:
- Use lowercase for description
- Use imperative mood (add, fix, update)
- No period at end
- Keep under 72 characters"

# Create temporary files for joining
TMP_RESULTS=$(mktemp)
TMP_RAW=$(mktemp)

# Extract id and generated message from results
jq -c '{id: .testResult.testCaseId, generated: .testResult.generatedMessage, valid: .validation.hasValidType}' "$RESULTS_FILE" > "$TMP_RESULTS"

# Extract id and diff from raw commits
jq -c '{id: .id, diff: .diff, repository: .repository, commitHash: .commitHash}' "$RAW_FILE" > "$TMP_RAW"

# Join and format as training data
> "$OUTPUT_FILE"
while IFS= read -r result; do
    id=$(echo "$result" | jq -r '.id')
    generated=$(echo "$result" | jq -r '.generated')
    valid=$(echo "$result" | jq -r '.valid')

    # Skip invalid outputs
    if [ "$valid" != "true" ]; then
        continue
    fi

    # Find matching raw commit
    raw=$(grep "\"id\":\"$id\"" "$TMP_RAW" | head -1)
    if [ -z "$raw" ]; then
        continue
    fi

    diff=$(echo "$raw" | jq -r '.diff')
    repo=$(echo "$raw" | jq -r '.repository')
    hash=$(echo "$raw" | jq -r '.commitHash')

    # Output training record (compact JSON)
    jq -c -n \
        --arg instruction "$INSTRUCTION" \
        --arg input "$diff" \
        --arg output "$generated" \
        --arg id "$id" \
        --arg repo "$repo" \
        --arg hash "$hash" \
        '{
            instruction: $instruction,
            input: $input,
            output: $output,
            metadata: {
                id: $id,
                repository: $repo,
                commit_hash: $hash
            }
        }' >> "$OUTPUT_FILE"
done < "$TMP_RESULTS"

rm -f "$TMP_RESULTS" "$TMP_RAW"

COUNT=$(wc -l < "$OUTPUT_FILE" | tr -d ' ')
echo "Converted $COUNT training examples to $OUTPUT_FILE"
