#!/bin/bash
#
# Example: How to use the prompt harness for prompt iteration
#
# This script demonstrates common workflows for testing and comparing prompts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "=================================="
echo "Prompt Harness Example Workflows"
echo "=================================="
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running"
    echo "Please start it with: ollama serve"
    exit 1
fi

echo "✓ Ollama is running"
echo ""

# Example 1: Test baseline prompt
echo "Example 1: Test baseline prompt with 5 examples"
echo "================================================"
./scripts/prompt_harness.py \
    --builtin baseline \
    --num-examples 5 \
    --verbose
echo ""

# Example 2: Compare multiple built-in prompts
echo ""
echo "Example 2: Compare all built-in prompts"
echo "========================================"
./scripts/prompt_harness.py \
    --builtin all \
    --num-examples 5
echo ""

# Example 3: Test with different temperatures
echo ""
echo "Example 3: Test strict-format-v1 with different temperatures"
echo "============================================================="
./scripts/prompt_harness.py \
    --builtin strict-format-v1 \
    --temperatures 0.1,0.3,0.5 \
    --num-examples 5
echo ""

# Example 4: Compare prompt files
if [ -f "$REPO_ROOT/prompts/baseline.txt" ] && [ -f "$REPO_ROOT/prompts/strict-format-v1.txt" ]; then
    echo ""
    echo "Example 4: Compare prompt files"
    echo "================================"
    ./scripts/prompt_harness.py \
        --prompts prompts/baseline.txt,prompts/strict-format-v1.txt \
        --num-examples 5
    echo ""
fi

echo ""
echo "=================================="
echo "Examples complete!"
echo ""
echo "Next steps:"
echo "  1. Create your own prompt file in prompts/"
echo "  2. Test it: ./scripts/prompt_harness.py --prompt prompts/yourfile.txt --num-examples 20"
echo "  3. Compare: ./scripts/prompt_harness.py --prompts prompts/baseline.txt,prompts/yourfile.txt --num-examples 20"
echo "  4. Document results in EXPERIMENT_LOG.md"
echo "=================================="
