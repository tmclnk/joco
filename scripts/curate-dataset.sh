#!/bin/bash
#
# Curate a HuggingFace-compatible dataset from joco test cases
#
# Usage: ./scripts/curate-dataset.sh
#
# Output:
#   dataset/train.jsonl       - Training set (90%)
#   dataset/validation.jsonl  - Validation set (10%)
#   dataset/stats.json        - Dataset statistics
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=== Joco Dataset Curator ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Compile and run the curator
echo "Compiling..."
./mvnw compile test-compile -q

echo "Running dataset curator..."
./mvnw exec:java -Pdataset -q

echo ""
echo "Done! Dataset files are in: $PROJECT_ROOT/dataset/"
echo ""
echo "Files created:"
ls -la "$PROJECT_ROOT/dataset/" 2>/dev/null || echo "No files created yet"
