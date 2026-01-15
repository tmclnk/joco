#!/bin/bash
# Extract benchmark test cases from cloned repositories
# Run setup-benchmark-repos.sh first to clone the repos

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TMP_DIR="$PROJECT_ROOT/tmp"
BENCHMARK_DIR="$PROJECT_ROOT/benchmark"

# Verify repos exist
if [ ! -d "$TMP_DIR/angular" ]; then
    echo "Error: Repos not found. Run setup-benchmark-repos.sh first."
    exit 1
fi

# Build first
echo "Building project..."
cd "$PROJECT_ROOT"
./mvnw -q package -DskipTests

# Create directories
echo "Creating benchmark directories..."
mkdir -p "$BENCHMARK_DIR/format-correctness"
mkdir -p "$BENCHMARK_DIR/content-quality/antirez"
mkdir -p "$BENCHMARK_DIR/content-quality/rich-hickey"
mkdir -p "$BENCHMARK_DIR/content-quality/linus-torvalds"
mkdir -p "$BENCHMARK_DIR/content-quality/bryan-cantrill"
mkdir -p "$BENCHMARK_DIR/content-quality/go-team"
mkdir -p "$BENCHMARK_DIR/content-quality/java-team"

run_harness() {
    ./mvnw -q exec:java -Pharness -Dexec.args="$*"
}

echo ""
echo "=== Extracting format correctness benchmarks ==="
echo "Extracting Angular (conventional commits)..."
run_harness "extract $TMP_DIR/angular $BENCHMARK_DIR/format-correctness/angular-commits.jsonl --max=100 --name=angular"

echo ""
echo "=== Extracting content quality benchmarks ==="

echo "Extracting antirez commits from Redis..."
run_harness "extract $TMP_DIR/redis $BENCHMARK_DIR/content-quality/antirez/redis-commits.jsonl --max=50 --name=redis-antirez --author=antirez --no-filter"

echo "Extracting Rich Hickey commits from Clojure..."
run_harness "extract $TMP_DIR/clojure $BENCHMARK_DIR/content-quality/rich-hickey/clojure-commits.jsonl --max=50 --name=clojure-hickey --author=richhickey@gmail.com --no-filter"

echo "Extracting Linus Torvalds commits from git..."
run_harness "extract $TMP_DIR/git $BENCHMARK_DIR/content-quality/linus-torvalds/git-commits.jsonl --max=50 --name=git-linus --author=torvalds --no-filter"

echo "Extracting Bryan Cantrill commits from Hubris..."
run_harness "extract $TMP_DIR/hubris $BENCHMARK_DIR/content-quality/bryan-cantrill/hubris-commits.jsonl --max=50 --name=hubris-cantrill --author=bryan@oxide.computer --no-filter"

echo "Extracting Go team commits..."
run_harness "extract $TMP_DIR/go $BENCHMARK_DIR/content-quality/go-team/go-commits.jsonl --max=50 --name=go --no-filter"

echo "Extracting Java team commits from OpenJDK..."
run_harness "extract $TMP_DIR/jdk $BENCHMARK_DIR/content-quality/java-team/jdk-commits.jsonl --max=50 --name=jdk --no-filter"

echo ""
echo "=== Extraction complete! ==="
echo "Benchmark data saved to $BENCHMARK_DIR"
find "$BENCHMARK_DIR" -name "*.jsonl" -exec wc -l {} \;
