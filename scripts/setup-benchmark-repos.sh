#!/bin/bash
# Setup benchmark repositories for joco evaluation
# Clones reference repos for format correctness and content quality benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TMP_DIR="$PROJECT_ROOT/tmp"

mkdir -p "$TMP_DIR"

echo "=== Cloning benchmark repositories to $TMP_DIR ==="

# Format correctness repos (conventional commits)
echo ""
echo "--- Format Correctness Repos ---"

echo "Cloning Angular (conventional commits gold standard)..."
[ -d "$TMP_DIR/angular" ] || git clone --depth=500 https://github.com/angular/angular.git "$TMP_DIR/angular"

# Content quality repos (notable developers)
echo ""
echo "--- Content Quality Repos ---"

echo "Cloning Redis (antirez)..."
[ -d "$TMP_DIR/redis" ] || git clone --depth=500 https://github.com/redis/redis.git "$TMP_DIR/redis"

echo "Cloning Clojure (Rich Hickey)..."
[ -d "$TMP_DIR/clojure" ] || git clone --depth=500 https://github.com/clojure/clojure.git "$TMP_DIR/clojure"

echo "Cloning git (Linus Torvalds)..."
[ -d "$TMP_DIR/git" ] || git clone --depth=500 https://github.com/git/git.git "$TMP_DIR/git"

echo "Cloning Hubris (Bryan Cantrill)..."
[ -d "$TMP_DIR/hubris" ] || git clone --depth=500 https://github.com/oxidecomputer/hubris.git "$TMP_DIR/hubris"

echo "Cloning Go stdlib..."
[ -d "$TMP_DIR/go" ] || git clone --depth=500 https://github.com/golang/go.git "$TMP_DIR/go"

echo "Cloning OpenJDK..."
[ -d "$TMP_DIR/jdk" ] || git clone --depth=500 https://github.com/openjdk/jdk.git "$TMP_DIR/jdk"

echo ""
echo "=== Done! Repos cloned to $TMP_DIR ==="
ls -la "$TMP_DIR"
