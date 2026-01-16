# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**joco** is a lightweight clone of opencommit that generates meaningful commit messages using Ollama local LLMs. It's a Java-based CLI tool designed to work with limited hardware (8GB RAM MacBook Air) using small, efficient language models.

## CRITICAL: Build Commands

NEVER use `mvn`. ALWAYS use `./mvnw` (the Maven wrapper in this project).

- Build: `./mvnw clean install`
- Test: `./mvnw test`
- Package: `./mvnw package`

The system `mvn` command should NOT be used under any circumstances.

### Run

The application is intended to be run as a CLI tool after building:

```bash
# Stage changes first
git add .

# Run joco to generate commit message
joco
```

## Architecture

### Technology Stack

- **Language**: Java 25
- **Build Tool**: Maven
- **LLM Integration**: Ollama (local inference)
- **Target Models**: Qwen2.5-Coder (0.5b, 1.5b, 3b), DeepSeek-Coder, TinyLlama

### Project Structure

- `src/main/java/org/example/Main.java` - Main entry point (currently contains placeholder code)
- `pom.xml` - Maven configuration (Java 25, minimal dependencies currently)
- No test suite implemented yet

### Configuration

The tool expects a `.joco.config` file in the user's home directory with:

- `model`: Ollama model name (default: qwen2.5-coder:1.5b)
- `maxTokens`: Maximum tokens for generation (default: 100)
- `temperature`: Generation temperature (default: 0.7)

### Core Functionality (To Be Implemented)

1. **Git Diff Analysis**: Read staged changes using `git diff --staged`
2. **Ollama Integration**: Send diff context to local Ollama API
3. **Commit Message Generation**: Generate concise, meaningful commit messages
4. **User Interaction**: Present generated message for approval/editing

### Design Constraints

- Must run efficiently on 8GB RAM systems
- Prefer smaller models (0.5b-3b parameters) over larger ones
- Keep dependencies minimal to reduce footprint
- Fast startup time is important for CLI tool usability

## Issue Tracking

This project uses **Beads** for AI-native local development tracking. Issues are stored in `.beads/issues.jsonl` and synced with git.

When working on a github issue, create a branch and link it back to the github issue. Branches should be of the form `P{issue_number}}-description-of-issue`.

### Common Beads Commands

```bash
bd create "issue description"    # Create new issue
bd list                          # View all issues
bd show <issue-id>               # View issue details
bd update <issue-id> --status in_progress
bd sync                          # Sync with remote
```

## Recommended Models

When testing or developing, use these Ollama models optimized for 8GB RAM:

- **qwen2.5-coder:1.5b** (986MB) - Best balance
- **qwen2.5-coder:0.5b** (398MB) - Ultra-lightweight
- **qwen2.5-coder:3b** (1.9GB) - Higher quality

Avoid 7B+ models unless working on a machine with 16GB+ RAM.
