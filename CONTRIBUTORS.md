# Contributing to joco

Thank you for your interest in contributing to joco!

## Minimal Dependencies Policy

joco is designed to run on resource-constrained hardware (8GB RAM MacBook Air). **Keeping dependencies minimal is a core design principle.**

Before proposing a new dependency:
1. Check if Java's standard library can solve the problem
2. Consider if the functionality is truly necessary
3. Evaluate the dependency's size and transitive dependencies

Current dependencies (intentionally minimal):
- **Gson** - JSON parsing for Ollama API and config files
- **SLF4J + Logback** - Logging framework

New dependencies require strong justification and maintainer approval.

## Branching Strategy

All work should be done in feature branches, not directly on `main`.

**Branch naming convention:** Use the GitHub-assigned branch name:
```
{issue_number}-feature-name
```

Examples:
- `42-add-commit-preview`
- `15-fix-empty-diff-handling`

## Development Workflow

1. Create or claim an issue
2. Create a feature branch from `main`
3. Make your changes
4. Build and test locally: `mvn clean package`
5. Submit a pull request

## Building

```bash
# Standard build
mvn clean package

# Native image (requires GraalVM)
mvn clean package -Pnative
```

## Code Style

Follow the patterns established in the existing codebase:
- Clear, descriptive method and variable names
- Javadoc comments on public classes and methods
- Handle errors gracefully with meaningful messages

## Issue Tracking

This project uses [Beads](https://github.com/anthropics/beads) for issue tracking:

```bash
bd create --title="description" --type=task  # Create an issue
bd list                                       # View issues
bd update <id> --status=in_progress           # Claim work
bd close <id>                                 # Complete work
```

## Release Process

joco uses GitHub Actions for automated releases. The workflow builds:
- **joco.jar** - Fat JAR (requires Java 21+)
- **joco-linux-x64** - Native binary for Linux
- **joco-macos-arm64** - Native binary for macOS (Apple Silicon)
- **joco-windows-x64.exe** - Native binary for Windows

### Creating a Release

1. Ensure all changes are merged to `main`
2. Create and push a version tag:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```
3. GitHub Actions automatically:
   - Builds the fat JAR
   - Builds native binaries for all platforms
   - Creates a GitHub Release with all artifacts

### Manual Release (Emergency)

For emergency releases without a tag:
1. Go to Actions > Release workflow
2. Click "Run workflow"
3. Enter the version number (e.g., `1.0.1`)
4. Click "Run workflow"

### Version Numbering

Follow semantic versioning (semver):
- **MAJOR.MINOR.PATCH** (e.g., `1.2.3`)
- Increment MAJOR for breaking changes
- Increment MINOR for new features
- Increment PATCH for bug fixes

### Supported Platforms

| Platform | Architecture | Artifact |
|----------|--------------|----------|
| Linux | x64 | joco-linux-x64 |
| macOS | ARM64 (Apple Silicon) | joco-macos-arm64 |
| Windows | x64 | joco-windows-x64.exe |
| Any (with Java 21+) | Any | joco.jar |
