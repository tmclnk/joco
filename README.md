[![CI](https://github.com/tmclnk/joco/actions/workflows/ci.yml/badge.svg)](https://github.com/tmclnk/joco/actions/workflows/ci.yml)

# joco

A lightweight clone of opencommit that generates meaningful commit messages using Ollama local LLMs.

## Recommended Model

```bash
ollama pull qwen2.5-coder:1.5b
```

Other options:

| Model | Size | Notes |
|-------|------|-------|
| qwen2.5-coder:0.5b | ~398MB | Ultra-lightweight |
| qwen2.5-coder:1.5b | ~986MB | **Recommended** |
| qwen2.5-coder:3b | ~1.9GB | Higher quality |
| deepseek-coder:1.3b | ~775MB | Alternative |

## Installation

### Prerequisites

1. **Ollama** - Install from [ollama.ai](https://ollama.ai), then pull your chosen model (see recommendations above)
2. **Java 21+** - Required for building. Recommended: use [SDKMAN](https://sdkman.io) for version management
3. **GraalVM 21** (optional) - Required only for native executable builds

#### Installing Java with SDKMAN

```bash
# Install SDKMAN
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# Install Java 21 (standard JDK for JAR builds)
sdk install java 21.0.5-tem

# Or install GraalVM 21 (for native builds)
sdk install java 21.0.2-graalce
sdk use java 21.0.2-graalce
```

### Build (JAR)

```bash
./mvnw clean package
```

Output: `target/joco.jar` - run with `java -jar target/joco.jar`

### Build (Native Executable)

Requires GraalVM with native-image:

```bash
# Ensure GraalVM is active
sdk use java 21.0.2-graalce

# Build native executable
./mvnw package -Pnative
```

Output: `target/joco` - a standalone native binary with ~50ms startup time

## Usage

```bash
# Stage your changes
git add .

# Generate commit message
joco
```

## Configuration

Create `.joco.config` in your home directory:

```json
{
  "model": "qwen2.5-coder:1.5b",
  "maxTokens": 100,
  "temperature": 0.7
}
```

## Acknowledgments

joco is inspired by [OpenCommit](https://github.com/di-sukharev/opencommit). Thanks to [@di-sukharev](https://github.com/di-sukharev) and the OpenCommit contributors for creating such a useful tool!

## License

MIT
