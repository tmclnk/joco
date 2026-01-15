[![CI](https://github.com/tmclnk/joco/actions/workflows/ci.yml/badge.svg)](https://github.com/tmclnk/joco/actions/workflows/ci.yml)

# joco

A lightweight clone of opencommit that generates meaningful commit messages using Ollama local LLMs.

## Recommended Ollama Models

joco works with lightweight, efficient models that provide fast inference. Here are the recommended options based on 2026 benchmarks:

### Primary Recommendation: Qwen2.5-Coder

**qwen2.5-coder:0.5b** (Ultra-lightweight)

```bash
ollama pull qwen2.5-coder:0.5b
```

- Size: ~398MB
- Lightning fast on any hardware
- Surprisingly capable for commit messages
- Perfect if you want minimal resource usage

**qwen2.5-coder:1.5b** (Best overall choice)

```bash
ollama pull qwen2.5-coder:1.5b
```

- Size: ~986MB
- Specifically trained for code understanding and generation
- Very fast inference with minimal memory footprint
- Excellent at understanding diffs and generating concise commit messages
- Most popular choice for lightweight coding assistants

**qwen2.5-coder:3b** (Higher quality)

```bash
ollama pull qwen2.5-coder:3b
```

- Size: ~1.9GB
- Better quality than 1.5b while still being lightweight
- Fast inference with modest memory requirements
- Recommended if you want more detailed commit messages

### Alternatives

**deepseek-coder:1.3b**

```bash
ollama pull deepseek-coder:1.3b
```

- Size: ~775MB
- Alternative lightweight code model
- Good code intelligence with minimal resources

**tinyllama:1.1b**

```bash
ollama pull tinyllama:1.1b
```

- Size: ~638MB
- Ultra-lightweight general-purpose model
- Works on extremely constrained hardware

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

## Model Performance Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| qwen2.5-coder:0.5b | ~398MB | Lightning Fast | Decent | Maximum speed |
| qwen2.5-coder:1.5b | ~986MB | Very Fast | Good | Best overall choice |
| qwen2.5-coder:3b | ~1.9GB | Fast | Better | Higher quality, still efficient |
| deepseek-coder:1.3b | ~775MB | Very Fast | Good | Alternative to Qwen |
| tinyllama:1.1b | ~638MB | Very Fast | Basic | Ultra-lightweight fallback |

## Performance Tips

- The 1.5b model is the sweet spot for most users
- Smaller models (0.5b-1.5b) are surprisingly capable for commit messages
- For 7B+ models, ensure you have at least 16GB RAM available
- Use Q4_K_M quantization if available for faster inference

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
