# joco

A lightweight clone of opencommit that generates meaningful commit messages using Ollama local LLMs.

## Recommended Ollama Models for Low-End Hardware

For a MacBook Air with 8GB RAM, you'll want to use smaller, efficient models. Here are the recommended options based on 2026 benchmarks:

### Primary Recommendation: Qwen2.5-Coder

**qwen2.5-coder:0.5b** (Ultra-lightweight)
```bash
ollama pull qwen2.5-coder:0.5b
```
- Size: ~398MB
- Lightning fast on any hardware
- Surprisingly capable for commit messages
- Perfect if you want minimal resource usage

**qwen2.5-coder:1.5b** (Best balance for 8GB RAM)
```bash
ollama pull qwen2.5-coder:1.5b
```
- Size: ~986MB
- Specifically trained for code understanding and generation
- Very fast inference on limited hardware
- Excellent at understanding diffs and generating concise commit messages
- Most popular choice for lightweight coding assistants

**qwen2.5-coder:3b** (Higher quality)
```bash
ollama pull qwen2.5-coder:3b
```
- Size: ~1.9GB
- Better quality than 1.5b while still being lightweight
- Runs comfortably on 8GB RAM
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

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull your chosen model (see recommendations above)
3. Ensure Java 17+ is installed

### Build

```bash
mvn clean package
```

## Usage

```bash
# Stage your changes
git add .

# Generate commit message
joco
```

## Model Performance Comparison

| Model | Size | Speed on 8GB | Quality | Best For |
|-------|------|--------------|---------|----------|
| qwen2.5-coder:0.5b | ~398MB | Lightning Fast | Decent | Absolute minimal resources |
| qwen2.5-coder:1.5b | ~986MB | Very Fast | Good | Best overall choice for 8GB |
| qwen2.5-coder:3b | ~1.9GB | Fast | Better | Higher quality, still efficient |
| deepseek-coder:1.3b | ~775MB | Very Fast | Good | Alternative to Qwen |
| tinyllama:1.1b | ~638MB | Very Fast | Basic | Ultra-lightweight fallback |

## Tips for 8GB RAM

- Start with the 1.5b model - it's the sweet spot for 8GB systems
- Close other memory-intensive applications when running joco
- The 0.5b and 1.5b models are surprisingly capable for commit messages
- Avoid 7B+ models unless you're willing to close all other apps (they need ~16GB recommended)
- Use Q4_K_M quantization if available for better performance

## Configuration

Create `.joco.config` in your home directory:

```json
{
  "model": "qwen2.5-coder:1.5b",
  "maxTokens": 100,
  "temperature": 0.7
}
```

## License

MIT
