package org.example.harness.generator;

/**
 * Interface for commit message generators.
 * Abstracts the backend (Ollama, Claude Code, etc.) used to generate commit messages.
 */
public interface CommitGenerator {

    /**
     * Generates a commit message from the given prompt.
     *
     * @param prompt The formatted prompt containing the diff and instructions
     * @param config Generation configuration (model, temperature, etc.)
     * @return The generation result containing the commit message and metadata
     * @throws GenerationException if generation fails
     */
    GenerationResult generate(String prompt, GenerationConfig config) throws GenerationException;

    /**
     * Checks if this generator is available and ready to use.
     *
     * @return true if the generator is available, false otherwise
     */
    boolean isAvailable();

    /**
     * Returns the name of this generator backend.
     *
     * @return The backend name (e.g., "ollama", "claude")
     */
    String getBackendName();
}
