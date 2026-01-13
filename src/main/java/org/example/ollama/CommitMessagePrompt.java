package org.example.ollama;

/**
 * Token-efficient prompt engineering for commit message generation.
 * Implements conventional commit format with minimal token usage.
 */
public class CommitMessagePrompt {

    private static final String SYSTEM_PROMPT = """
        Generate a concise git commit message from the diff.

        Rules:
        - Use conventional commit format: type(scope): description
        - Types: feat, fix, chore, docs, refactor, test, style, perf
        - Max 50 chars for subject line
        - No body or footer
        - Imperative mood (add, fix, update)
        - Lowercase except proper nouns

        Examples:
        feat(auth): add OAuth2 login
        fix(api): resolve null pointer in handler
        chore(deps): update spring to 3.2.0
        """;

    /**
     * Gets the system prompt for commit message generation.
     * Designed to be token-efficient while providing clear instructions.
     *
     * @return the system prompt string
     */
    public static String getSystemPrompt() {
        return SYSTEM_PROMPT;
    }

    /**
     * Formats a git diff into a user prompt.
     * Keeps token usage minimal by avoiding redundant context.
     *
     * @param gitDiff the git diff output
     * @return formatted user prompt
     * @throws IllegalArgumentException if gitDiff is null or blank
     */
    public static String formatUserPrompt(String gitDiff) {
        if (gitDiff == null || gitDiff.isBlank()) {
            throw new IllegalArgumentException("Git diff cannot be null or blank");
        }

        // Token-efficient format: just the diff with minimal wrapper
        return "Diff:\n" + gitDiff.trim();
    }

    /**
     * Creates a complete prompt by combining system and user prompts.
     * This is useful for models that don't support separate system prompts.
     *
     * @param gitDiff the git diff output
     * @return combined prompt string
     * @throws IllegalArgumentException if gitDiff is null or blank
     */
    public static String createCompletePrompt(String gitDiff) {
        if (gitDiff == null || gitDiff.isBlank()) {
            throw new IllegalArgumentException("Git diff cannot be null or blank");
        }

        return SYSTEM_PROMPT + "\n\n" + formatUserPrompt(gitDiff);
    }

    /**
     * Truncates a git diff to stay within token limits.
     * Preserves the most important parts of the diff (start and file changes).
     *
     * @param gitDiff the git diff output
     * @param maxChars approximate character limit (4 chars ≈ 1 token)
     * @return truncated diff if needed
     * @throws IllegalArgumentException if gitDiff is null or maxChars is not positive
     */
    public static String truncateDiff(String gitDiff, int maxChars) {
        if (gitDiff == null) {
            throw new IllegalArgumentException("Git diff cannot be null");
        }
        if (maxChars <= 0) {
            throw new IllegalArgumentException("Max chars must be positive, got: " + maxChars);
        }

        if (gitDiff.length() <= maxChars) {
            return gitDiff;
        }

        // Keep first portion and add truncation notice
        int keepLength = maxChars - 50; // Reserve space for truncation message
        if (keepLength < 100) {
            keepLength = Math.min(100, maxChars - 20);
        }

        String truncated = gitDiff.substring(0, keepLength);
        return truncated + "\n\n[... diff truncated for token efficiency ...]";
    }
}
