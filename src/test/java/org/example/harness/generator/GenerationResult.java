package org.example.harness.generator;

/**
 * Result of a commit message generation.
 */
public record GenerationResult(
    String response,
    long durationMs,
    int promptTokens,
    int completionTokens,
    String backendInfo
) {
    /**
     * Creates a successful result with token counts.
     */
    public static GenerationResult success(
        String response,
        long durationMs,
        int promptTokens,
        int completionTokens,
        String backendInfo
    ) {
        return new GenerationResult(response, durationMs, promptTokens, completionTokens, backendInfo);
    }

    /**
     * Creates a result without token counts (for backends that don't report them).
     */
    public static GenerationResult success(String response, long durationMs, String backendInfo) {
        return new GenerationResult(response, durationMs, 0, 0, backendInfo);
    }
}
