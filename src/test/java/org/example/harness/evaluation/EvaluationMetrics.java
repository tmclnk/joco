package org.example.harness.evaluation;

import java.util.Map;

/**
 * Aggregated metrics from evaluating a test run.
 */
public record EvaluationMetrics(
    // Counts
    int totalTests,
    int successfulGenerations,
    int failedGenerations,

    // Structural metrics (as percentages 0.0-1.0)
    double conventionalCommitRate,
    double scopeInclusionRate,
    double lengthComplianceRate,
    double recommendedLengthRate,

    // Type distribution
    Map<String, Integer> typeDistribution,

    // Quality indicators
    double averageSubjectLength,
    double averageScore,
    double successRate,
    double averageGenerationTimeMs,

    // Token efficiency
    int totalPromptTokens,
    int totalCompletionTokens,
    double averagePromptTokens,
    double averageCompletionTokens
) {
    /**
     * Returns a formatted summary string.
     */
    public String toSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Evaluation Metrics ===\n\n");

        sb.append("Generation Results:\n");
        sb.append(String.format("  Total tests: %d%n", totalTests));
        sb.append(String.format("  Successful: %d (%.1f%%)%n", successfulGenerations, successRate * 100));
        sb.append(String.format("  Failed: %d%n", failedGenerations));
        sb.append("\n");

        sb.append("Quality Metrics:\n");
        sb.append(String.format("  Conventional Commit Format: %.1f%%%n", conventionalCommitRate * 100));
        sb.append(String.format("  Scope Included: %.1f%%%n", scopeInclusionRate * 100));
        sb.append(String.format("  Length <= 72 chars: %.1f%%%n", lengthComplianceRate * 100));
        sb.append(String.format("  Length <= 50 chars: %.1f%%%n", recommendedLengthRate * 100));
        sb.append(String.format("  Average Score: %.1f/100%n", averageScore));
        sb.append(String.format("  Average Subject Length: %.1f chars%n", averageSubjectLength));
        sb.append("\n");

        sb.append("Type Distribution:\n");
        typeDistribution.forEach((type, count) ->
            sb.append(String.format("  %s: %d%n", type, count)));
        sb.append("\n");

        sb.append("Performance:\n");
        sb.append(String.format("  Avg Generation Time: %.0f ms%n", averageGenerationTimeMs));
        sb.append(String.format("  Avg Prompt Tokens: %.0f%n", averagePromptTokens));
        sb.append(String.format("  Avg Completion Tokens: %.0f%n", averageCompletionTokens));

        return sb.toString();
    }
}
