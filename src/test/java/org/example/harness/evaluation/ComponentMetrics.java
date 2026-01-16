package org.example.harness.evaluation;

/**
 * Metrics specifically for commit message component evaluation.
 *
 * Tracks:
 * - Type accuracy rate (how often the generated type matches expected)
 * - Description similarity (average similarity score)
 *
 * Note: Scope matching is intentionally excluded since we generate
 * `type: description` format without scopes.
 */
public record ComponentMetrics(
    // Type metrics
    double typeAccuracyRate,
    int typeMatchCount,
    int typeComparisonCount,

    // Description metrics
    double averageDescriptionSimilarity,
    double minDescriptionSimilarity,
    double maxDescriptionSimilarity,

    // Valid comparisons count (both expected and generated are valid)
    int validComparisonCount
) {

    /**
     * Returns a formatted summary string for these metrics.
     */
    public String toSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Component Metrics:\n");
        sb.append(String.format("  Type Accuracy: %.1f%% (%d/%d)%n",
            typeAccuracyRate * 100, typeMatchCount, typeComparisonCount));
        sb.append(String.format("  Description Similarity: %.2f (min: %.2f, max: %.2f)%n",
            averageDescriptionSimilarity, minDescriptionSimilarity, maxDescriptionSimilarity));
        sb.append(String.format("  Valid Comparisons: %d%n", validComparisonCount));
        return sb.toString();
    }

    /**
     * Creates empty metrics (used when no results are available).
     */
    public static ComponentMetrics empty() {
        return new ComponentMetrics(
            0.0, 0, 0,
            0.0, 0.0, 0.0,
            0
        );
    }
}
