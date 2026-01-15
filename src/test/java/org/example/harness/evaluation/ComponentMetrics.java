package org.example.harness.evaluation;

/**
 * Metrics specifically for commit message component evaluation.
 *
 * Tracks:
 * - Type accuracy rate (how often the generated type matches expected)
 * - Scope match rate (when expected has scope, how often generated matches)
 * - Scope presence rate (how often scope presence/absence matches)
 * - Description similarity (average similarity score)
 */
public record ComponentMetrics(
    // Type metrics
    double typeAccuracyRate,
    int typeMatchCount,
    int typeComparisonCount,

    // Scope metrics
    double scopeMatchRate,
    int scopeMatchCount,
    int scopeComparisonCount,

    // Scope presence metrics (both have or both don't have scope)
    double scopePresenceMatchRate,
    int scopePresenceMatchCount,

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
        sb.append(String.format("  Scope Match (when expected): %.1f%% (%d/%d)%n",
            scopeMatchRate * 100, scopeMatchCount, scopeComparisonCount));
        sb.append(String.format("  Scope Presence Match: %.1f%% (%d/%d)%n",
            scopePresenceMatchRate * 100, scopePresenceMatchCount, validComparisonCount));
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
            0.0, 0, 0,
            0.0, 0,
            0.0, 0.0, 0.0,
            0
        );
    }
}
