package org.example.harness.evaluation;

/**
 * Result of comparing expected and generated commit message components.
 */
public record ComponentComparisonResult(
    CommitComponents expected,
    CommitComponents generated,
    boolean typeMatches,
    boolean scopeMatches,
    boolean scopePresenceMatches,
    double descriptionSimilarity
) {

    /**
     * Returns true if both expected and generated are valid conventional commits.
     */
    public boolean bothValid() {
        return expected.valid() && generated.valid();
    }

    /**
     * Returns true if the expected message has a scope.
     */
    public boolean expectedHasScope() {
        return expected.hasScope();
    }

    /**
     * Returns a human-readable summary of the comparison.
     */
    public String toSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Component Comparison:\n");
        sb.append(String.format("  Expected: type=%s, scope=%s, desc=%s%n",
            expected.type(), expected.scope(), truncate(expected.description(), 40)));
        sb.append(String.format("  Generated: type=%s, scope=%s, desc=%s%n",
            generated.type(), generated.scope(), truncate(generated.description(), 40)));
        sb.append(String.format("  Type Match: %s%n", typeMatches));
        sb.append(String.format("  Scope Match: %s%n", scopeMatches));
        sb.append(String.format("  Description Similarity: %.2f%n", descriptionSimilarity));
        return sb.toString();
    }

    private static String truncate(String s, int maxLen) {
        if (s == null) return "null";
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }
}
