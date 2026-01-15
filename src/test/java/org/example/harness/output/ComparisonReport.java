package org.example.harness.output;

import org.example.harness.evaluation.EvaluationMetrics;

/**
 * Generates comparison reports between test runs.
 */
public class ComparisonReport {

    /**
     * Generates a markdown comparison between two test runs.
     */
    public String compare(String runId1, String runId2,
                          EvaluationMetrics metrics1, EvaluationMetrics metrics2) {
        StringBuilder md = new StringBuilder();

        md.append("# Comparison: ").append(runId1).append(" vs ").append(runId2).append("\n\n");

        md.append("## Summary\n\n");
        md.append("| Metric | ").append(runId1).append(" | ").append(runId2).append(" | Delta |\n");
        md.append("|--------|-------|-------|-------|\n");

        appendRow(md, "Success Rate", metrics1.successRate(), metrics2.successRate());
        appendRow(md, "Conventional Commit", metrics1.conventionalCommitRate(), metrics2.conventionalCommitRate());
        appendRow(md, "Scope Inclusion", metrics1.scopeInclusionRate(), metrics2.scopeInclusionRate());
        appendRow(md, "Length Compliance", metrics1.lengthComplianceRate(), metrics2.lengthComplianceRate());
        appendRow(md, "Recommended Length", metrics1.recommendedLengthRate(), metrics2.recommendedLengthRate());
        appendScoreRow(md, "Average Score", metrics1.averageScore(), metrics2.averageScore());
        appendTimeRow(md, "Avg Gen Time (ms)", metrics1.averageGenerationTimeMs(), metrics2.averageGenerationTimeMs());
        appendTokenRow(md, "Avg Completion Tokens", metrics1.averageCompletionTokens(), metrics2.averageCompletionTokens());

        md.append("\n## Type Distribution Comparison\n\n");
        md.append("| Type | ").append(runId1).append(" | ").append(runId2).append(" |\n");
        md.append("|------|-------|-------|\n");

        // Combine all types from both runs
        var allTypes = new java.util.TreeSet<String>();
        allTypes.addAll(metrics1.typeDistribution().keySet());
        allTypes.addAll(metrics2.typeDistribution().keySet());

        for (String type : allTypes) {
            int count1 = metrics1.typeDistribution().getOrDefault(type, 0);
            int count2 = metrics2.typeDistribution().getOrDefault(type, 0);
            md.append(String.format("| %s | %d | %d |%n", type, count1, count2));
        }

        md.append("\n## Interpretation\n\n");
        md.append(generateInterpretation(metrics1, metrics2));

        return md.toString();
    }

    private void appendRow(StringBuilder md, String name, double v1, double v2) {
        double delta = (v2 - v1) * 100;
        String deltaStr = formatDelta(delta);
        md.append(String.format("| %s | %.1f%% | %.1f%% | %s |%n",
            name, v1 * 100, v2 * 100, deltaStr));
    }

    private void appendScoreRow(StringBuilder md, String name, double v1, double v2) {
        double delta = v2 - v1;
        String deltaStr = formatDelta(delta);
        md.append(String.format("| %s | %.1f | %.1f | %s |%n", name, v1, v2, deltaStr));
    }

    private void appendTimeRow(StringBuilder md, String name, double v1, double v2) {
        double delta = v2 - v1;
        String deltaStr = delta >= 0 ?
            String.format("+%.0f", delta) :
            String.format("%.0f", delta);
        md.append(String.format("| %s | %.0f | %.0f | %s |%n", name, v1, v2, deltaStr));
    }

    private void appendTokenRow(StringBuilder md, String name, double v1, double v2) {
        double delta = v2 - v1;
        String deltaStr = delta >= 0 ?
            String.format("+%.1f", delta) :
            String.format("%.1f", delta);
        md.append(String.format("| %s | %.1f | %.1f | %s |%n", name, v1, v2, deltaStr));
    }

    private String formatDelta(double delta) {
        if (delta > 0) {
            return String.format("+%.1f%%", delta);
        } else {
            return String.format("%.1f%%", delta);
        }
    }

    private String generateInterpretation(EvaluationMetrics m1, EvaluationMetrics m2) {
        StringBuilder sb = new StringBuilder();

        // Overall quality change
        double scoreDelta = m2.averageScore() - m1.averageScore();
        if (scoreDelta > 5) {
            sb.append("- **Overall**: The second run shows improved quality (+")
              .append(String.format("%.1f", scoreDelta)).append(" points)\n");
        } else if (scoreDelta < -5) {
            sb.append("- **Overall**: The second run shows decreased quality (")
              .append(String.format("%.1f", scoreDelta)).append(" points)\n");
        } else {
            sb.append("- **Overall**: Quality is similar between runs\n");
        }

        // Format compliance
        double formatDelta = m2.conventionalCommitRate() - m1.conventionalCommitRate();
        if (Math.abs(formatDelta) > 0.05) {
            sb.append("- **Format**: Conventional commit compliance ")
              .append(formatDelta > 0 ? "improved" : "decreased")
              .append(" by ").append(String.format("%.1f%%", Math.abs(formatDelta) * 100))
              .append("\n");
        }

        // Token efficiency
        double tokenDelta = m2.averageCompletionTokens() - m1.averageCompletionTokens();
        if (Math.abs(tokenDelta) > 5) {
            sb.append("- **Efficiency**: Token usage ")
              .append(tokenDelta > 0 ? "increased" : "decreased")
              .append(" by ").append(String.format("%.1f", Math.abs(tokenDelta)))
              .append(" tokens on average\n");
        }

        return sb.toString();
    }
}
