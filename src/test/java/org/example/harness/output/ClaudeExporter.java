package org.example.harness.output;

import org.example.harness.evaluation.EvaluationMetrics;
import org.example.harness.evaluation.EvaluationResult;
import org.example.harness.runner.TestResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Exports results in markdown format suitable for Claude review.
 */
public class ClaudeExporter {

    private static final Logger logger = LoggerFactory.getLogger(ClaudeExporter.class);
    private static final Path RESULTS_DIR = Path.of("test-results");

    /**
     * Exports results in a markdown format suitable for Claude review.
     */
    public String exportForReview(String runId, List<EvaluationResult> results, EvaluationMetrics metrics) {
        StringBuilder md = new StringBuilder();

        md.append("# Prompt Evaluation Results: ").append(runId).append("\n\n");

        // Summary section
        md.append("## Summary Metrics\n\n");
        md.append("| Metric | Value |\n");
        md.append("|--------|-------|\n");
        md.append(String.format("| Total Tests | %d |%n", metrics.totalTests()));
        md.append(String.format("| Success Rate | %.1f%% |%n", metrics.successRate() * 100));
        md.append(String.format("| Conventional Commit Rate | %.1f%% |%n", metrics.conventionalCommitRate() * 100));
        md.append(String.format("| Scope Inclusion Rate | %.1f%% |%n", metrics.scopeInclusionRate() * 100));
        md.append(String.format("| Length Compliance (72 char) | %.1f%% |%n", metrics.lengthComplianceRate() * 100));
        md.append(String.format("| Recommended Length (50 char) | %.1f%% |%n", metrics.recommendedLengthRate() * 100));
        md.append(String.format("| Average Score | %.1f/100 |%n", metrics.averageScore()));
        md.append(String.format("| Avg Generation Time | %.0f ms |%n", metrics.averageGenerationTimeMs()));
        md.append("\n");

        // Type distribution
        if (!metrics.typeDistribution().isEmpty()) {
            md.append("## Type Distribution\n\n");
            md.append("| Type | Count |\n");
            md.append("|------|-------|\n");
            metrics.typeDistribution().forEach((type, count) ->
                md.append(String.format("| %s | %d |%n", type, count)));
            md.append("\n");
        }

        // Sample comparisons
        md.append("## Sample Results\n\n");

        // Good examples
        md.append("### Successful Examples\n\n");
        results.stream()
            .filter(r -> r.testResult().success() && r.validation().isValid())
            .limit(5)
            .forEach(r -> appendExample(md, r, false));

        if (results.stream().noneMatch(r -> r.testResult().success() && r.validation().isValid())) {
            md.append("_No successful examples with valid conventional commit format._\n\n");
        }

        // Problem examples
        md.append("### Examples with Issues\n\n");
        results.stream()
            .filter(r -> r.testResult().success() && !r.validation().isValid())
            .limit(10)
            .forEach(r -> appendExample(md, r, true));

        // Failed generations
        long failedCount = results.stream().filter(r -> !r.testResult().success()).count();
        if (failedCount > 0) {
            md.append("### Failed Generations\n\n");
            md.append(String.format("_%d test cases failed to generate._\n\n", failedCount));
            results.stream()
                .filter(r -> !r.testResult().success())
                .limit(5)
                .forEach(r -> {
                    md.append(String.format("- **%s**: %s%n",
                        r.testResult().testCaseId(),
                        r.testResult().errorMessage()));
                });
            md.append("\n");
        }

        // Instructions for Claude
        md.append("## Review Instructions\n\n");
        md.append("Please analyze these results and provide:\n");
        md.append("1. **Quality Assessment**: Are the generated messages capturing the essence of the changes?\n");
        md.append("2. **Common Failure Patterns**: What types of commits is the model struggling with?\n");
        md.append("3. **Prompt Improvement Suggestions**: What changes to the prompt might improve results?\n");
        md.append("4. **Type Accuracy**: Is the model choosing appropriate commit types (feat, fix, etc.)?\n");
        md.append("5. **Scope Detection**: How well is the model identifying the affected component/module?\n");

        return md.toString();
    }

    private void appendExample(StringBuilder md, EvaluationResult r, boolean showIssues) {
        TestResult tr = r.testResult();

        md.append(String.format("#### %s%n%n", tr.testCaseId()));
        md.append(String.format("**Expected:** `%s`%n%n", tr.expectedMessage()));
        md.append(String.format("**Generated:** `%s`%n%n", tr.generatedMessage()));

        if (r.typeMatches()) {
            md.append("- Type: MATCH\n");
        } else {
            md.append(String.format("- Type: expected=%s, got=%s%n",
                getExpectedType(tr.expectedMessage()),
                r.validation().type()));
        }

        if (r.scopeMatches()) {
            md.append("- Scope: MATCH\n");
        }

        md.append(String.format("- Score: %d/100%n", r.validation().score()));

        if (showIssues && !r.validation().issues().isEmpty()) {
            md.append(String.format("- **Issues:** %s%n", String.join(", ", r.validation().issues())));
        }

        md.append("\n---\n\n");
    }

    private String getExpectedType(String message) {
        if (message == null) return "?";
        int idx = message.indexOf('(');
        if (idx < 0) idx = message.indexOf(':');
        if (idx > 0) return message.substring(0, idx);
        return "?";
    }

    /**
     * Saves the markdown export to a file.
     */
    public Path saveExport(String runId, String markdown) throws IOException {
        Path exportFile = RESULTS_DIR.resolve(runId).resolve("claude-review.md");
        Files.writeString(exportFile, markdown);
        logger.info("Saved Claude review export to {}", exportFile);
        return exportFile;
    }
}
