package org.example.harness.evaluation;

import org.example.harness.runner.TestResult;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Calculates aggregate metrics from evaluation results.
 */
public class MetricsCalculator {

    /**
     * Computes metrics from a list of evaluation results.
     */
    public EvaluationMetrics compute(List<EvaluationResult> results) {
        if (results.isEmpty()) {
            return emptyMetrics();
        }

        int totalTests = results.size();
        int successfulGenerations = 0;
        int conventionalCommitCount = 0;
        int scopeCount = 0;
        int lengthCompliantCount = 0;
        int recommendedLengthCount = 0;

        int totalSubjectLength = 0;
        int totalScore = 0;
        long totalGenerationTime = 0;
        int totalPromptTokens = 0;
        int totalCompletionTokens = 0;

        Map<String, Integer> typeDistribution = new HashMap<>();

        for (EvaluationResult result : results) {
            TestResult tr = result.testResult();
            StructuralValidator.ValidationResult v = result.validation();

            if (tr.success()) {
                successfulGenerations++;
            }

            totalGenerationTime += tr.generationTimeMs();
            totalPromptTokens += tr.promptTokens();
            totalCompletionTokens += tr.completionTokens();

            if (v.hasValidType()) {
                conventionalCommitCount++;
            }
            if (v.hasScope()) {
                scopeCount++;
            }
            if (v.subjectWithinLimit()) {
                lengthCompliantCount++;
            }
            if (v.subjectWithinRecommended()) {
                recommendedLengthCount++;
            }

            totalSubjectLength += v.subjectLength();
            totalScore += v.score();

            if (v.type() != null) {
                typeDistribution.merge(v.type(), 1, Integer::sum);
            }
        }

        int failedGenerations = totalTests - successfulGenerations;
        double successRate = (double) successfulGenerations / totalTests;

        // Use successfulGenerations as denominator for quality metrics
        int qualityBase = successfulGenerations > 0 ? successfulGenerations : 1;

        return new EvaluationMetrics(
            totalTests,
            successfulGenerations,
            failedGenerations,
            (double) conventionalCommitCount / qualityBase,
            (double) scopeCount / qualityBase,
            (double) lengthCompliantCount / qualityBase,
            (double) recommendedLengthCount / qualityBase,
            typeDistribution,
            (double) totalSubjectLength / qualityBase,
            (double) totalScore / qualityBase,
            successRate,
            (double) totalGenerationTime / totalTests,
            totalPromptTokens,
            totalCompletionTokens,
            (double) totalPromptTokens / totalTests,
            (double) totalCompletionTokens / totalTests
        );
    }

    private EvaluationMetrics emptyMetrics() {
        return new EvaluationMetrics(
            0, 0, 0,
            0.0, 0.0, 0.0, 0.0,
            Map.of(),
            0.0, 0.0, 0.0, 0.0,
            0, 0, 0.0, 0.0
        );
    }
}
