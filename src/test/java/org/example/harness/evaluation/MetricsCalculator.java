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

        // Component metrics tracking
        int typeMatchCount = 0;
        int typeComparisonCount = 0;
        int scopeMatchCount = 0;
        int scopeComparisonCount = 0; // Only when expected has scope
        int scopePresenceMatchCount = 0;
        int validComparisonCount = 0;
        double totalDescriptionSimilarity = 0.0;
        double minDescriptionSimilarity = Double.MAX_VALUE;
        double maxDescriptionSimilarity = Double.MIN_VALUE;

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

            // Process component comparison metrics
            ComponentComparisonResult cc = result.componentComparison();
            if (cc != null && cc.bothValid()) {
                validComparisonCount++;

                // Type accuracy
                typeComparisonCount++;
                if (cc.typeMatches()) {
                    typeMatchCount++;
                }

                // Scope match (only when expected has scope)
                if (cc.expectedHasScope()) {
                    scopeComparisonCount++;
                    if (cc.scopeMatches()) {
                        scopeMatchCount++;
                    }
                }

                // Scope presence match
                if (cc.scopePresenceMatches()) {
                    scopePresenceMatchCount++;
                }

                // Description similarity
                double descSim = cc.descriptionSimilarity();
                totalDescriptionSimilarity += descSim;
                if (descSim < minDescriptionSimilarity) {
                    minDescriptionSimilarity = descSim;
                }
                if (descSim > maxDescriptionSimilarity) {
                    maxDescriptionSimilarity = descSim;
                }
            }
        }

        int failedGenerations = totalTests - successfulGenerations;
        double successRate = (double) successfulGenerations / totalTests;

        // Use successfulGenerations as denominator for quality metrics
        int qualityBase = successfulGenerations > 0 ? successfulGenerations : 1;

        // Compute component metrics
        ComponentMetrics componentMetrics = computeComponentMetrics(
            typeMatchCount, typeComparisonCount,
            scopeMatchCount, scopeComparisonCount,
            scopePresenceMatchCount, validComparisonCount,
            totalDescriptionSimilarity,
            minDescriptionSimilarity, maxDescriptionSimilarity
        );

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
            (double) totalCompletionTokens / totalTests,
            componentMetrics
        );
    }

    /**
     * Computes ComponentMetrics from aggregated values.
     */
    private ComponentMetrics computeComponentMetrics(
            int typeMatchCount, int typeComparisonCount,
            int scopeMatchCount, int scopeComparisonCount,
            int scopePresenceMatchCount, int validComparisonCount,
            double totalDescriptionSimilarity,
            double minDescriptionSimilarity, double maxDescriptionSimilarity) {

        if (validComparisonCount == 0) {
            return ComponentMetrics.empty();
        }

        double typeAccuracyRate = typeComparisonCount > 0
            ? (double) typeMatchCount / typeComparisonCount : 0.0;

        double scopeMatchRate = scopeComparisonCount > 0
            ? (double) scopeMatchCount / scopeComparisonCount : 0.0;

        double scopePresenceMatchRate = validComparisonCount > 0
            ? (double) scopePresenceMatchCount / validComparisonCount : 0.0;

        double avgDescSimilarity = validComparisonCount > 0
            ? totalDescriptionSimilarity / validComparisonCount : 0.0;

        // Handle edge cases for min/max
        if (minDescriptionSimilarity == Double.MAX_VALUE) {
            minDescriptionSimilarity = 0.0;
        }
        if (maxDescriptionSimilarity == Double.MIN_VALUE) {
            maxDescriptionSimilarity = 0.0;
        }

        return new ComponentMetrics(
            typeAccuracyRate, typeMatchCount, typeComparisonCount,
            scopeMatchRate, scopeMatchCount, scopeComparisonCount,
            scopePresenceMatchRate, scopePresenceMatchCount,
            avgDescSimilarity, minDescriptionSimilarity, maxDescriptionSimilarity,
            validComparisonCount
        );
    }

    private EvaluationMetrics emptyMetrics() {
        return new EvaluationMetrics(
            0, 0, 0,
            0.0, 0.0, 0.0, 0.0,
            Map.of(),
            0.0, 0.0, 0.0, 0.0,
            0, 0, 0.0, 0.0,
            ComponentMetrics.empty()
        );
    }
}
