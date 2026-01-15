package org.example.harness.runner;

import java.nio.file.Path;
import java.time.Instant;

/**
 * Configuration for a test run.
 */
public record TestRunConfig(
    String runId,
    String model,
    double temperature,
    int maxTokens,
    String promptTemplateId,
    Path testCasesFile,
    int maxTestCases
) {
    public TestRunConfig {
        if (runId == null || runId.isBlank()) {
            runId = "run-" + Instant.now().toEpochMilli();
        }
        if (model == null || model.isBlank()) {
            model = "qwen2.5-coder:1.5b";
        }
        if (temperature < 0.0 || temperature > 2.0) {
            throw new IllegalArgumentException("Temperature must be between 0.0 and 2.0");
        }
        if (maxTokens < 10 || maxTokens > 500) {
            throw new IllegalArgumentException("maxTokens must be between 10 and 500");
        }
        if (promptTemplateId == null || promptTemplateId.isBlank()) {
            promptTemplateId = "baseline-v1";
        }
        if (testCasesFile == null) {
            testCasesFile = Path.of("test-cases/angular-commits.jsonl");
        }
    }

    /**
     * Creates a default configuration.
     */
    public static TestRunConfig defaults() {
        return new TestRunConfig(
            null,
            "qwen2.5-coder:1.5b",
            0.7,
            100,
            "baseline-v1",
            Path.of("test-cases/angular-commits.jsonl"),
            0
        );
    }

    /**
     * Creates a config with a specific template.
     */
    public TestRunConfig withTemplate(String templateId) {
        return new TestRunConfig(
            runId,
            model,
            temperature,
            maxTokens,
            templateId,
            testCasesFile,
            maxTestCases
        );
    }

    /**
     * Creates a config with a specific model.
     */
    public TestRunConfig withModel(String newModel) {
        return new TestRunConfig(
            runId,
            newModel,
            temperature,
            maxTokens,
            promptTemplateId,
            testCasesFile,
            maxTestCases
        );
    }

    /**
     * Creates a config limited to N test cases.
     */
    public TestRunConfig withMaxTestCases(int max) {
        return new TestRunConfig(
            runId,
            model,
            temperature,
            maxTokens,
            promptTemplateId,
            testCasesFile,
            max
        );
    }
}
