package org.example.harness.runner;

/**
 * Result of running a single test case.
 */
public record TestResult(
    String testCaseId,
    String promptTemplateId,
    String model,
    String expectedMessage,
    String generatedMessage,
    long generationTimeMs,
    int promptTokens,
    int completionTokens,
    boolean success,
    String errorMessage
) {
    /**
     * Creates a successful result.
     */
    public static TestResult success(
        String testCaseId,
        String promptTemplateId,
        String model,
        String expectedMessage,
        String generatedMessage,
        long generationTimeMs,
        int promptTokens,
        int completionTokens
    ) {
        return new TestResult(
            testCaseId,
            promptTemplateId,
            model,
            expectedMessage,
            generatedMessage,
            generationTimeMs,
            promptTokens,
            completionTokens,
            true,
            null
        );
    }

    /**
     * Creates a failed result.
     */
    public static TestResult failure(
        String testCaseId,
        String promptTemplateId,
        String model,
        String expectedMessage,
        long generationTimeMs,
        String errorMessage
    ) {
        return new TestResult(
            testCaseId,
            promptTemplateId,
            model,
            expectedMessage,
            null,
            generationTimeMs,
            0,
            0,
            false,
            errorMessage
        );
    }
}
