package org.example.harness.evaluation;

import org.example.harness.runner.TestResult;

/**
 * Complete evaluation for a single test case.
 */
public record EvaluationResult(
    TestResult testResult,
    StructuralValidator.ValidationResult validation,
    boolean typeMatches,
    boolean scopeMatches
) {
    /**
     * Creates an evaluation result from a test result.
     */
    public static EvaluationResult evaluate(TestResult testResult, StructuralValidator validator) {
        if (!testResult.success() || testResult.generatedMessage() == null) {
            return new EvaluationResult(
                testResult,
                validator.validate(""),
                false,
                false
            );
        }

        StructuralValidator.ValidationResult validation =
            validator.validate(testResult.generatedMessage());

        // Check if type matches expected
        StructuralValidator.ValidationResult expectedValidation =
            validator.validate(testResult.expectedMessage());

        boolean typeMatches = validation.type() != null &&
            validation.type().equals(expectedValidation.type());

        boolean scopeMatches = validation.scope() != null &&
            validation.scope().equals(expectedValidation.scope());

        return new EvaluationResult(testResult, validation, typeMatches, scopeMatches);
    }
}
