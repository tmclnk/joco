package org.example.harness.evaluation;

import org.example.harness.runner.TestResult;

/**
 * Complete evaluation for a single test case.
 */
public record EvaluationResult(
    TestResult testResult,
    StructuralValidator.ValidationResult validation,
    boolean typeMatches,
    boolean scopeMatches,
    ComponentComparisonResult componentComparison
) {
    /**
     * Creates an evaluation result from a test result.
     */
    public static EvaluationResult evaluate(TestResult testResult, StructuralValidator validator) {
        return evaluate(testResult, validator, new CommitComponentEvaluator());
    }

    /**
     * Creates an evaluation result from a test result with a custom component evaluator.
     */
    public static EvaluationResult evaluate(
            TestResult testResult,
            StructuralValidator validator,
            CommitComponentEvaluator componentEvaluator) {

        if (!testResult.success() || testResult.generatedMessage() == null) {
            return new EvaluationResult(
                testResult,
                validator.validate(""),
                false,
                false,
                componentEvaluator.compare(testResult.expectedMessage(), "")
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

        // Perform component-level comparison
        ComponentComparisonResult componentComparison =
            componentEvaluator.compare(testResult.expectedMessage(), testResult.generatedMessage());

        return new EvaluationResult(testResult, validation, typeMatches, scopeMatches, componentComparison);
    }
}
