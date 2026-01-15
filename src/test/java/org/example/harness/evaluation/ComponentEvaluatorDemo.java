package org.example.harness.evaluation;

/**
 * Demonstration of the CommitComponentEvaluator functionality.
 *
 * Run with: ./mvnw exec:java -Dexec.mainClass=org.example.harness.evaluation.ComponentEvaluatorDemo -Dexec.classpathScope=test
 */
public class ComponentEvaluatorDemo {

    public static void main(String[] args) {
        CommitComponentEvaluator evaluator = new CommitComponentEvaluator();

        System.out.println("=== CommitComponentEvaluator Demo ===\n");

        // Test case 1: Exact match
        demo(evaluator, "Exact Match",
            "feat(core): add new feature",
            "feat(core): add new feature");

        // Test case 2: Type match, different description
        demo(evaluator, "Type Match, Different Description",
            "fix(auth): resolve login issue",
            "fix(auth): handle authentication error");

        // Test case 3: Type mismatch
        demo(evaluator, "Type Mismatch",
            "feat(api): add new endpoint",
            "fix(api): add new endpoint");

        // Test case 4: Scope mismatch
        demo(evaluator, "Scope Mismatch",
            "feat(core): implement feature",
            "feat(api): implement feature");

        // Test case 5: Missing scope in generated
        demo(evaluator, "Missing Scope",
            "feat(core): implement feature",
            "feat: implement feature");

        // Test case 6: No scope in both
        demo(evaluator, "No Scope in Both",
            "feat: implement feature",
            "feat: implement feature");

        // Test case 7: Invalid generated
        demo(evaluator, "Invalid Generated Format",
            "feat(core): add feature",
            "Added a new feature to the system");

        // Test case 8: High description similarity
        demo(evaluator, "High Description Similarity",
            "fix(db): resolve database connection timeout issue",
            "fix(db): fix database connection timeout problem");

        // Test case 9: Low description similarity
        demo(evaluator, "Low Description Similarity",
            "feat(ui): add user registration form",
            "feat(ui): implement login page");

        System.out.println("=== Demo Complete ===");
    }

    private static void demo(CommitComponentEvaluator evaluator, String testName,
                             String expected, String generated) {
        System.out.println("--- " + testName + " ---");
        System.out.println("Expected:  " + expected);
        System.out.println("Generated: " + generated);

        ComponentComparisonResult result = evaluator.compare(expected, generated);

        System.out.println("\nResults:");
        System.out.println("  Both Valid:       " + result.bothValid());
        System.out.println("  Type Matches:     " + result.typeMatches());
        System.out.println("  Scope Matches:    " + result.scopeMatches());
        System.out.println("  Scope Presence:   " + result.scopePresenceMatches());
        System.out.println("  Desc Similarity:  " + String.format("%.3f", result.descriptionSimilarity()));

        if (result.expected().valid()) {
            System.out.println("\n  Expected Components:");
            System.out.println("    Type:  " + result.expected().type());
            System.out.println("    Scope: " + result.expected().scope());
            System.out.println("    Desc:  " + result.expected().description());
        }

        if (result.generated().valid()) {
            System.out.println("\n  Generated Components:");
            System.out.println("    Type:  " + result.generated().type());
            System.out.println("    Scope: " + result.generated().scope());
            System.out.println("    Desc:  " + result.generated().description());
        }

        System.out.println("\n");
    }
}
