package org.example.harness;

import org.example.harness.data.TestCase;
import org.example.harness.data.TestCaseExtractor;
import org.example.harness.data.TestCaseLoader;
import org.example.harness.evaluation.EvaluationMetrics;
import org.example.harness.evaluation.EvaluationResult;
import org.example.harness.evaluation.MetricsCalculator;
import org.example.harness.evaluation.StructuralValidator;
import org.example.harness.output.ClaudeExporter;
import org.example.harness.output.ComparisonReport;
import org.example.harness.output.ResultStore;
import org.example.harness.prompt.PromptTemplate;
import org.example.harness.prompt.PromptTemplateRegistry;
import org.example.harness.runner.TestRunConfig;
import org.example.harness.runner.TestResult;
import org.example.harness.runner.TestRunner;
import org.example.ollama.OllamaClient;

import java.nio.file.Path;
import java.util.List;

/**
 * CLI entry point for the prompt test harness.
 *
 * Usage:
 *   harness extract <repo-path> <output.jsonl> [--max=N] [--name=repo-name]
 *   harness run [--template=ID] [--model=MODEL] [--max=N] [--cases=FILE]
 *   harness compare <run1> <run2>
 *   harness templates
 *   harness runs
 */
public class HarnessMain {

    public static void main(String[] args) {
        if (args.length == 0) {
            printUsage();
            return;
        }

        String command = args[0];

        try {
            switch (command) {
                case "extract" -> runExtract(args);
                case "run" -> runTests(args);
                case "compare" -> runCompare(args);
                case "templates" -> listTemplates();
                case "runs" -> listRuns();
                case "help", "--help", "-h" -> printUsage();
                default -> {
                    System.err.println("Unknown command: " + command);
                    printUsage();
                    System.exit(1);
                }
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private static void runExtract(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: harness extract <repo-path> <output.jsonl> [--max=N] [--name=repo-name]");
            System.exit(1);
        }

        Path repoPath = Path.of(args[1]);
        Path outputFile = Path.of(args[2]);

        int maxCommits = 50;
        String repoName = repoPath.getFileName().toString();

        for (int i = 3; i < args.length; i++) {
            if (args[i].startsWith("--max=")) {
                maxCommits = Integer.parseInt(args[i].substring(6));
            } else if (args[i].startsWith("--name=")) {
                repoName = args[i].substring(7);
            }
        }

        System.out.println("Extracting test cases from: " + repoPath);
        System.out.println("Repository name: " + repoName);
        System.out.println("Max commits: " + maxCommits);

        TestCaseExtractor extractor = new TestCaseExtractor(repoPath, repoName);
        List<TestCase> testCases = extractor.extract(maxCommits, 50000); // 50k char max diff

        TestCaseLoader loader = new TestCaseLoader();
        loader.saveToFile(testCases, outputFile);

        System.out.println("\nExtracted " + testCases.size() + " test cases to: " + outputFile);
    }

    private static void runTests(String[] args) throws Exception {
        // Parse arguments
        String templateId = "baseline-v1";
        String model = "qwen2.5-coder:1.5b";
        int maxTests = 0;
        double temperature = 0.7;
        Path casesFile = Path.of("test-cases/angular-commits.jsonl");
        String runId = null;

        for (int i = 1; i < args.length; i++) {
            if (args[i].startsWith("--template=")) {
                templateId = args[i].substring(11);
            } else if (args[i].startsWith("--model=")) {
                model = args[i].substring(8);
            } else if (args[i].startsWith("--max=")) {
                maxTests = Integer.parseInt(args[i].substring(6));
            } else if (args[i].startsWith("--cases=")) {
                casesFile = Path.of(args[i].substring(8));
            } else if (args[i].startsWith("--run-id=")) {
                runId = args[i].substring(9);
            } else if (args[i].startsWith("--temp=")) {
                temperature = Double.parseDouble(args[i].substring(7));
            }
        }

        // Check Ollama availability
        OllamaClient client = new OllamaClient();
        if (!client.isAvailable()) {
            System.err.println("Error: Ollama is not running.");
            System.err.println("Start with: ollama serve");
            System.exit(1);
        }

        // Build config
        TestRunConfig config = new TestRunConfig(
            runId,
            model,
            temperature,
            100,
            templateId,
            casesFile,
            maxTests
        );

        System.out.println("=== Test Run Configuration ===");
        System.out.println("Run ID: " + config.runId());
        System.out.println("Template: " + config.promptTemplateId());
        System.out.println("Model: " + config.model());
        System.out.println("Temperature: " + config.temperature());
        System.out.println("Test cases: " + config.testCasesFile());
        System.out.println("Max tests: " + (config.maxTestCases() > 0 ? config.maxTestCases() : "all"));
        System.out.println();

        // Run tests
        TestRunner runner = new TestRunner(client);
        List<TestResult> results = runner.runTests(config);

        // Evaluate results
        StructuralValidator validator = new StructuralValidator();
        List<EvaluationResult> evaluated = results.stream()
            .map(r -> EvaluationResult.evaluate(r, validator))
            .toList();

        // Compute metrics
        MetricsCalculator calculator = new MetricsCalculator();
        EvaluationMetrics metrics = calculator.compute(evaluated);

        // Store results
        ResultStore store = new ResultStore();
        Path outputDir = store.saveResults(config.runId(), evaluated, metrics);

        // Save prompt template for tracking
        PromptTemplateRegistry registry = new PromptTemplateRegistry();
        PromptTemplate template = registry.get(config.promptTemplateId());
        store.savePromptTemplate(config.runId(), template.getId(),
            template.getDescription(), template.getTemplateText());

        // Export for Claude review
        ClaudeExporter exporter = new ClaudeExporter();
        String markdown = exporter.exportForReview(config.runId(), evaluated, metrics);
        exporter.saveExport(config.runId(), markdown);

        // Print summary
        System.out.println();
        System.out.println(metrics.toSummary());
        System.out.println("Results saved to: " + outputDir);
        System.out.println("Claude review: " + outputDir.resolve("claude-review.md"));
    }

    private static void runCompare(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: harness compare <run1> <run2>");
            System.exit(1);
        }

        String runId1 = args[1];
        String runId2 = args[2];

        ResultStore store = new ResultStore();
        EvaluationMetrics metrics1 = store.loadMetrics(runId1);
        EvaluationMetrics metrics2 = store.loadMetrics(runId2);

        ComparisonReport report = new ComparisonReport();
        String comparison = report.compare(runId1, runId2, metrics1, metrics2);

        System.out.println(comparison);
    }

    private static void listTemplates() {
        PromptTemplateRegistry registry = new PromptTemplateRegistry();

        System.out.println("Available prompt templates:\n");
        for (PromptTemplate template : registry.getAll()) {
            System.out.println("  " + template.getId());
            System.out.println("    " + template.getDescription());
            System.out.println();
        }
    }

    private static void listRuns() throws Exception {
        ResultStore store = new ResultStore();
        List<String> runs = store.listRuns();

        if (runs.isEmpty()) {
            System.out.println("No test runs found in test-results/");
            return;
        }

        System.out.println("Available test runs:\n");
        for (String runId : runs) {
            System.out.println("  " + runId);
        }
    }

    private static void printUsage() {
        System.out.println("""
            Joco Prompt Test Harness

            Usage:
              harness extract <repo-path> <output.jsonl> [options]
                  Extract test cases from a git repository.
                  Options:
                    --max=N       Maximum commits to extract (default: 50)
                    --name=NAME   Repository name (default: directory name)

              harness run [options]
                  Run tests with a prompt template.
                  Options:
                    --template=ID   Prompt template ID (default: baseline-v1)
                    --model=MODEL   Ollama model (default: qwen2.5-coder:1.5b)
                    --max=N         Limit test cases (default: all)
                    --cases=FILE    Test cases file (default: test-cases/angular-commits.jsonl)
                    --run-id=ID     Custom run ID (default: timestamp)

              harness compare <run1> <run2>
                  Compare metrics between two test runs.

              harness templates
                  List available prompt templates.

              harness runs
                  List previous test runs.

            Examples:
              # Extract test cases from Angular
              harness extract /tmp/angular test-cases/angular-commits.jsonl --max=100

              # Run tests with baseline prompt
              harness run

              # Run tests with few-shot prompt
              harness run --template=few-shot-v1 --max=20

              # Compare two runs
              harness compare run-1234567890 run-1234567891
            """);
    }
}
