package org.example.harness.runner;

import org.example.harness.data.TestCase;
import org.example.harness.data.TestCaseLoader;
import org.example.harness.prompt.PromptTemplate;
import org.example.harness.prompt.PromptTemplateRegistry;
import org.example.ollama.GenerateRequest;
import org.example.ollama.GenerateResponse;
import org.example.ollama.OllamaClient;
import org.example.util.MessageValidator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates test execution against Ollama.
 */
public class TestRunner {

    private static final Logger logger = LoggerFactory.getLogger(TestRunner.class);

    private final OllamaClient ollamaClient;
    private final PromptTemplateRegistry registry;
    private final TestCaseLoader loader;

    public TestRunner(OllamaClient client) {
        this.ollamaClient = client;
        this.registry = new PromptTemplateRegistry();
        this.loader = new TestCaseLoader();
    }

    public TestRunner(OllamaClient client, PromptTemplateRegistry registry) {
        this.ollamaClient = client;
        this.registry = registry;
        this.loader = new TestCaseLoader();
    }

    /**
     * Runs all test cases against a specific prompt template.
     */
    public List<TestResult> runTests(TestRunConfig config) throws IOException {
        List<TestCase> testCases = loader.loadFromFile(config.testCasesFile());

        if (config.maxTestCases() > 0 && config.maxTestCases() < testCases.size()) {
            testCases = testCases.subList(0, config.maxTestCases());
        }

        PromptTemplate template = registry.get(config.promptTemplateId());
        List<TestResult> results = new ArrayList<>();

        logger.info("Running {} test cases with template '{}' and model '{}'",
            testCases.size(), config.promptTemplateId(), config.model());

        for (int i = 0; i < testCases.size(); i++) {
            TestCase testCase = testCases.get(i);
            TestResult result = runSingleTest(testCase, template, config);
            results.add(result);

            // Progress logging
            String status = result.success() ? "OK" : "FAIL";
            logger.info("[{}/{}] {} - {} -> {}",
                i + 1, testCases.size(),
                status,
                testCase.id(),
                result.generatedMessage() != null ?
                    truncate(result.generatedMessage(), 50) :
                    result.errorMessage());
        }

        // Summary
        long successCount = results.stream().filter(TestResult::success).count();
        logger.info("Completed: {}/{} successful", successCount, results.size());

        return results;
    }

    /**
     * Runs a single test case.
     */
    public TestResult runSingleTest(TestCase testCase, PromptTemplate template, TestRunConfig config) {
        long startTime = System.currentTimeMillis();

        try {
            String prompt = template.generatePrompt(testCase.diff());

            GenerateRequest request = new GenerateRequest(
                config.model(),
                prompt,
                false,
                new GenerateRequest.Options(config.temperature(), config.maxTokens())
            );

            GenerateResponse response = ollamaClient.generate(request);
            String rawResponse = response.response();

            // Clean the response
            String cleaned = cleanResponse(rawResponse);

            return TestResult.success(
                testCase.id(),
                template.getId(),
                config.model(),
                testCase.expectedMessage(),
                cleaned,
                System.currentTimeMillis() - startTime,
                response.promptEvalCount() != null ? response.promptEvalCount() : 0,
                response.evalCount() != null ? response.evalCount() : 0
            );

        } catch (Exception e) {
            logger.warn("Test case {} failed: {}", testCase.id(), e.getMessage());
            return TestResult.failure(
                testCase.id(),
                template.getId(),
                config.model(),
                testCase.expectedMessage(),
                System.currentTimeMillis() - startTime,
                e.getMessage()
            );
        }
    }

    /**
     * Cleans the raw LLM response to extract just the commit message.
     */
    private String cleanResponse(String raw) {
        if (raw == null || raw.isBlank()) {
            return "";
        }

        // Use the existing MessageValidator for cleaning
        try {
            return MessageValidator.validateAndClean(raw);
        } catch (Exception e) {
            // If validation fails, do basic cleanup
            return raw.trim().lines().findFirst().orElse(raw.trim());
        }
    }

    private String truncate(String s, int maxLen) {
        if (s == null) return "";
        if (s.length() <= maxLen) return s;
        return s.substring(0, maxLen - 3) + "...";
    }
}
