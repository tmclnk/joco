package org.example.ollama;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Multi-step commit message generator that breaks generation into focused queries:
 * 1. Type classification (with file-pattern-based rules)
 * 2. Description generation
 *
 * This approach achieves 100% format compliance by constraining each step.
 */
public class MultiStepCommitGenerator {

    private static final Logger logger = LoggerFactory.getLogger(MultiStepCommitGenerator.class);

    private final OllamaClient client;
    private final String model;
    private final double temperature;

    // Token limits for each step
    private static final int TYPE_MAX_TOKENS = 10;
    private static final int DESC_MAX_TOKENS = 30;

    // Type classification prompt - file-pattern based
    private static final String TYPE_PROMPT = """
        Classify this git diff. Check file patterns FIRST:

        BUILD (dependencies/build config):
        - package.json, pom.xml, build.gradle, Cargo.toml, go.mod, go.sum
        - *.lock, yarn.lock, Gemfile, requirements.txt, pyproject.toml
        - Makefile, CMakeLists.txt, meson.build, BUILD, WORKSPACE

        DOCS (documentation only):
        - *.md, README*, CONTRIBUTING*, docs/*, doc/*

        TEST (test files):
        - *_test.go, *_test.py, *_test.js, *_test.rs
        - *.spec.*, *.test.*, __tests__/*, test/*, tests/*

        CI (CI/CD config):
        - .github/*, .gitlab-ci*, .circleci/*, Jenkinsfile
        - .travis.yml, azure-pipelines.yml, .drone.yml

        CHORE (config/tooling):
        - CHANGELOG*, .gitignore, .eslintrc*, .prettierrc*
        - renovate.json, dependabot.yml, .editorconfig

        IF NO FILE MATCH, check content:
        - Fixes bug/error/crash/issue -> fix
        - Adds NEW feature/endpoint/API -> feat
        - Code cleanup, no behavior change -> refactor

        Diff:
        %s

        Type (one word):""";

    // Description generation prompt
    private static final String DESC_PROMPT = """
        Write a short commit description (max 10 words) for this %s change.

        Diff:
        %s

        Description (no type prefix, just the description):""";

    /**
     * Creates a new multi-step generator.
     *
     * @param client the Ollama client to use
     * @param model the model name
     * @param temperature the generation temperature
     */
    public MultiStepCommitGenerator(OllamaClient client, String model, double temperature) {
        this.client = client;
        this.model = model;
        this.temperature = temperature;
    }

    /**
     * Generates a commit message using the 2-step approach.
     *
     * @param diff the git diff to generate a message for
     * @return the generated commit message in "type: description" format
     * @throws OllamaException if generation fails
     */
    public String generate(String diff) throws OllamaException {
        logger.info("Starting multi-step generation with model: {}", model);
        long startTime = System.currentTimeMillis();

        // Step 1: Get the type
        logger.debug("Step 1: Determining commit type");
        String typePrompt = TYPE_PROMPT.formatted(truncateDiff(diff, 2000));
        GenerateResponse typeResponse = query(typePrompt, TYPE_MAX_TOKENS);
        String type = cleanType(typeResponse.response());
        logger.info("Step 1 result - Type: {}", type);

        // Step 2: Get the description
        logger.debug("Step 2: Generating description");
        String descPrompt = DESC_PROMPT.formatted(type, truncateDiff(diff, 2000));
        GenerateResponse descResponse = query(descPrompt, DESC_MAX_TOKENS);
        String description = cleanDescription(descResponse.response());
        logger.info("Step 2 result - Description: {}", description);

        // Assemble final commit message
        String commitMessage = type + ": " + description;

        long durationMs = System.currentTimeMillis() - startTime;
        logger.info("Multi-step generation complete in {}ms: {}", durationMs, commitMessage);

        return commitMessage;
    }

    private GenerateResponse query(String prompt, int maxTokens) throws OllamaException {
        GenerateRequest request = new GenerateRequest(
            model,
            prompt,
            false,
            new GenerateRequest.Options(temperature, maxTokens)
        );
        return client.generate(request);
    }

    private String truncateDiff(String diff, int maxChars) {
        if (diff.length() <= maxChars) {
            return diff;
        }
        return diff.substring(0, maxChars) + "\n[...truncated...]";
    }

    private String cleanType(String response) {
        String cleaned = response.trim().toLowerCase();
        String[] validTypes = {"feat", "fix", "docs", "refactor", "test", "build", "ci", "chore"};
        for (String type : validTypes) {
            if (cleaned.contains(type)) {
                return type;
            }
        }
        logger.warn("Could not determine type from response: '{}', defaulting to 'chore'", response);
        return "chore";
    }

    private String cleanDescription(String response) {
        String cleaned = response.trim();
        // Remove quotes if present
        cleaned = cleaned.replaceAll("^\"|\"$", "");
        cleaned = cleaned.replaceAll("^'|'$", "");
        // Remove any type prefix the model might have added (case-insensitive)
        cleaned = cleaned.replaceAll("(?i)^(feat|fix|docs|refactor|test|build|ci|chore)(\\([^)]*\\))?:\\s*", "");
        // Also remove standalone type words at the start
        cleaned = cleaned.replaceAll("(?i)^(feat|fix|docs|refactor|test|build|ci|chore)\\s+", "");
        // Ensure lowercase first letter
        if (!cleaned.isEmpty()) {
            cleaned = Character.toLowerCase(cleaned.charAt(0)) + cleaned.substring(1);
        }
        // Remove trailing period
        cleaned = cleaned.replaceAll("\\.$", "");
        // Truncate if too long
        if (cleaned.length() > 60) {
            cleaned = cleaned.substring(0, 57) + "...";
        }
        return cleaned;
    }
}
