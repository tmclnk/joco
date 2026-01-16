package org.example.harness.generator;

import org.example.ollama.GenerateRequest;
import org.example.ollama.GenerateResponse;
import org.example.ollama.OllamaClient;
import org.example.ollama.OllamaException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Multi-step commit message generator that breaks generation into separate queries:
 * 1. Determine the commit type (feat, fix, docs, etc.)
 * 2. Determine the scope (optional)
 * 3. Generate the description
 *
 * This approach uses focused prompts with lower token limits for each step,
 * which may work better with small models.
 */
public class MultiStepGenerator implements CommitGenerator {

    private static final Logger logger = LoggerFactory.getLogger(MultiStepGenerator.class);

    private final OllamaClient client;

    // Focused prompt for type classification - comprehensive file-based criteria
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

    // Focused prompt for scope extraction
    private static final String SCOPE_PROMPT = """
        Based on this git diff, what is the main module or component affected?

        Reply with ONLY a single short word (e.g., auth, api, ui, core, cli).
        If unclear or multiple areas, reply: none

        Diff:
        %s

        Scope:""";

    // Focused prompt for description generation
    private static final String DESC_PROMPT_WITH_SCOPE = """
        Write a short commit description (max 10 words) for this %s(%s) change.

        Diff:
        %s

        Description (no type prefix, just the description):""";

    private static final String DESC_PROMPT_NO_SCOPE = """
        Write a short commit description (max 10 words) for this %s change.

        Diff:
        %s

        Description (no type prefix, just the description):""";

    public MultiStepGenerator() {
        this(new OllamaClient());
    }

    public MultiStepGenerator(OllamaClient client) {
        this.client = client;
    }

    @Override
    public GenerationResult generate(String diff, GenerationConfig config) throws GenerationException {
        logger.info("Starting multi-step generation with model: {}", config.model());

        long startTime = System.currentTimeMillis();
        int totalPromptTokens = 0;
        int totalCompletionTokens = 0;

        try {
            // Step 1: Get the type
            logger.debug("Step 1: Determining commit type");
            String typePrompt = TYPE_PROMPT.formatted(truncateDiff(diff, 2000));
            GenerateResponse typeResponse = query(typePrompt, config.model(), 10);
            String type = cleanType(typeResponse.response());
            totalPromptTokens += typeResponse.promptEvalCount() != null ? typeResponse.promptEvalCount() : 0;
            totalCompletionTokens += typeResponse.evalCount() != null ? typeResponse.evalCount() : 0;
            logger.info("Step 1 result - Type: {}", type);

            // Step 2: Get the description (no scope for now)
            logger.debug("Step 2: Generating description");
            String descPrompt = DESC_PROMPT_NO_SCOPE.formatted(type, truncateDiff(diff, 2000));
            GenerateResponse descResponse = query(descPrompt, config.model(), 30);
            String description = cleanDescription(descResponse.response());
            totalPromptTokens += descResponse.promptEvalCount() != null ? descResponse.promptEvalCount() : 0;
            totalCompletionTokens += descResponse.evalCount() != null ? descResponse.evalCount() : 0;
            logger.info("Step 2 result - Description: {}", description);

            // Assemble final commit message: type: description
            String commitMessage = type + ": " + description;

            long durationMs = System.currentTimeMillis() - startTime;
            logger.info("Multi-step generation complete in {}ms: {}", durationMs, commitMessage);

            return GenerationResult.success(
                commitMessage,
                durationMs,
                totalPromptTokens,
                totalCompletionTokens,
                "ollama-multistep:" + config.model()
            );

        } catch (OllamaException e) {
            logger.error("Multi-step generation failed", e);
            throw new GenerationException("Multi-step generation failed: " + e.getMessage(), e);
        }
    }

    private GenerateResponse query(String prompt, String model, int maxTokens) throws OllamaException {
        GenerateRequest request = new GenerateRequest(
            model,
            prompt,
            false,
            new GenerateRequest.Options(0.3, maxTokens)  // Low temperature for consistency
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
        // Extract just the type word
        String[] validTypes = {"feat", "fix", "docs", "refactor", "test", "build", "ci", "chore"};
        for (String type : validTypes) {
            if (cleaned.contains(type)) {
                return type;
            }
        }
        // Default to chore if we can't determine
        logger.warn("Could not determine type from response: '{}', defaulting to 'chore'", response);
        return "chore";
    }

    private String cleanScope(String response) {
        String cleaned = response.trim().toLowerCase();
        // Remove common noise
        cleaned = cleaned.replaceAll("[^a-z0-9-]", "");

        if (cleaned.isEmpty() || cleaned.equals("none") || cleaned.equals("na") || cleaned.length() > 15) {
            return "";
        }
        return cleaned;
    }

    private String cleanDescription(String response) {
        String cleaned = response.trim();
        // Remove quotes if present
        cleaned = cleaned.replaceAll("^\"|\"$", "");
        cleaned = cleaned.replaceAll("^'|'$", "");
        // Remove any type prefix the model might have added (case-insensitive)
        cleaned = cleaned.replaceAll("(?i)^(feat|fix|docs|refactor|test|build|ci|chore)(\\([^)]*\\))?:\\s*", "");
        // Also remove standalone type words at the start (e.g., "Refactor the..." -> "the...")
        cleaned = cleaned.replaceAll("(?i)^(feat|fix|docs|refactor|test|build|ci|chore)\\s+", "");
        // Ensure lowercase first letter (conventional commit style)
        if (!cleaned.isEmpty()) {
            cleaned = Character.toLowerCase(cleaned.charAt(0)) + cleaned.substring(1);
        }
        // Remove trailing period
        cleaned = cleaned.replaceAll("\\.$", "");
        // Truncate if too long
        if (cleaned.length() > 72) {
            cleaned = cleaned.substring(0, 69) + "...";
        }
        return cleaned;
    }

    @Override
    public boolean isAvailable() {
        return client.isAvailable();
    }

    @Override
    public String getBackendName() {
        return "ollama-multistep";
    }
}
