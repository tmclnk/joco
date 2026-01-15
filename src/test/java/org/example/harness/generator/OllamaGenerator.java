package org.example.harness.generator;

import org.example.ollama.GenerateRequest;
import org.example.ollama.GenerateResponse;
import org.example.ollama.OllamaClient;
import org.example.ollama.OllamaException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CommitGenerator implementation that wraps OllamaClient.
 */
public class OllamaGenerator implements CommitGenerator {

    private static final Logger logger = LoggerFactory.getLogger(OllamaGenerator.class);

    private final OllamaClient client;

    /**
     * Creates a new OllamaGenerator with default settings.
     */
    public OllamaGenerator() {
        this(new OllamaClient());
    }

    /**
     * Creates a new OllamaGenerator wrapping an existing OllamaClient.
     *
     * @param client The Ollama client to use
     */
    public OllamaGenerator(OllamaClient client) {
        this.client = client;
    }

    @Override
    public GenerationResult generate(String prompt, GenerationConfig config) throws GenerationException {
        logger.debug("Generating commit message with Ollama model: {}", config.model());

        long startTime = System.currentTimeMillis();

        try {
            GenerateRequest request = new GenerateRequest(
                config.model(),
                prompt,
                false,
                new GenerateRequest.Options(config.temperature(), config.maxTokens())
            );

            GenerateResponse response = client.generate(request);

            long durationMs = System.currentTimeMillis() - startTime;

            return GenerationResult.success(
                response.response(),
                durationMs,
                response.promptEvalCount() != null ? response.promptEvalCount() : 0,
                response.evalCount() != null ? response.evalCount() : 0,
                "ollama:" + config.model()
            );

        } catch (OllamaException e) {
            logger.error("Ollama generation failed", e);
            throw new GenerationException("Ollama generation failed: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean isAvailable() {
        return client.isAvailable();
    }

    @Override
    public String getBackendName() {
        return "ollama";
    }

    /**
     * Returns the underlying OllamaClient.
     *
     * @return The Ollama client
     */
    public OllamaClient getClient() {
        return client;
    }
}
