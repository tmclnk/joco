package org.example.config;

import java.util.Objects;

/**
 * Configuration domain model representing the application settings.
 * Contains model selection, token limits, and temperature settings for LLM interactions.
 */
public class Config {
    private static final String DEFAULT_MODEL = "qwen2.5-coder:1.5b";
    private static final int DEFAULT_MAX_TOKENS = 100;
    private static final double DEFAULT_TEMPERATURE = 0.3;

    private final String model;
    private final int maxTokens;
    private final double temperature;

    /**
     * Creates a Config with default values.
     */
    public Config() {
        this(DEFAULT_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE);
    }

    /**
     * Creates a Config with specified values.
     *
     * @param model the LLM model identifier
     * @param maxTokens the maximum number of tokens
     * @param temperature the sampling temperature
     * @throws IllegalArgumentException if validation fails
     */
    public Config(String model, int maxTokens, double temperature) {
        validateModel(model);
        validateMaxTokens(maxTokens);
        validateTemperature(temperature);

        this.model = model;
        this.maxTokens = maxTokens;
        this.temperature = temperature;
    }

    /**
     * Validates the model parameter.
     * Model must follow the Ollama format: name or name:tag
     * where name contains alphanumeric characters, dots, hyphens, or underscores
     * and optional tag follows a colon.
     *
     * @param model the model to validate
     * @throws IllegalArgumentException if model is null, empty, or has invalid format
     */
    private void validateModel(String model) {
        if (model == null || model.trim().isEmpty()) {
            throw new IllegalArgumentException("Model cannot be null or empty");
        }

        // Validate model format: name or name:tag
        // Name can contain alphanumeric chars, dots, hyphens, underscores
        // Tag (optional) can contain alphanumeric chars, dots, hyphens, underscores
        String modelPattern = "^[a-zA-Z0-9._-]+(:?[a-zA-Z0-9._-]+)?$";
        if (!model.matches(modelPattern)) {
            throw new IllegalArgumentException(
                "Model name must follow Ollama format (e.g., 'qwen2.5-coder' or 'qwen2.5-coder:1.5b'). " +
                "Only alphanumeric characters, dots, hyphens, and underscores are allowed, got: " + model);
        }
    }

    /**
     * Validates the maxTokens parameter.
     *
     * @param maxTokens the maxTokens to validate
     * @throws IllegalArgumentException if maxTokens is not in the valid range [50, 500]
     */
    private void validateMaxTokens(int maxTokens) {
        if (maxTokens < 50 || maxTokens > 500) {
            throw new IllegalArgumentException(
                "Max tokens must be between 50 and 500, got: " + maxTokens);
        }
    }

    /**
     * Validates the temperature parameter.
     *
     * @param temperature the temperature to validate
     * @throws IllegalArgumentException if temperature is not in valid range [0.0, 2.0]
     */
    private void validateTemperature(double temperature) {
        if (temperature < 0.0 || temperature > 2.0) {
            throw new IllegalArgumentException(
                "Temperature must be between 0.0 and 2.0, got: " + temperature);
        }
    }

    public String getModel() {
        return model;
    }

    public int getMaxTokens() {
        return maxTokens;
    }

    public double getTemperature() {
        return temperature;
    }

    /**
     * Creates a new Config with the specified model, keeping other settings.
     *
     * @param newModel the new model identifier
     * @return a new Config instance with the updated model
     * @throws IllegalArgumentException if the new model is invalid
     */
    public Config withModel(String newModel) {
        return new Config(newModel, this.maxTokens, this.temperature);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Config config = (Config) o;
        return maxTokens == config.maxTokens &&
               Double.compare(config.temperature, temperature) == 0 &&
               Objects.equals(model, config.model);
    }

    @Override
    public int hashCode() {
        return Objects.hash(model, maxTokens, temperature);
    }

    @Override
    public String toString() {
        return "Config{" +
               "model='" + model + '\'' +
               ", maxTokens=" + maxTokens +
               ", temperature=" + temperature +
               '}';
    }
}
