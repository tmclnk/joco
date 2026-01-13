package org.example.config;

import java.util.Objects;

/**
 * Configuration domain model representing the application settings.
 * Contains model selection, token limits, and temperature settings for LLM interactions.
 */
public class Config {
    private static final String DEFAULT_MODEL = "qwen2.5-coder:1.5b";
    private static final int DEFAULT_MAX_TOKENS = 100;
    private static final double DEFAULT_TEMPERATURE = 0.7;

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
     *
     * @param model the model to validate
     * @throws IllegalArgumentException if model is null or empty
     */
    private void validateModel(String model) {
        if (model == null || model.trim().isEmpty()) {
            throw new IllegalArgumentException("Model cannot be null or empty");
        }
    }

    /**
     * Validates the maxTokens parameter.
     *
     * @param maxTokens the maxTokens to validate
     * @throws IllegalArgumentException if maxTokens is not positive
     */
    private void validateMaxTokens(int maxTokens) {
        if (maxTokens <= 0) {
            throw new IllegalArgumentException("Max tokens must be positive, got: " + maxTokens);
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
