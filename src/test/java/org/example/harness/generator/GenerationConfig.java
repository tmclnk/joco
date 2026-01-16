package org.example.harness.generator;

/**
 * Configuration for commit message generation.
 */
public record GenerationConfig(
    String model,
    double temperature,
    int maxTokens
) {
    public GenerationConfig {
        if (model == null || model.isBlank()) {
            model = "default";
        }
        if (temperature < 0.0 || temperature > 2.0) {
            throw new IllegalArgumentException("Temperature must be between 0.0 and 2.0");
        }
        if (maxTokens < 10 || maxTokens > 4096) {
            throw new IllegalArgumentException("maxTokens must be between 10 and 4096");
        }
    }

    /**
     * Creates a default configuration.
     */
    public static GenerationConfig defaults() {
        return new GenerationConfig("default", 0.7, 100);
    }

    /**
     * Creates a config with a specific model.
     */
    public GenerationConfig withModel(String newModel) {
        return new GenerationConfig(newModel, temperature, maxTokens);
    }

    /**
     * Creates a config with a specific temperature.
     */
    public GenerationConfig withTemperature(double newTemperature) {
        return new GenerationConfig(model, newTemperature, maxTokens);
    }

    /**
     * Creates a config with specific max tokens.
     */
    public GenerationConfig withMaxTokens(int newMaxTokens) {
        return new GenerationConfig(model, temperature, newMaxTokens);
    }
}
