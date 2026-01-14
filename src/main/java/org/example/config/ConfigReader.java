package org.example.config;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Reads configuration from ~/.joco.config file.
 * Handles missing files gracefully by returning default configuration.
 * Uses Gson for JSON parsing.
 */
public class ConfigReader {
    private static final Logger logger = LoggerFactory.getLogger(ConfigReader.class);
    private static final String CONFIG_FILE_NAME = ".joco.config";

    private final Gson gson;
    private final Path configPath;

    /**
     * Creates a ConfigReader with default configuration file location (~/.joco.config).
     */
    public ConfigReader() {
        this(getDefaultConfigPath());
    }

    /**
     * Creates a ConfigReader with a custom configuration file path.
     * Used primarily for testing.
     *
     * @param configPath the path to the configuration file
     */
    public ConfigReader(Path configPath) {
        this.gson = new Gson();
        this.configPath = configPath;
    }

    /**
     * Reads the configuration from the config file.
     * If the file doesn't exist or cannot be read, returns default configuration.
     *
     * @return the parsed Config object, or default Config if file is missing or invalid
     */
    public Config read() {
        // If config file doesn't exist, return default config
        if (!Files.exists(configPath)) {
            logger.debug("Configuration file not found at {}, using defaults", configPath);
            return new Config();
        }

        // Try to read and parse the config file
        try {
            String content = Files.readString(configPath);
            ConfigDto dto = gson.fromJson(content, ConfigDto.class);

            // If parsing resulted in null or empty content, use defaults
            if (dto == null) {
                logger.warn("Configuration file at {} is empty or invalid, using defaults", configPath);
                return new Config();
            }

            return createConfigFromDto(dto);
        } catch (IOException e) {
            logger.warn("Failed to read configuration file at {}: {}, using defaults",
                configPath, e.getMessage());
            return new Config();
        } catch (JsonSyntaxException e) {
            logger.warn("Invalid JSON in configuration file at {}: {}, using defaults",
                configPath, e.getMessage());
            return new Config();
        } catch (IllegalArgumentException e) {
            logger.warn("Invalid configuration values in {}: {}, using defaults",
                configPath, e.getMessage());
            return new Config();
        }
    }

    /**
     * Creates a Config object from a ConfigDto, using defaults for missing or invalid values.
     * Logs warnings for each invalid field while using defaults.
     *
     * @param dto the data transfer object from JSON parsing
     * @return a Config object with values from dto or defaults
     */
    private Config createConfigFromDto(ConfigDto dto) {
        // Create default config to get default values
        Config defaults = new Config();

        // Validate and use model
        String model = defaults.getModel();
        if (dto.model != null && !dto.model.trim().isEmpty()) {
            try {
                // Try to create a temp config to validate the model
                new Config(dto.model, defaults.getMaxTokens(), defaults.getTemperature());
                model = dto.model;
            } catch (IllegalArgumentException e) {
                logger.warn("Invalid model value '{}': {}. Using default: {}",
                    dto.model, e.getMessage(), defaults.getModel());
            }
        } else if (dto.model != null) {
            logger.warn("Empty model value provided. Using default: {}", defaults.getModel());
        }

        // Validate and use maxTokens
        int maxTokens = defaults.getMaxTokens();
        if (dto.maxTokens != 0) {
            try {
                // Try to create a temp config to validate maxTokens
                new Config(defaults.getModel(), dto.maxTokens, defaults.getTemperature());
                maxTokens = dto.maxTokens;
            } catch (IllegalArgumentException e) {
                logger.warn("Invalid maxTokens value '{}': {}. Using default: {}",
                    dto.maxTokens, e.getMessage(), defaults.getMaxTokens());
            }
        }

        // Validate and use temperature
        double temperature = defaults.getTemperature();
        if (dto.temperature != 0.0) {
            try {
                // Try to create a temp config to validate temperature
                new Config(defaults.getModel(), defaults.getMaxTokens(), dto.temperature);
                temperature = dto.temperature;
            } catch (IllegalArgumentException e) {
                logger.warn("Invalid temperature value '{}': {}. Using default: {}",
                    dto.temperature, e.getMessage(), defaults.getTemperature());
            }
        }

        // Create final config with validated values
        // At this point all values are valid, so this won't throw
        return new Config(model, maxTokens, temperature);
    }

    /**
     * Gets the default configuration file path (~/.joco.config).
     *
     * @return the path to the default configuration file
     */
    private static Path getDefaultConfigPath() {
        String userHome = System.getProperty("user.home");
        return Paths.get(userHome, CONFIG_FILE_NAME);
    }

    /**
     * Data transfer object for JSON deserialization.
     * Package-private for testing.
     */
    static class ConfigDto {
        String model;
        int maxTokens;
        double temperature;
    }
}
