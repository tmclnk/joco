package org.example.ollama;

import com.google.gson.annotations.SerializedName;

/**
 * Request model for Ollama's /api/generate endpoint.
 */
public record GenerateRequest(
    String model,
    String prompt,
    @SerializedName("stream") boolean stream,
    Options options
) {
    public record Options(
        Double temperature,
        @SerializedName("num_predict") Integer numPredict
    ) {
        public Options {
            if (temperature != null && (temperature < 0.0 || temperature > 2.0)) {
                throw new IllegalArgumentException("Temperature must be between 0.0 and 2.0");
            }
        }
    }

    public GenerateRequest {
        if (model == null || model.isBlank()) {
            throw new IllegalArgumentException("Model cannot be null or blank");
        }
        if (prompt == null || prompt.isBlank()) {
            throw new IllegalArgumentException("Prompt cannot be null or blank");
        }
    }
}
