package org.example.ollama;

import com.google.gson.annotations.SerializedName;

/**
 * Response model for Ollama's /api/generate endpoint.
 */
public record GenerateResponse(
    String model,
    @SerializedName("created_at") String createdAt,
    String response,
    boolean done,
    @SerializedName("total_duration") Long totalDuration,
    @SerializedName("load_duration") Long loadDuration,
    @SerializedName("prompt_eval_count") Integer promptEvalCount,
    @SerializedName("prompt_eval_duration") Long promptEvalDuration,
    @SerializedName("eval_count") Integer evalCount,
    @SerializedName("eval_duration") Long evalDuration
) {
}
