package org.example.ollama;

import com.google.gson.Gson;
import com.google.gson.JsonSyntaxException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.StringReader;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.function.Consumer;

/**
 * HTTP client for interacting with the Ollama API at localhost:11434.
 * Provides methods to generate text using local LLM models via Ollama.
 */
public class OllamaClient {
    private static final Logger logger = LoggerFactory.getLogger(OllamaClient.class);

    private static final String DEFAULT_BASE_URL = "http://localhost:11434";
    private static final String GENERATE_ENDPOINT = "/api/generate";
    private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(5);
    private static final Duration REQUEST_TIMEOUT = Duration.ofMinutes(2);

    private final String baseUrl;
    private final HttpClient httpClient;
    private final Gson gson;

    /**
     * Creates a new OllamaClient with default settings (localhost:11434).
     */
    public OllamaClient() {
        this(DEFAULT_BASE_URL);
    }

    /**
     * Creates a new OllamaClient with a custom base URL.
     *
     * @param baseUrl The base URL for the Ollama API (e.g., "http://localhost:11434")
     */
    public OllamaClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(DEFAULT_TIMEOUT)
            .build();
        this.gson = new Gson();
    }

    /**
     * Generates text using the Ollama API.
     *
     * @param request The generation request containing model, prompt, and options
     * @return The generation response with the generated text
     * @throws OllamaConnectionException if Ollama is not running or unreachable
     * @throws OllamaApiException if the API returns an error
     * @throws OllamaException for other errors during the request
     */
    public GenerateResponse generate(GenerateRequest request) throws OllamaException {
        logger.debug("Generating text with model: {}", request.model());

        String requestBody = gson.toJson(request);
        URI uri = URI.create(baseUrl + GENERATE_ENDPOINT);

        HttpRequest httpRequest = HttpRequest.newBuilder()
            .uri(uri)
            .header("Content-Type", "application/json")
            .timeout(REQUEST_TIMEOUT)
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        try {
            HttpResponse<String> httpResponse = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            if (httpResponse.statusCode() >= 200 && httpResponse.statusCode() < 300) {
                try {
                    GenerateResponse response = gson.fromJson(httpResponse.body(), GenerateResponse.class);
                    logger.debug("Successfully generated text ({} chars)",
                        response.response() != null ? response.response().length() : 0);
                    return response;
                } catch (JsonSyntaxException e) {
                    logger.error("Failed to parse Ollama API response", e);
                    throw new OllamaApiException("Invalid JSON response from Ollama API", e);
                }
            } else {
                logger.error("Ollama API returned error status: {}", httpResponse.statusCode());
                throw new OllamaApiException(
                    "Ollama API returned error status " + httpResponse.statusCode() + ": " + httpResponse.body()
                );
            }

        } catch (ConnectException e) {
            logger.error("Failed to connect to Ollama at {}", baseUrl, e);
            throw new OllamaConnectionException(
                "Cannot connect to Ollama at " + baseUrl + ". Is Ollama running? " +
                "Start it with: ollama serve",
                e
            );
        } catch (IOException e) {
            logger.error("I/O error while communicating with Ollama", e);
            throw new OllamaException("I/O error while communicating with Ollama: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Request to Ollama was interrupted", e);
            throw new OllamaException("Request to Ollama was interrupted", e);
        }
    }

    /**
     * Generates text using the Ollama API with streaming enabled.
     * The response is delivered incrementally via a callback as JSON chunks arrive.
     *
     * @param request The generation request containing model, prompt, and options
     * @param chunkCallback Consumer that receives each GenerateResponse chunk as it arrives
     * @return The final accumulated GenerateResponse
     * @throws OllamaConnectionException if Ollama is not running or unreachable
     * @throws OllamaApiException if the API returns an error
     * @throws OllamaException for other errors during the request
     */
    public GenerateResponse generateStreaming(GenerateRequest request, Consumer<GenerateResponse> chunkCallback) throws OllamaException {
        logger.debug("Generating streaming text with model: {}", request.model());

        // Create a new request with streaming enabled
        GenerateRequest streamingRequest = new GenerateRequest(
            request.model(),
            request.prompt(),
            true, // Enable streaming
            request.options()
        );

        String requestBody = gson.toJson(streamingRequest);
        URI uri = URI.create(baseUrl + GENERATE_ENDPOINT);

        HttpRequest httpRequest = HttpRequest.newBuilder()
            .uri(uri)
            .header("Content-Type", "application/json")
            .timeout(REQUEST_TIMEOUT)
            .POST(HttpRequest.BodyPublishers.ofString(requestBody))
            .build();

        try {
            HttpResponse<String> httpResponse = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());

            if (httpResponse.statusCode() >= 200 && httpResponse.statusCode() < 300) {
                return parseStreamingResponse(httpResponse.body(), chunkCallback);
            } else {
                logger.error("Ollama API returned error status: {}", httpResponse.statusCode());
                throw new OllamaApiException(
                    "Ollama API returned error status " + httpResponse.statusCode() + ": " + httpResponse.body()
                );
            }

        } catch (ConnectException e) {
            logger.error("Failed to connect to Ollama at {}", baseUrl, e);
            throw new OllamaConnectionException(
                "Cannot connect to Ollama at " + baseUrl + ". Is Ollama running? " +
                "Start it with: ollama serve",
                e
            );
        } catch (IOException e) {
            logger.error("I/O error while communicating with Ollama", e);
            throw new OllamaException("I/O error while communicating with Ollama: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Request to Ollama was interrupted", e);
            throw new OllamaException("Request to Ollama was interrupted", e);
        }
    }

    /**
     * Parses streaming JSON response from Ollama, where each line is a separate JSON object.
     * Accumulates the response text from each chunk and invokes the callback for each chunk.
     *
     * @param responseBody The full response body containing newline-separated JSON objects
     * @param chunkCallback Consumer that receives each GenerateResponse chunk
     * @return The final accumulated GenerateResponse with complete text
     * @throws OllamaApiException if JSON parsing fails
     */
    private GenerateResponse parseStreamingResponse(String responseBody, Consumer<GenerateResponse> chunkCallback) throws OllamaApiException {
        StringBuilder accumulatedResponse = new StringBuilder();
        GenerateResponse lastChunk = null;

        try (BufferedReader reader = new BufferedReader(new StringReader(responseBody))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.trim().isEmpty()) {
                    continue;
                }

                try {
                    GenerateResponse chunk = gson.fromJson(line, GenerateResponse.class);
                    lastChunk = chunk;

                    // Accumulate the response text
                    if (chunk.response() != null) {
                        accumulatedResponse.append(chunk.response());
                    }

                    // Notify the callback with the current chunk
                    if (chunkCallback != null) {
                        chunkCallback.accept(chunk);
                    }

                    logger.debug("Received chunk: done={}, response_length={}",
                        chunk.done(),
                        chunk.response() != null ? chunk.response().length() : 0);

                } catch (JsonSyntaxException e) {
                    logger.error("Failed to parse JSON chunk: {}", line, e);
                    throw new OllamaApiException("Invalid JSON in streaming response: " + line, e);
                }
            }

            if (lastChunk == null) {
                throw new OllamaApiException("No valid JSON chunks received in streaming response");
            }

            // Return the final response with accumulated text
            String fullResponse = accumulatedResponse.toString();
            logger.debug("Streaming complete. Total response length: {}", fullResponse.length());

            // Create a final response with the accumulated text and metadata from the last chunk
            return new GenerateResponse(
                lastChunk.model(),
                lastChunk.createdAt(),
                fullResponse,
                lastChunk.done(),
                lastChunk.totalDuration(),
                lastChunk.loadDuration(),
                lastChunk.promptEvalCount(),
                lastChunk.promptEvalDuration(),
                lastChunk.evalCount(),
                lastChunk.evalDuration()
            );

        } catch (IOException e) {
            logger.error("I/O error while parsing streaming response", e);
            throw new OllamaApiException("I/O error while parsing streaming response", e);
        }
    }

    /**
     * Checks if Ollama is running and reachable.
     *
     * @return true if Ollama is reachable, false otherwise
     */
    public boolean isAvailable() {
        try {
            URI uri = URI.create(baseUrl + "/api/tags");
            HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(uri)
                .timeout(DEFAULT_TIMEOUT)
                .GET()
                .build();

            HttpResponse<String> response = httpClient.send(httpRequest, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;

        } catch (Exception e) {
            logger.debug("Ollama not available at {}: {}", baseUrl, e.getMessage());
            return false;
        }
    }

    /**
     * Gets the base URL of this client.
     *
     * @return The base URL
     */
    public String getBaseUrl() {
        return baseUrl;
    }
}
