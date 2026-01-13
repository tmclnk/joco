package org.example.ollama;

/**
 * Exception thrown when the Ollama API returns an error response.
 * This includes HTTP error status codes and invalid JSON responses.
 */
public class OllamaApiException extends OllamaException {

    public OllamaApiException(String message) {
        super(message);
    }

    public OllamaApiException(String message, Throwable cause) {
        super(message, cause);
    }
}
