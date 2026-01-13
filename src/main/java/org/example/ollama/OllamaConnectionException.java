package org.example.ollama;

/**
 * Exception thrown when unable to connect to the Ollama service.
 * This typically indicates that Ollama is not running or is unreachable.
 */
public class OllamaConnectionException extends OllamaException {

    public OllamaConnectionException(String message) {
        super(message);
    }

    public OllamaConnectionException(String message, Throwable cause) {
        super(message, cause);
    }
}
