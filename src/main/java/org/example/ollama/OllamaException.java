package org.example.ollama;

/**
 * Base exception for all Ollama-related errors.
 */
public class OllamaException extends Exception {

    public OllamaException(String message) {
        super(message);
    }

    public OllamaException(String message, Throwable cause) {
        super(message, cause);
    }
}
