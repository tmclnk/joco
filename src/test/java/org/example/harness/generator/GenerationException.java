package org.example.harness.generator;

/**
 * Exception thrown when commit message generation fails.
 */
public class GenerationException extends Exception {

    public GenerationException(String message) {
        super(message);
    }

    public GenerationException(String message, Throwable cause) {
        super(message, cause);
    }
}
