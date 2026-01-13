package org.example.git;

/**
 * Base exception for all Git-related errors.
 */
public class GitException extends Exception {

    public GitException(String message) {
        super(message);
    }

    public GitException(String message, Throwable cause) {
        super(message, cause);
    }
}
