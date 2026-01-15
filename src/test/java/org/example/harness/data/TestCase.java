package org.example.harness.data;

import java.util.Map;

/**
 * Represents a test case for commit message generation.
 * Contains a git diff and the expected commit message from a gold-standard repository.
 */
public record TestCase(
    String id,
    String diff,
    String expectedMessage,
    String repository,
    String commitHash,
    Map<String, String> metadata
) {
    public TestCase {
        if (id == null || id.isBlank()) {
            throw new IllegalArgumentException("id cannot be null or blank");
        }
        if (diff == null || diff.isBlank()) {
            throw new IllegalArgumentException("diff cannot be null or blank");
        }
        if (expectedMessage == null || expectedMessage.isBlank()) {
            throw new IllegalArgumentException("expectedMessage cannot be null or blank");
        }
        if (metadata == null) {
            metadata = Map.of();
        }
    }

    /**
     * Creates a TestCase with minimal required fields.
     */
    public static TestCase of(String id, String diff, String expectedMessage) {
        return new TestCase(id, diff, expectedMessage, null, null, Map.of());
    }
}
