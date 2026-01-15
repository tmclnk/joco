package org.example.harness.evaluation;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Represents the decomposed parts of a conventional commit message.
 *
 * A conventional commit has the format: type(scope): description
 * where scope is optional.
 */
public record CommitComponents(
    String type,
    String scope,
    String description,
    boolean valid
) {

    private static final Pattern CONVENTIONAL_COMMIT = Pattern.compile(
        "^(feat|fix|chore|docs|refactor|test|style|perf|build|ci)(\\(([\\w.-]+)\\))?: (.+)$"
    );

    /**
     * Parses a commit message into its component parts.
     *
     * @param message The commit message to parse
     * @return CommitComponents with parsed values, or invalid components if parsing fails
     */
    public static CommitComponents parse(String message) {
        if (message == null || message.isBlank()) {
            return invalid();
        }

        // Get first line only (subject)
        String subject = message.lines().findFirst().orElse(message).trim();

        Matcher matcher = CONVENTIONAL_COMMIT.matcher(subject);
        if (!matcher.matches()) {
            return invalid();
        }

        String type = matcher.group(1);
        String scope = matcher.group(3); // Group 3 is inside the parentheses
        String description = matcher.group(4);

        return new CommitComponents(type, scope, description, true);
    }

    /**
     * Returns an invalid CommitComponents instance.
     */
    public static CommitComponents invalid() {
        return new CommitComponents(null, null, null, false);
    }

    /**
     * Returns true if this commit has a scope.
     */
    public boolean hasScope() {
        return scope != null && !scope.isEmpty();
    }

    /**
     * Returns true if this commit has a description.
     */
    public boolean hasDescription() {
        return description != null && !description.isEmpty();
    }
}
