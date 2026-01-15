package org.example.harness.evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Validates commit messages against conventional commit standards.
 */
public class StructuralValidator {

    private static final Pattern CONVENTIONAL_COMMIT = Pattern.compile(
        "^(feat|fix|chore|docs|refactor|test|style|perf|build|ci)(\\(([\\w.-]+)\\))?: (.+)$"
    );

    private static final int MAX_SUBJECT_LENGTH = 72;
    private static final int RECOMMENDED_SUBJECT_LENGTH = 50;

    /**
     * Result of validating a commit message.
     */
    public record ValidationResult(
        boolean hasValidType,
        String type,
        boolean hasScope,
        String scope,
        boolean hasDescription,
        String description,
        int subjectLength,
        boolean subjectWithinLimit,
        boolean subjectWithinRecommended,
        boolean startsWithLowercase,
        boolean endsWithPeriod,
        List<String> issues
    ) {
        /**
         * Returns true if the message is valid conventional commit.
         */
        public boolean isValid() {
            return hasValidType && hasDescription && subjectWithinLimit && !endsWithPeriod;
        }

        /**
         * Returns a quality score from 0-100.
         */
        public int score() {
            int score = 0;
            if (hasValidType) score += 30;
            if (hasDescription) score += 20;
            if (hasScope) score += 10;
            if (subjectWithinLimit) score += 15;
            if (subjectWithinRecommended) score += 10;
            if (startsWithLowercase) score += 10;
            if (!endsWithPeriod) score += 5;
            return score;
        }
    }

    /**
     * Validates a commit message against conventional commit standards.
     */
    public ValidationResult validate(String message) {
        List<String> issues = new ArrayList<>();

        if (message == null || message.isBlank()) {
            return new ValidationResult(
                false, null, false, null, false, null,
                0, false, false, false, false,
                List.of("Message is empty")
            );
        }

        // Get first line only
        String subject = message.lines().findFirst().orElse(message).trim();

        Matcher matcher = CONVENTIONAL_COMMIT.matcher(subject);
        boolean hasValidType = matcher.matches();

        String type = null;
        String scope = null;
        String description = null;

        if (hasValidType) {
            type = matcher.group(1);
            scope = matcher.group(3); // Group 3 is inside the parens
            description = matcher.group(4);
        } else {
            issues.add("Does not match conventional commit format: type(scope): description");
        }

        boolean hasScope = scope != null && !scope.isEmpty();
        boolean hasDescription = description != null && !description.isEmpty();
        int subjectLength = subject.length();
        boolean subjectWithinLimit = subjectLength <= MAX_SUBJECT_LENGTH;
        boolean subjectWithinRecommended = subjectLength <= RECOMMENDED_SUBJECT_LENGTH;

        boolean startsWithLowercase = description != null &&
            !description.isEmpty() &&
            Character.isLowerCase(description.charAt(0));

        boolean endsWithPeriod = subject.endsWith(".");

        if (!subjectWithinLimit) {
            issues.add("Subject exceeds " + MAX_SUBJECT_LENGTH + " characters (" + subjectLength + ")");
        } else if (!subjectWithinRecommended) {
            issues.add("Subject exceeds recommended " + RECOMMENDED_SUBJECT_LENGTH + " characters");
        }

        if (endsWithPeriod) {
            issues.add("Subject should not end with period");
        }

        if (!startsWithLowercase && description != null && !description.isEmpty()) {
            issues.add("Description should start with lowercase");
        }

        return new ValidationResult(
            hasValidType, type, hasScope, scope, hasDescription, description,
            subjectLength, subjectWithinLimit, subjectWithinRecommended,
            startsWithLowercase, endsWithPeriod, issues
        );
    }
}
