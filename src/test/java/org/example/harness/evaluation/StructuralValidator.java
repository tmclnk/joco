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
        boolean hasMetaDescription,
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
         * Scoring:
         * - Valid type: 30 pts
         * - Has description: 20 pts
         * - Within 72 char limit: 20 pts
         * - Starts with lowercase: 10 pts
         * - No trailing period: 5 pts
         * - Good length (30-60 chars): 15 pts
         * - Meta-description penalty: -20 pts
         */
        public int score() {
            int score = 0;
            if (hasValidType) score += 30;
            if (hasDescription) score += 20;
            if (subjectWithinLimit) score += 20;
            if (startsWithLowercase) score += 10;
            if (!endsWithPeriod) score += 5;
            // Reward good length (not too short, not too long)
            if (subjectLength >= 30 && subjectLength <= 60) score += 15;
            // Penalize meta-descriptions
            if (hasMetaDescription) score -= 20;
            return Math.max(0, score);
        }
    }

    // Patterns that indicate meta-description (model explaining instead of committing)
    private static final Pattern META_DESCRIPTION = Pattern.compile(
        "(?i)^(here is|here's|this is|this commit|i would|i can|based on|the following|" +
        "a short|short commit|commit description|commit message|summarizes?|captures?)"
    );

    /**
     * Validates a commit message against conventional commit standards.
     */
    public ValidationResult validate(String message) {
        List<String> issues = new ArrayList<>();

        if (message == null || message.isBlank()) {
            return new ValidationResult(
                false, null, false, null, false, null,
                0, false, false, false, false, false,
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

        // Check for meta-descriptions (model explaining instead of generating)
        boolean hasMetaDescription = description != null &&
            META_DESCRIPTION.matcher(description).find();

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

        if (hasMetaDescription) {
            issues.add("Description appears to be meta-text, not an actual commit message");
        }

        return new ValidationResult(
            hasValidType, type, hasScope, scope, hasDescription, description,
            subjectLength, subjectWithinLimit, subjectWithinRecommended,
            startsWithLowercase, endsWithPeriod, hasMetaDescription, issues
        );
    }
}
