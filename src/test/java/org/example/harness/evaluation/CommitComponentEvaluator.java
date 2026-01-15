package org.example.harness.evaluation;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Evaluates and compares the components of conventional commit messages.
 *
 * Provides separate evaluation for:
 * - Type accuracy (exact match)
 * - Scope presence and accuracy
 * - Description similarity (using multiple metrics)
 */
public class CommitComponentEvaluator {

    /**
     * Compares expected and generated commit messages at the component level.
     *
     * @param expectedMessage The expected (gold standard) commit message
     * @param generatedMessage The generated commit message
     * @return ComponentComparisonResult with detailed comparison data
     */
    public ComponentComparisonResult compare(String expectedMessage, String generatedMessage) {
        CommitComponents expected = CommitComponents.parse(expectedMessage);
        CommitComponents generated = CommitComponents.parse(generatedMessage);

        boolean typeMatches = compareTypes(expected, generated);
        boolean scopeMatches = compareScopes(expected, generated);
        boolean scopePresenceMatches = compareScopePresence(expected, generated);
        double descriptionSimilarity = compareDescriptions(expected, generated);

        return new ComponentComparisonResult(
            expected,
            generated,
            typeMatches,
            scopeMatches,
            scopePresenceMatches,
            descriptionSimilarity
        );
    }

    /**
     * Compares types for exact match.
     */
    private boolean compareTypes(CommitComponents expected, CommitComponents generated) {
        if (!expected.valid() || !generated.valid()) {
            return false;
        }
        return expected.type() != null && expected.type().equals(generated.type());
    }

    /**
     * Compares scopes for exact match.
     */
    private boolean compareScopes(CommitComponents expected, CommitComponents generated) {
        if (!expected.valid() || !generated.valid()) {
            return false;
        }
        // Both null or empty is a match
        if (!expected.hasScope() && !generated.hasScope()) {
            return true;
        }
        // One has scope, other doesn't
        if (expected.hasScope() != generated.hasScope()) {
            return false;
        }
        // Both have scopes - compare them
        return expected.scope().equalsIgnoreCase(generated.scope());
    }

    /**
     * Compares whether both messages have the same scope presence (both have or both don't).
     */
    private boolean compareScopePresence(CommitComponents expected, CommitComponents generated) {
        if (!expected.valid() || !generated.valid()) {
            return false;
        }
        return expected.hasScope() == generated.hasScope();
    }

    /**
     * Computes description similarity using multiple metrics.
     * Returns a score from 0.0 to 1.0.
     */
    private double compareDescriptions(CommitComponents expected, CommitComponents generated) {
        if (!expected.valid() || !generated.valid()) {
            return 0.0;
        }
        if (!expected.hasDescription() || !generated.hasDescription()) {
            return 0.0;
        }

        String expDesc = expected.description().toLowerCase();
        String genDesc = generated.description().toLowerCase();

        // Calculate multiple similarity metrics and combine them
        double wordOverlap = calculateWordOverlap(expDesc, genDesc);
        double lengthRatio = calculateLengthRatio(expDesc, genDesc);
        double levenshteinSimilarity = calculateLevenshteinSimilarity(expDesc, genDesc);

        // Weighted average: word overlap is most important, then Levenshtein, then length
        return (wordOverlap * 0.5) + (levenshteinSimilarity * 0.35) + (lengthRatio * 0.15);
    }

    /**
     * Calculates word overlap using Jaccard similarity.
     * Returns a score from 0.0 to 1.0.
     */
    private double calculateWordOverlap(String s1, String s2) {
        Set<String> words1 = tokenize(s1);
        Set<String> words2 = tokenize(s2);

        if (words1.isEmpty() && words2.isEmpty()) {
            return 1.0;
        }
        if (words1.isEmpty() || words2.isEmpty()) {
            return 0.0;
        }

        Set<String> intersection = new HashSet<>(words1);
        intersection.retainAll(words2);

        Set<String> union = new HashSet<>(words1);
        union.addAll(words2);

        return (double) intersection.size() / union.size();
    }

    /**
     * Tokenizes a string into a set of words, filtering out stop words.
     */
    private Set<String> tokenize(String s) {
        // Simple tokenization: split on non-word characters
        return Arrays.stream(s.split("\\W+"))
            .map(String::toLowerCase)
            .filter(w -> w.length() > 2) // Filter very short words
            .filter(w -> !isStopWord(w))
            .collect(Collectors.toSet());
    }

    /**
     * Returns true if the word is a common stop word.
     */
    private boolean isStopWord(String word) {
        Set<String> stopWords = Set.of(
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "it", "its", "not"
        );
        return stopWords.contains(word);
    }

    /**
     * Calculates length ratio similarity.
     * Returns 1.0 if lengths are equal, decreasing as they differ.
     */
    private double calculateLengthRatio(String s1, String s2) {
        int len1 = s1.length();
        int len2 = s2.length();

        if (len1 == 0 && len2 == 0) {
            return 1.0;
        }

        int maxLen = Math.max(len1, len2);
        int minLen = Math.min(len1, len2);

        return (double) minLen / maxLen;
    }

    /**
     * Calculates normalized Levenshtein similarity.
     * Returns a score from 0.0 to 1.0 (1.0 means identical).
     */
    private double calculateLevenshteinSimilarity(String s1, String s2) {
        int distance = levenshteinDistance(s1, s2);
        int maxLen = Math.max(s1.length(), s2.length());

        if (maxLen == 0) {
            return 1.0;
        }

        return 1.0 - ((double) distance / maxLen);
    }

    /**
     * Computes Levenshtein edit distance between two strings.
     */
    private int levenshteinDistance(String s1, String s2) {
        int m = s1.length();
        int n = s2.length();

        // Optimization: use single array instead of matrix
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];

        // Initialize first row
        for (int j = 0; j <= n; j++) {
            prev[j] = j;
        }

        for (int i = 1; i <= m; i++) {
            curr[0] = i;

            for (int j = 1; j <= n; j++) {
                int cost = (s1.charAt(i - 1) == s2.charAt(j - 1)) ? 0 : 1;
                curr[j] = Math.min(
                    Math.min(curr[j - 1] + 1, prev[j] + 1),
                    prev[j - 1] + cost
                );
            }

            // Swap arrays
            int[] temp = prev;
            prev = curr;
            curr = temp;
        }

        return prev[n];
    }
}
