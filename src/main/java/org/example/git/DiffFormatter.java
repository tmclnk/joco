package org.example.git;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Formats git diff output for optimal LLM consumption.
 * Provides clear, structured formatting that helps LLMs understand code changes,
 * while managing size constraints to prevent token overflow.
 */
public class DiffFormatter {
    private static final Logger logger = LoggerFactory.getLogger(DiffFormatter.class);

    // Default maximum size for formatted diff (approximately 100k tokens at ~4 chars/token)
    private static final int DEFAULT_MAX_SIZE = 400_000;

    // Size thresholds for warnings
    private static final int WARNING_SIZE = 200_000; // Warn at 50k tokens

    private final int maxSize;

    // Statistics tracking fields
    private int lastTotalFiles;
    private int lastIncludedFiles;
    private int lastSkippedFiles;
    private int lastEstimatedTokens;
    private boolean lastWasTruncated;

    /**
     * Statistics about the last format operation.
     *
     * @param totalFiles total number of files in the diff
     * @param includedFiles number of files included in formatted output
     * @param skippedFiles number of files skipped due to size limits
     * @param estimatedTokens estimated token count of formatted output
     * @param wasTruncated whether any files were skipped
     */
    public record FormatStatistics(
        int totalFiles,
        int includedFiles,
        int skippedFiles,
        int estimatedTokens,
        boolean wasTruncated
    ) {}

    /**
     * Creates a DiffFormatter with default size limits.
     */
    public DiffFormatter() {
        this(DEFAULT_MAX_SIZE);
    }

    /**
     * Creates a DiffFormatter with custom size limit.
     *
     * @param maxSize maximum size in characters for formatted output
     */
    public DiffFormatter(int maxSize) {
        this.maxSize = maxSize;
        logger.debug("DiffFormatter created with max size: {}", maxSize);
    }

    /**
     * Formats a list of file changes into a clear, structured format for LLM consumption.
     * Includes size management to prevent token overflow.
     *
     * @param changes list of file changes to format
     * @return formatted diff string suitable for LLM processing
     */
    public String format(List<DiffParser.FileChange> changes) {
        // Reset statistics
        lastTotalFiles = 0;
        lastIncludedFiles = 0;
        lastSkippedFiles = 0;
        lastEstimatedTokens = 0;
        lastWasTruncated = false;

        if (changes == null || changes.isEmpty()) {
            return "No changes detected.";
        }

        lastTotalFiles = changes.size();

        StringBuilder formatted = new StringBuilder();

        // Add summary header
        formatted.append("=".repeat(70)).append("\n");
        formatted.append("SUMMARY OF CHANGES\n");
        formatted.append("=".repeat(70)).append("\n\n");

        // Count changes by type
        long added = changes.stream().filter(c -> c.getType() == DiffParser.FileChange.ChangeType.ADDED).count();
        long modified = changes.stream().filter(c -> c.getType() == DiffParser.FileChange.ChangeType.MODIFIED).count();
        long deleted = changes.stream().filter(c -> c.getType() == DiffParser.FileChange.ChangeType.DELETED).count();
        long renamed = changes.stream().filter(c -> c.getType() == DiffParser.FileChange.ChangeType.RENAMED).count();
        long binary = changes.stream().filter(DiffParser.FileChange::isBinary).count();

        formatted.append("Total files changed: ").append(changes.size()).append("\n");
        if (added > 0) formatted.append("  - Added: ").append(added).append("\n");
        if (modified > 0) formatted.append("  - Modified: ").append(modified).append("\n");
        if (deleted > 0) formatted.append("  - Deleted: ").append(deleted).append("\n");
        if (renamed > 0) formatted.append("  - Renamed: ").append(renamed).append("\n");
        if (binary > 0) formatted.append("  - Binary files: ").append(binary).append("\n");

        formatted.append("\n");

        // Format each file change
        int totalContentSize = formatted.length();

        for (DiffParser.FileChange change : changes) {
            String fileSection = formatFileChange(change);
            int sectionSize = fileSection.length();

            // Check if adding this file would exceed the limit
            if (totalContentSize + sectionSize > maxSize) {
                lastSkippedFiles++;
                logger.warn("Skipping file {} due to size limit (current: {}, section: {}, max: {})",
                    change.getPath(), totalContentSize, sectionSize, maxSize);
                continue;
            }

            formatted.append(fileSection);
            totalContentSize += sectionSize;
            lastIncludedFiles++;
        }

        // Add truncation warning if needed
        if (lastSkippedFiles > 0) {
            formatted.append("\n");
            formatted.append("=".repeat(70)).append("\n");
            formatted.append("WARNING: ").append(lastSkippedFiles).append(" file(s) skipped due to size limits\n");
            formatted.append("Consider staging fewer files or files with smaller changes.\n");
            formatted.append("=".repeat(70)).append("\n");
        }

        // Log size information and update statistics
        String result = formatted.toString();
        int finalSize = result.length();
        lastEstimatedTokens = finalSize / 4;
        lastWasTruncated = lastSkippedFiles > 0;

        logger.info("Formatted {} files (skipped {}) - Size: {} chars (~{} tokens)",
            lastIncludedFiles, lastSkippedFiles, finalSize, lastEstimatedTokens);

        if (finalSize > WARNING_SIZE) {
            logger.warn("Large diff detected: {} chars (~{} tokens). Consider reducing scope.",
                finalSize, lastEstimatedTokens);
        }

        return result;
    }

    /**
     * Returns statistics from the last format operation.
     * Call this after calling format() to get information about what was included/excluded.
     *
     * @return statistics about the last format operation
     */
    public FormatStatistics getLastStatistics() {
        return new FormatStatistics(
            lastTotalFiles,
            lastIncludedFiles,
            lastSkippedFiles,
            lastEstimatedTokens,
            lastWasTruncated
        );
    }

    /**
     * Formats a single file change with clear structure and context.
     */
    private String formatFileChange(DiffParser.FileChange change) {
        StringBuilder sb = new StringBuilder();

        // File header
        sb.append("-".repeat(70)).append("\n");

        // Format header based on change type
        switch (change.getType()) {
            case ADDED -> sb.append("[NEW FILE] ").append(change.getPath()).append("\n");
            case DELETED -> sb.append("[DELETED FILE] ").append(change.getPath()).append("\n");
            case MODIFIED -> sb.append("[MODIFIED] ").append(change.getPath()).append("\n");
            case RENAMED -> {
                sb.append("[RENAMED] ");
                if (change.getOldPath() != null) {
                    sb.append(change.getOldPath()).append(" -> ");
                }
                sb.append(change.getPath()).append("\n");
            }
        }

        // Binary file handling
        if (change.isBinary()) {
            sb.append("(Binary file - content not shown)\n");
        } else if (change.getDiffContent() != null && !change.getDiffContent().trim().isEmpty()) {
            sb.append("\n");
            sb.append(formatDiffContent(change.getDiffContent()));
        } else {
            sb.append("(No content changes or metadata-only change)\n");
        }

        sb.append("\n");

        return sb.toString();
    }

    /**
     * Formats diff content with clear markers for additions and deletions.
     */
    private String formatDiffContent(String diffContent) {
        StringBuilder sb = new StringBuilder();
        String[] lines = diffContent.split("\n");

        for (String line : lines) {
            // Preserve git diff markers for clarity
            if (line.startsWith("@@")) {
                // Hunk header - make it stand out
                sb.append("\n").append(line).append("\n");
            } else if (line.startsWith("+") && !line.startsWith("+++")) {
                // Addition
                sb.append(line).append("\n");
            } else if (line.startsWith("-") && !line.startsWith("---")) {
                // Deletion
                sb.append(line).append("\n");
            } else if (line.startsWith(" ")) {
                // Context line
                sb.append(line).append("\n");
            } else {
                // Other diff metadata or content
                sb.append(line).append("\n");
            }
        }

        return sb.toString();
    }

    /**
     * Estimates the token count for a given text (rough approximation).
     * Uses a simple heuristic of ~4 characters per token.
     *
     * @param text the text to estimate
     * @return estimated number of tokens
     */
    public static int estimateTokens(String text) {
        if (text == null || text.isEmpty()) {
            return 0;
        }
        return text.length() / 4;
    }

    /**
     * Checks if the formatted diff is within safe size limits.
     *
     * @param changes list of file changes to check
     * @return true if the total size is within limits
     */
    public boolean isWithinSizeLimit(List<DiffParser.FileChange> changes) {
        if (changes == null || changes.isEmpty()) {
            return true;
        }

        int totalSize = 0;
        for (DiffParser.FileChange change : changes) {
            totalSize += change.getContentSize();
        }

        return totalSize <= maxSize;
    }

    /**
     * Gets the maximum allowed size for formatted diffs.
     *
     * @return maximum size in characters
     */
    public int getMaxSize() {
        return maxSize;
    }
}
