package org.example.git;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses git diff output to extract structured information about changed files.
 * Identifies file changes, binary files, and organizes diff content for processing.
 */
public class DiffParser {
    private static final Logger logger = LoggerFactory.getLogger(DiffParser.class);

    // Patterns for parsing git diff output
    private static final Pattern DIFF_HEADER = Pattern.compile("^diff --git a/(.*) b/(.*)$");
    private static final Pattern NEW_FILE = Pattern.compile("^new file mode (.*)$");
    private static final Pattern DELETED_FILE = Pattern.compile("^deleted file mode (.*)$");
    private static final Pattern BINARY_FILE = Pattern.compile("^Binary files .* differ$");
    private static final Pattern FILE_PATH_A = Pattern.compile("^--- a/(.*)$");
    private static final Pattern FILE_PATH_B = Pattern.compile("^\\+\\+\\+ b/(.*)$");
    private static final Pattern RENAME_FROM = Pattern.compile("^rename from (.*)$");
    private static final Pattern RENAME_TO = Pattern.compile("^rename to (.*)$");

    /**
     * Represents a single file change in a diff.
     */
    public static class FileChange {
        private final String path;
        private final ChangeType type;
        private final boolean isBinary;
        private final String diffContent;
        private final String oldPath; // For renames

        public enum ChangeType {
            ADDED,
            MODIFIED,
            DELETED,
            RENAMED
        }

        public FileChange(String path, ChangeType type, boolean isBinary, String diffContent, String oldPath) {
            this.path = path;
            this.type = type;
            this.isBinary = isBinary;
            this.diffContent = diffContent;
            this.oldPath = oldPath;
        }

        public String getPath() {
            return path;
        }

        public ChangeType getType() {
            return type;
        }

        public boolean isBinary() {
            return isBinary;
        }

        public String getDiffContent() {
            return diffContent;
        }

        public String getOldPath() {
            return oldPath;
        }

        public int getContentSize() {
            return diffContent != null ? diffContent.length() : 0;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("FileChange{");
            sb.append("path='").append(path).append('\'');
            sb.append(", type=").append(type);
            sb.append(", isBinary=").append(isBinary);
            if (oldPath != null) {
                sb.append(", oldPath='").append(oldPath).append('\'');
            }
            sb.append(", contentSize=").append(getContentSize());
            sb.append('}');
            return sb.toString();
        }
    }

    /**
     * Parses git diff output and extracts structured information about changed files.
     *
     * @param diffOutput the raw git diff output
     * @return a list of FileChange objects representing each changed file
     */
    public List<FileChange> parse(String diffOutput) {
        if (diffOutput == null || diffOutput.trim().isEmpty()) {
            logger.debug("Empty diff output provided");
            return new ArrayList<>();
        }

        List<FileChange> changes = new ArrayList<>();
        String[] lines = diffOutput.split("\n");

        int i = 0;
        while (i < lines.length) {
            String line = lines[i];
            Matcher diffMatcher = DIFF_HEADER.matcher(line);

            if (diffMatcher.matches()) {
                // Start of a new file diff
                String pathA = diffMatcher.group(1);
                String pathB = diffMatcher.group(2);

                // Parse this file's diff section
                ParsedFile parsed = parseFileDiff(lines, i);
                i = parsed.nextIndex;

                // Create FileChange object
                FileChange change = createFileChange(pathA, pathB, parsed);
                changes.add(change);

                logger.debug("Parsed file change: {}", change);
            } else {
                i++;
            }
        }

        logger.info("Parsed {} file changes from diff", changes.size());
        return changes;
    }

    /**
     * Helper class to hold parsed file information.
     */
    private static class ParsedFile {
        boolean isNew = false;
        boolean isDeleted = false;
        boolean isBinary = false;
        boolean isRenamed = false;
        String renameFrom = null;
        String renameTo = null;
        StringBuilder content = new StringBuilder();
        int nextIndex;
    }

    /**
     * Parses a single file's diff section.
     */
    private ParsedFile parseFileDiff(String[] lines, int startIndex) {
        ParsedFile parsed = new ParsedFile();
        int i = startIndex + 1; // Skip the diff header line

        // Parse metadata and content
        boolean inHunk = false;
        while (i < lines.length) {
            String line = lines[i];

            // Check if we've reached the next file's diff header
            if (DIFF_HEADER.matcher(line).matches()) {
                break;
            }

            // Check for file status markers
            if (NEW_FILE.matcher(line).matches()) {
                parsed.isNew = true;
            } else if (DELETED_FILE.matcher(line).matches()) {
                parsed.isDeleted = true;
            } else if (BINARY_FILE.matcher(line).matches()) {
                parsed.isBinary = true;
            } else if (line.startsWith("rename from ")) {
                Matcher m = RENAME_FROM.matcher(line);
                if (m.matches()) {
                    parsed.isRenamed = true;
                    parsed.renameFrom = m.group(1);
                }
            } else if (line.startsWith("rename to ")) {
                Matcher m = RENAME_TO.matcher(line);
                if (m.matches()) {
                    parsed.isRenamed = true;
                    parsed.renameTo = m.group(1);
                }
            } else if (line.startsWith("@@")) {
                // Start of a hunk
                inHunk = true;
                parsed.content.append(line).append("\n");
            } else if (inHunk) {
                // Inside a hunk - content lines
                parsed.content.append(line).append("\n");
            } else if (line.startsWith("---") || line.startsWith("+++") ||
                       line.startsWith("index ") || line.startsWith("similarity index") ||
                       line.startsWith("dissimilarity index")) {
                // Metadata lines - skip for content but continue parsing
            } else if (!line.trim().isEmpty()) {
                // Other metadata or content
                parsed.content.append(line).append("\n");
            }

            i++;
        }

        parsed.nextIndex = i;
        return parsed;
    }

    /**
     * Creates a FileChange object from parsed data.
     */
    private FileChange createFileChange(String pathA, String pathB, ParsedFile parsed) {
        String path = pathB;
        String oldPath = null;
        FileChange.ChangeType type;

        if (parsed.isRenamed) {
            type = FileChange.ChangeType.RENAMED;
            path = parsed.renameTo != null ? parsed.renameTo : pathB;
            oldPath = parsed.renameFrom != null ? parsed.renameFrom : pathA;
        } else if (parsed.isNew) {
            type = FileChange.ChangeType.ADDED;
        } else if (parsed.isDeleted) {
            type = FileChange.ChangeType.DELETED;
        } else {
            type = FileChange.ChangeType.MODIFIED;
        }

        String diffContent = parsed.isBinary ? null : parsed.content.toString();

        return new FileChange(path, type, parsed.isBinary, diffContent, oldPath);
    }
}
