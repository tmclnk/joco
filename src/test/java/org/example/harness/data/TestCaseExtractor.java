package org.example.harness.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Extracts test cases from a git repository.
 * Filters for conventional commit format only.
 */
public class TestCaseExtractor {

    private static final Logger logger = LoggerFactory.getLogger(TestCaseExtractor.class);

    private static final Pattern CONVENTIONAL_COMMIT = Pattern.compile(
        "^(feat|fix|chore|docs|refactor|test|style|perf|build|ci)(\\([\\w.-]+\\))?: .+"
    );

    private final Path repoPath;
    private final String repoName;
    private boolean filterConventional = true;
    private String authorFilter = null;

    public TestCaseExtractor(Path repoPath, String repoName) {
        this.repoPath = repoPath;
        this.repoName = repoName;
    }

    /**
     * Sets whether to filter for conventional commits only.
     * When false, all commits are extracted regardless of format.
     */
    public TestCaseExtractor setFilterConventional(boolean filter) {
        this.filterConventional = filter;
        return this;
    }

    /**
     * Sets an author email or name pattern to filter commits.
     * When set, only commits from matching authors are extracted.
     */
    public TestCaseExtractor setAuthorFilter(String author) {
        this.authorFilter = author;
        return this;
    }

    /**
     * Extracts commit + diff pairs from the repository.
     *
     * @param maxCommits Maximum number of commits to extract
     * @param maxDiffSize Maximum diff size in characters (larger diffs are skipped)
     * @return List of test cases
     */
    public List<TestCase> extract(int maxCommits, int maxDiffSize) throws IOException {
        if (!Files.isDirectory(repoPath.resolve(".git"))) {
            throw new IOException("Not a git repository: " + repoPath);
        }

        List<String> commitHashes = getRecentCommits(maxCommits * 3); // Get more to filter
        List<TestCase> testCases = new ArrayList<>();

        logger.info("Found {} commits to analyze", commitHashes.size());

        for (String hash : commitHashes) {
            if (testCases.size() >= maxCommits) {
                break;
            }

            try {
                String message = getCommitMessage(hash);

                // Optionally filter for conventional commits only
                if (filterConventional && !isConventionalCommit(message)) {
                    logger.debug("Skipping non-conventional commit: {}", hash.substring(0, 8));
                    continue;
                }

                String diff = getCommitDiff(hash);

                // Skip very large diffs
                if (diff.length() > maxDiffSize) {
                    logger.debug("Skipping large diff ({} chars): {}", diff.length(), hash.substring(0, 8));
                    continue;
                }

                // Skip empty or tiny diffs
                if (diff.length() < 10) {
                    logger.debug("Skipping empty diff: {}", hash.substring(0, 8));
                    continue;
                }

                String id = repoName.replaceAll("[^a-zA-Z0-9]", "-") + "-" + hash.substring(0, 8);
                String firstLine = message.lines().findFirst().orElse(message);
                String author = getCommitAuthor(hash);

                TestCase tc = new TestCase(
                    id,
                    diff,
                    firstLine,
                    repoName,
                    hash,
                    Map.of("author", author)
                );

                testCases.add(tc);
                logger.info("Extracted {}: {}", id, firstLine);

            } catch (Exception e) {
                logger.warn("Failed to extract commit {}: {}", hash, e.getMessage());
            }
        }

        logger.info("Extracted {} test cases", testCases.size());
        return testCases;
    }

    private boolean isConventionalCommit(String message) {
        String firstLine = message.lines().findFirst().orElse(message);
        return CONVENTIONAL_COMMIT.matcher(firstLine).matches();
    }

    private List<String> getRecentCommits(int count) throws IOException {
        List<String> args = new ArrayList<>(List.of(
            "git", "log", "--format=%H", "-n", String.valueOf(count)
        ));

        if (authorFilter != null && !authorFilter.isBlank()) {
            args.add("--author=" + authorFilter);
        }

        ProcessBuilder pb = new ProcessBuilder(args);
        pb.directory(repoPath.toFile());

        return runCommand(pb);
    }

    private String getCommitMessage(String hash) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(
            "git", "log", "-1", "--format=%B", hash
        );
        pb.directory(repoPath.toFile());

        List<String> lines = runCommand(pb);
        return String.join("\n", lines).trim();
    }

    private String getCommitDiff(String hash) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(
            "git", "show", hash, "--format=", "--stat=80", "--patch"
        );
        pb.directory(repoPath.toFile());

        List<String> lines = runCommand(pb);
        return String.join("\n", lines);
    }

    private String getCommitAuthor(String hash) throws IOException {
        ProcessBuilder pb = new ProcessBuilder(
            "git", "log", "-1", "--format=%an <%ae>", hash
        );
        pb.directory(repoPath.toFile());

        List<String> lines = runCommand(pb);
        return lines.isEmpty() ? "" : lines.get(0);
    }

    private List<String> runCommand(ProcessBuilder pb) throws IOException {
        pb.redirectErrorStream(true);
        Process process = pb.start();

        List<String> lines = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
            String line;
            while ((line = reader.readLine()) != null) {
                lines.add(line);
            }
        }

        try {
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new IOException("Git command failed with exit code " + exitCode + ": " + String.join("\n", lines));
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Git command interrupted", e);
        }

        return lines;
    }
}
