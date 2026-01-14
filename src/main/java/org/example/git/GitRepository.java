package org.example.git;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Handles Git repository operations including detection, navigation, and diff retrieval.
 * Provides methods to check if current directory is a Git repository and retrieve staged changes.
 */
public class GitRepository {
    private static final Logger logger = LoggerFactory.getLogger(GitRepository.class);

    /**
     * Checks if the current directory is inside a Git repository.
     *
     * @return true if inside a Git repository, false otherwise
     */
    public boolean isInsideRepository() {
        try {
            Process process = new ProcessBuilder("git", "rev-parse", "--git-dir")
                .redirectErrorStream(true)
                .start();

            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (IOException | InterruptedException e) {
            logger.debug("Failed to check if inside Git repository: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Gets the root directory of the Git repository.
     *
     * @return the path to the repository root
     * @throws GitException if not in a Git repository or command fails
     */
    public Path getRepositoryRoot() throws GitException {
        try {
            Process process = new ProcessBuilder("git", "rev-parse", "--show-toplevel")
                .redirectErrorStream(true)
                .start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String rootPath = reader.readLine();

            int exitCode = process.waitFor();
            if (exitCode != 0 || rootPath == null) {
                throw new GitException("Not inside a Git repository");
            }

            return Paths.get(rootPath);
        } catch (IOException e) {
            throw new GitException("Failed to get repository root: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new GitException("Operation interrupted while getting repository root", e);
        }
    }

    /**
     * Retrieves the staged changes (git diff --cached).
     *
     * @return the git diff output for staged changes
     * @throws GitException if not in a Git repository or no staged changes exist
     */
    public String getStagedDiff() throws GitException {
        try {
            Process process = new ProcessBuilder("git", "diff", "--cached")
                .redirectErrorStream(true)
                .start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder diff = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                diff.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode != 0) {
                throw new GitException("Failed to get staged diff (exit code: " + exitCode + ")");
            }

            String diffOutput = diff.toString().trim();
            if (diffOutput.isEmpty()) {
                throw new GitException("No staged changes found. Use 'git add' to stage your changes first.");
            }

            return diffOutput;
        } catch (IOException e) {
            throw new GitException("Failed to retrieve staged diff: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new GitException("Operation interrupted while retrieving staged diff", e);
        }
    }

    /**
     * Checks if there are any staged changes.
     *
     * @return true if there are staged changes, false otherwise
     */
    public boolean hasStagedChanges() {
        try {
            Process process = new ProcessBuilder("git", "diff", "--cached", "--quiet")
                .redirectErrorStream(true)
                .start();

            // Exit code 0 means no changes, exit code 1 means changes exist
            int exitCode = process.waitFor();
            return exitCode == 1;
        } catch (IOException | InterruptedException e) {
            logger.debug("Failed to check for staged changes: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Executes a git commit with the provided message.
     *
     * @param message the commit message to use
     * @return the commit hash (SHA) of the created commit
     * @throws GitException if the commit fails or cannot retrieve the commit hash
     */
    public String commit(String message) throws GitException {
        if (message == null || message.trim().isEmpty()) {
            throw new GitException("Commit message cannot be empty");
        }

        try {
            // Execute git commit
            Process commitProcess = new ProcessBuilder("git", "commit", "-m", message)
                .redirectErrorStream(true)
                .start();

            BufferedReader commitReader = new BufferedReader(new InputStreamReader(commitProcess.getInputStream()));
            StringBuilder commitOutput = new StringBuilder();
            String line;
            while ((line = commitReader.readLine()) != null) {
                commitOutput.append(line).append("\n");
            }

            int commitExitCode = commitProcess.waitFor();
            if (commitExitCode != 0) {
                String errorMsg = commitOutput.toString().trim();
                if (errorMsg.isEmpty()) {
                    errorMsg = "Commit failed with exit code: " + commitExitCode;
                }
                throw new GitException("Failed to create commit: " + errorMsg);
            }

            // Get the commit hash of the newly created commit
            Process hashProcess = new ProcessBuilder("git", "rev-parse", "HEAD")
                .redirectErrorStream(true)
                .start();

            BufferedReader hashReader = new BufferedReader(new InputStreamReader(hashProcess.getInputStream()));
            String commitHash = hashReader.readLine();

            int hashExitCode = hashProcess.waitFor();
            if (hashExitCode != 0 || commitHash == null || commitHash.trim().isEmpty()) {
                throw new GitException("Failed to retrieve commit hash after successful commit");
            }

            logger.info("Commit created successfully with hash: {}", commitHash.trim());
            return commitHash.trim();

        } catch (IOException e) {
            throw new GitException("Failed to execute commit: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new GitException("Operation interrupted while creating commit", e);
        }
    }
}
