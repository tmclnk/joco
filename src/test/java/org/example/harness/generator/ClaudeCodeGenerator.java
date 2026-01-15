package org.example.harness.generator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

/**
 * CommitGenerator implementation that uses Claude Code CLI for generation.
 *
 * Invokes the `claude` CLI with the prompt and captures the output.
 * Uses the --print flag to get just the response without interactive mode.
 */
public class ClaudeCodeGenerator implements CommitGenerator {

    private static final Logger logger = LoggerFactory.getLogger(ClaudeCodeGenerator.class);

    private static final Duration DEFAULT_TIMEOUT = Duration.ofMinutes(2);
    private static final String CLAUDE_CLI = "claude";

    private final Duration timeout;

    /**
     * Creates a new ClaudeCodeGenerator with default timeout.
     */
    public ClaudeCodeGenerator() {
        this(DEFAULT_TIMEOUT);
    }

    /**
     * Creates a new ClaudeCodeGenerator with a custom timeout.
     *
     * @param timeout The timeout for CLI execution
     */
    public ClaudeCodeGenerator(Duration timeout) {
        this.timeout = timeout;
    }

    @Override
    public GenerationResult generate(String prompt, GenerationConfig config) throws GenerationException {
        logger.debug("Generating commit message with Claude Code CLI");

        long startTime = System.currentTimeMillis();

        try {
            ProcessBuilder pb = buildProcess(prompt);
            Process process = pb.start();

            // Write prompt to stdin
            try (OutputStreamWriter writer = new OutputStreamWriter(
                    process.getOutputStream(), StandardCharsets.UTF_8)) {
                writer.write(prompt);
                writer.flush();
            }

            // Wait for completion with timeout
            boolean completed = process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS);

            if (!completed) {
                process.destroyForcibly();
                throw new GenerationException(
                    "Claude CLI timed out after " + timeout.toSeconds() + " seconds"
                );
            }

            int exitCode = process.exitValue();

            // Read stdout
            String stdout = readStream(process.inputReader(StandardCharsets.UTF_8));
            String stderr = readStream(process.errorReader(StandardCharsets.UTF_8));

            if (exitCode != 0) {
                logger.error("Claude CLI exited with code {}: {}", exitCode, stderr);
                throw new GenerationException(
                    "Claude CLI failed with exit code " + exitCode + ": " + stderr
                );
            }

            long durationMs = System.currentTimeMillis() - startTime;
            String response = stdout.trim();

            logger.debug("Claude CLI generated response ({} chars) in {}ms",
                response.length(), durationMs);

            return GenerationResult.success(
                response,
                durationMs,
                "claude-code-cli"
            );

        } catch (IOException e) {
            logger.error("I/O error while executing Claude CLI", e);
            throw new GenerationException("I/O error while executing Claude CLI: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Claude CLI execution was interrupted", e);
            throw new GenerationException("Claude CLI execution was interrupted", e);
        }
    }

    /**
     * Builds the ProcessBuilder for invoking Claude CLI.
     *
     * Uses `claude --print -p "prompt"` to get direct output without interactive mode.
     * The prompt is passed via stdin to handle large diffs.
     */
    private ProcessBuilder buildProcess(String prompt) {
        // Use --print to output just the response
        // Use -p to indicate we're sending a prompt
        // Prompt content will be piped via stdin to handle large diffs
        ProcessBuilder pb = new ProcessBuilder(
            CLAUDE_CLI,
            "--print",
            "-p", "-"  // Read prompt from stdin
        );

        // Inherit environment for PATH, etc.
        pb.environment().putAll(System.getenv());

        // Redirect stderr to capture any error messages
        pb.redirectErrorStream(false);

        return pb;
    }

    /**
     * Reads all content from a BufferedReader.
     */
    private String readStream(BufferedReader reader) throws IOException {
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            if (sb.length() > 0) {
                sb.append("\n");
            }
            sb.append(line);
        }
        return sb.toString();
    }

    @Override
    public boolean isAvailable() {
        try {
            ProcessBuilder pb = new ProcessBuilder(CLAUDE_CLI, "--version");
            pb.environment().putAll(System.getenv());
            Process process = pb.start();

            boolean completed = process.waitFor(5, TimeUnit.SECONDS);
            if (!completed) {
                process.destroyForcibly();
                return false;
            }

            return process.exitValue() == 0;

        } catch (Exception e) {
            logger.debug("Claude CLI not available: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public String getBackendName() {
        return "claude";
    }
}
