package org.example;

import org.example.cli.CommitMessageEditor;
import org.example.config.Config;
import org.example.config.ConfigReader;
import org.example.git.GitException;
import org.example.git.GitRepository;
import org.example.ollama.CommitMessagePrompt;
import org.example.ollama.GenerateRequest;
import org.example.ollama.GenerateResponse;
import org.example.ollama.OllamaClient;
import org.example.ollama.OllamaConnectionException;
import org.example.ollama.OllamaException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main entry point for the joco commit message generator.
 * Orchestrates the workflow: load config, check git, get diff, call Ollama, prompt user.
 */
public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        try {
            // Step 1: Load configuration
            System.out.println("Loading configuration...");
            ConfigReader configReader = new ConfigReader();
            Config config = configReader.read();
            logger.info("Configuration loaded: {}", config);
            System.out.println("Using model: " + config.getModel());

            // Step 2: Check Git repository
            System.out.println("\nChecking Git repository...");
            GitRepository gitRepo = new GitRepository();

            if (!gitRepo.isInsideRepository()) {
                System.out.println("ERROR: Not inside a Git repository.");
                System.out.println("Please run this command from within a Git repository.");
                System.exit(1);
            }

            logger.info("Inside Git repository at: {}", gitRepo.getRepositoryRoot());
            System.out.println("Git repository detected.");

            // Step 3: Get staged changes
            System.out.println("\nRetrieving staged changes...");
            String diff;
            try {
                diff = gitRepo.getStagedDiff();
            } catch (GitException e) {
                System.out.println("ERROR: " + e.getMessage());
                if (e.getMessage().contains("No staged changes")) {
                    System.out.println("\nTo stage changes, use:");
                    System.out.println("  git add <files>    # Stage specific files");
                    System.out.println("  git add .          # Stage all changes");
                }
                System.exit(1);
                return; // Unreachable, but helps compiler
            }

            logger.info("Retrieved diff with {} characters", diff.length());
            System.out.println("Found staged changes.");

            // Step 4: Check Ollama availability
            System.out.println("\nConnecting to Ollama...");
            OllamaClient ollamaClient = new OllamaClient();

            if (!ollamaClient.isAvailable()) {
                System.out.println("ERROR: Cannot connect to Ollama at " + ollamaClient.getBaseUrl());
                System.out.println("\nPlease ensure Ollama is running:");
                System.out.println("  1. Start Ollama: ollama serve");
                System.out.println("  2. Pull the model: ollama pull " + config.getModel());
                System.exit(1);
            }

            System.out.println("Connected to Ollama.");

            // Step 5: Generate commit message
            System.out.println("\nGenerating commit message...");
            String prompt = CommitMessagePrompt.createCompletePrompt(diff);

            GenerateRequest request = new GenerateRequest(
                config.getModel(),
                prompt,
                false,
                new GenerateRequest.Options(
                    config.getTemperature(),
                    config.getMaxTokens()
                )
            );

            GenerateResponse response;
            try {
                response = ollamaClient.generate(request);
            } catch (OllamaConnectionException e) {
                System.out.println("ERROR: Lost connection to Ollama during generation.");
                System.out.println(e.getMessage());
                System.exit(1);
                return; // Unreachable, but helps compiler
            } catch (OllamaException e) {
                System.out.println("ERROR: Failed to generate commit message.");
                System.out.println(e.getMessage());
                logger.error("Ollama generation failed", e);
                System.exit(1);
                return; // Unreachable, but helps compiler
            }

            String commitMessage = response.response().trim();
            logger.info("Generated commit message: {}", commitMessage);

            // Step 6: Interactive prompt for user action
            CommitMessageEditor editor = new CommitMessageEditor();
            CommitMessageEditor.Result result = editor.prompt(commitMessage);

            // Handle user's choice
            switch (result.action()) {
                case ACCEPT -> {
                    // Continue to commit with the current message
                    commitMessage = result.commitMessage();
                    logger.info("User accepted commit message");
                }
                case EDIT -> {
                    // User edited the message, use the edited version
                    commitMessage = result.commitMessage();
                    logger.info("User edited commit message to: {}", commitMessage);
                }
                case REGENERATE -> {
                    // TODO: Implement regeneration in future task (joco-fco or related)
                    System.out.println("\nRegeneration not yet implemented.");
                    System.out.println("This feature will be available in a future version.");
                    System.exit(0);
                    return;
                }
                case CANCEL -> {
                    // User cancelled, exit gracefully
                    logger.info("User cancelled commit operation");
                    System.exit(0);
                    return;
                }
            }

            // Step 7: Execute commit
            System.out.println("\nCommitting changes...");
            String commitHash;
            try {
                commitHash = gitRepo.commit(commitMessage);
            } catch (GitException e) {
                System.out.println("ERROR: Failed to create commit.");
                System.out.println(e.getMessage());
                logger.error("Commit execution failed", e);
                System.exit(1);
                return; // Unreachable, but helps compiler
            }

            // Step 8: Display success message with commit hash
            System.out.println("\n" + "=".repeat(60));
            System.out.println("Commit created successfully!");
            System.out.println("=".repeat(60));
            System.out.println("Commit hash: " + commitHash);
            System.out.println("=".repeat(60));

        } catch (GitException e) {
            System.out.println("\nERROR: Git operation failed.");
            System.out.println(e.getMessage());
            logger.error("Git operation failed", e);
            System.exit(1);
        } catch (Exception e) {
            System.out.println("\nERROR: Unexpected error occurred.");
            System.out.println(e.getMessage());
            logger.error("Unexpected error", e);
            System.exit(1);
        }
    }
}
