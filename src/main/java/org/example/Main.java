package org.example;

import org.example.cli.CommandLineArgs;
import org.example.cli.CommitMessageEditor;
import org.example.config.Config;
import org.example.config.ConfigReader;
import org.example.git.DiffFormatter;
import org.example.git.GitException;
import org.example.git.GitRepository;
import org.example.ollama.GenerateRequest;
import org.example.ollama.GenerateResponse;
import org.example.ollama.MultiStepCommitGenerator;
import org.example.ollama.OllamaClient;
import org.example.ollama.OllamaConnectionException;
import org.example.ollama.OllamaException;
import org.example.util.MessageValidator;
import org.example.util.OutputFormatter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Main entry point for the joco commit message generator.
 * Orchestrates the workflow: load config, check git, get diff, call Ollama, prompt user.
 */
public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        try {
            // Step 0: Parse command-line arguments
            CommandLineArgs cmdArgs = CommandLineArgs.parse(args);

            // Handle parsing errors
            if (cmdArgs.hasErrors()) {
                for (String error : cmdArgs.getErrors()) {
                    OutputFormatter.error(error);
                }
                OutputFormatter.hint("Use --help for usage information.");
                System.exit(1);
            }

            // Handle --help flag
            if (cmdArgs.isHelp()) {
                CommandLineArgs.printHelp();
                System.exit(0);
            }

            // Handle --version flag
            if (cmdArgs.isVersion()) {
                CommandLineArgs.printVersion();
                System.exit(0);
            }

            // Step 1: Load configuration
            System.out.println("Loading configuration...");
            ConfigReader configReader = new ConfigReader();
            Config config = configReader.read();

            // Apply model override if provided
            if (cmdArgs.getModelOverride() != null) {
                try {
                    config = config.withModel(cmdArgs.getModelOverride());
                    logger.info("Model overridden via command line: {}", cmdArgs.getModelOverride());
                } catch (IllegalArgumentException e) {
                    OutputFormatter.error("Invalid model name: " + e.getMessage());
                    OutputFormatter.hint("Check available models with:");
                    OutputFormatter.suggestion("ollama list");
                    System.exit(1);
                }
            }

            logger.info("Configuration loaded: {}", config);
            System.out.println("Using model: " + config.getModel());

            // Check if dry-run mode is enabled
            final boolean isDryRun = cmdArgs.isDryRun();
            if (isDryRun) {
                System.out.println("Running in dry-run mode (no commit will be created)");
            }

            // Step 2: Check Git repository
            System.out.println("\nChecking Git repository...");
            GitRepository gitRepo = new GitRepository();

            if (!gitRepo.isInsideRepository()) {
                OutputFormatter.error("Not inside a Git repository.");
                OutputFormatter.hint("Run this command from within a Git repository.");
                System.exit(1);
            }

            logger.info("Inside Git repository at: {}", gitRepo.getRepositoryRoot());
            System.out.println("Git repository detected.");

            // Check for detached HEAD state
            if (gitRepo.isDetachedHead()) {
                OutputFormatter.warning("You are in detached HEAD state.");
                OutputFormatter.hint("Commits made here may be lost. Consider checking out a branch:");
                OutputFormatter.suggestion("git checkout -b <new-branch-name>");
                OutputFormatter.info("");
                // Continue anyway - user may want to commit in detached state
            }

            // Check for merge conflicts
            if (gitRepo.hasMergeConflicts()) {
                List<String> conflictedFiles = gitRepo.getConflictedFiles();
                OutputFormatter.error("There are unresolved merge conflicts.");
                OutputFormatter.hint("Resolve conflicts in the following files:");
                for (String file : conflictedFiles) {
                    OutputFormatter.suggestion("- " + file);
                }
                OutputFormatter.info("");
                OutputFormatter.hint("After resolving, stage the files with 'git add' and try again.");
                return;
            }

            // Info for initial commit
            if (gitRepo.isInitialCommit()) {
                System.out.println("Note: This will be the initial commit for this repository.");
                System.out.println();
            }

            // Step 3: Get staged changes (formatted for LLM consumption)
            System.out.println("\nRetrieving staged changes...");
            String diff;
            try {
                diff = gitRepo.getFormattedStagedDiff();
            } catch (GitException e) {
                OutputFormatter.error(e.getMessage());
                if (e.getMessage().contains("No staged changes")) {
                    OutputFormatter.hint("To stage changes, use:");
                    OutputFormatter.suggestion("git add <files>    # Stage specific files");
                    OutputFormatter.suggestion("git add .          # Stage all changes");
                }
                System.exit(1);
                return; // Unreachable, but helps compiler
            }

            // Check for diff truncation and warn user
            DiffFormatter.FormatStatistics stats = gitRepo.getDiffFormatter().getLastStatistics();
            if (stats.wasTruncated()) {
                OutputFormatter.info("");
                OutputFormatter.warning("Large diff detected - " + stats.skippedFiles() + " file(s) were excluded.");
                OutputFormatter.info("Included " + stats.includedFiles() + " of " + stats.totalFiles() + " files (~" + stats.estimatedTokens() + " tokens).");
                OutputFormatter.hint("For better results, consider staging fewer files at a time.");
                OutputFormatter.info("");
            }

            logger.info("Retrieved diff with {} characters", diff.length());
            System.out.println("Found staged changes.");

            // Step 4: Check Ollama availability
            System.out.println("\nConnecting to Ollama...");
            OllamaClient ollamaClient = new OllamaClient();

            if (!ollamaClient.isAvailable()) {
                OutputFormatter.error("Cannot connect to Ollama at " + ollamaClient.getBaseUrl());
                OutputFormatter.hint("Make sure Ollama is running:");
                OutputFormatter.suggestion("ollama serve");
                OutputFormatter.hint("And pull the required model:");
                OutputFormatter.suggestion("ollama pull " + config.getModel());
                System.exit(1);
            }

            System.out.println("Connected to Ollama.");

            // Step 5: Generate commit message using multi-step approach
            System.out.println("\nGenerating commit message...");

            MultiStepCommitGenerator generator = new MultiStepCommitGenerator(
                ollamaClient,
                config.getModel(),
                config.getTemperature()
            );

            String commitMessage;
            try {
                commitMessage = generator.generate(diff);
                logger.info("Generated commit message: {}", commitMessage);
            } catch (OllamaConnectionException e) {
                OutputFormatter.error("Lost connection to Ollama during generation.");
                OutputFormatter.info(e.getMessage());
                OutputFormatter.hint("Make sure Ollama is running and try again:");
                OutputFormatter.suggestion("ollama serve");
                System.exit(1);
                return; // Unreachable, but helps compiler
            } catch (OllamaException e) {
                OutputFormatter.error("Failed to generate commit message.");
                OutputFormatter.info(e.getMessage());
                logger.error("Ollama generation failed", e);
                System.exit(1);
                return; // Unreachable, but helps compiler
            }

            // Validate the generated message
            try {
                commitMessage = MessageValidator.validateAndClean(commitMessage);
                logger.info("Validated commit message: {}", commitMessage);
            } catch (IllegalArgumentException e) {
                OutputFormatter.error(e.getMessage());
                OutputFormatter.hint("Try regenerating the message or use a different model.");
                logger.error("Message validation failed", e);
                System.exit(1);
                return; // Unreachable, but helps compiler
            }

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
                    OutputFormatter.warning("Regeneration not yet implemented.");
                    OutputFormatter.hint("This feature will be available in a future version.");
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

            // Step 7: Execute commit (or skip if dry-run)
            if (isDryRun) {
                // Dry-run mode: just display the message
                System.out.println("\n" + "=".repeat(60));
                System.out.println("DRY-RUN MODE: Commit message generated");
                System.out.println("=".repeat(60));
                System.out.println("\nGenerated commit message:");
                System.out.println("---");
                System.out.println(commitMessage);
                System.out.println("---");
                System.out.println("\nNo commit was created (dry-run mode).");
                System.out.println("To create the commit, run without --dry-run flag.");
                logger.info("Dry-run completed, no commit created");
            } else {
                // Normal mode: create the commit
                System.out.println("\nCommitting changes...");
                String commitHash;
                try {
                    commitHash = gitRepo.commit(commitMessage);
                } catch (GitException e) {
                    OutputFormatter.error("Failed to create commit.");
                    OutputFormatter.info(e.getMessage());
                    logger.error("Commit execution failed", e);
                    System.exit(1);
                    return; // Unreachable, but helps compiler
                }

                // Step 8: Display success message with commit hash
                OutputFormatter.info("\n" + "=".repeat(60));
                OutputFormatter.success("Commit created successfully!");
                OutputFormatter.info("=".repeat(60));
                OutputFormatter.info("Commit hash: " + commitHash);
                OutputFormatter.info("=".repeat(60));
            }

        } catch (GitException e) {
            OutputFormatter.error("Git operation failed.");
            OutputFormatter.info(e.getMessage());
            logger.error("Git operation failed", e);
            System.exit(1);
        } catch (Exception e) {
            OutputFormatter.error("Unexpected error occurred.");
            OutputFormatter.info(e.getMessage());
            logger.error("Unexpected error", e);
            System.exit(1);
        }
    }
}
