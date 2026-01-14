package org.example.cli;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Interactive commit message editor that prompts the user for actions.
 * Provides options to accept, edit, regenerate, or cancel a generated commit message.
 */
public class CommitMessageEditor {
    private static final Logger logger = LoggerFactory.getLogger(CommitMessageEditor.class);

    /**
     * Result of the interactive prompt session.
     */
    public enum Action {
        ACCEPT,      // User accepted the commit message
        EDIT,        // User wants to edit the commit message
        REGENERATE,  // User wants to regenerate the commit message
        CANCEL       // User wants to cancel the operation
    }

    /**
     * Result containing the action and potentially modified commit message.
     */
    public record Result(Action action, String commitMessage) {}

    /**
     * Displays the generated commit message and prompts the user for action.
     *
     * @param commitMessage The generated commit message to display
     * @return Result containing the user's chosen action and the (possibly edited) commit message
     */
    public Result prompt(String commitMessage) {
        logger.debug("Starting interactive prompt with commit message: {}", commitMessage);

        // Display the generated commit message
        displayCommitMessage(commitMessage);

        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        while (true) {
            // Display prompt
            System.out.println("\nWhat would you like to do?");
            System.out.println("  [a] Accept - Use this commit message");
            System.out.println("  [e] Edit - Modify the commit message");
            System.out.println("  [r] Regenerate - Generate a new commit message");
            System.out.println("  [c] Cancel - Abort the commit");
            System.out.print("\nYour choice: ");
            System.out.flush();

            try {
                String input = reader.readLine();

                if (input == null) {
                    // EOF reached (Ctrl+D)
                    logger.info("EOF reached, cancelling");
                    System.out.println("\nCancelled.");
                    return new Result(Action.CANCEL, commitMessage);
                }

                input = input.trim().toLowerCase();
                logger.debug("User input: {}", input);

                switch (input) {
                    case "a", "accept" -> {
                        logger.info("User accepted commit message");
                        return new Result(Action.ACCEPT, commitMessage);
                    }
                    case "e", "edit" -> {
                        logger.info("User chose to edit commit message");
                        String editedMessage = editCommitMessage(commitMessage, reader);
                        if (editedMessage != null) {
                            return new Result(Action.EDIT, editedMessage);
                        }
                        // If edit was cancelled, continue the loop
                    }
                    case "r", "regenerate" -> {
                        logger.info("User chose to regenerate commit message");
                        return new Result(Action.REGENERATE, commitMessage);
                    }
                    case "c", "cancel" -> {
                        logger.info("User cancelled operation");
                        System.out.println("\nCancelled.");
                        return new Result(Action.CANCEL, commitMessage);
                    }
                    default -> {
                        System.out.println("Invalid choice. Please enter 'a', 'e', 'r', or 'c'.");
                    }
                }
            } catch (IOException e) {
                logger.error("Error reading user input", e);
                System.out.println("\nError reading input. Cancelling.");
                return new Result(Action.CANCEL, commitMessage);
            }
        }
    }

    /**
     * Displays the commit message in a formatted box.
     *
     * @param commitMessage The commit message to display
     */
    private void displayCommitMessage(String commitMessage) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Generated Commit Message:");
        System.out.println("=".repeat(60));
        System.out.println(commitMessage);
        System.out.println("=".repeat(60));
    }

    /**
     * Allows the user to edit the commit message interactively.
     * Launches the default editor (from EDITOR environment variable or fallback to nano/vim).
     *
     * @param originalMessage The original commit message
     * @param reader BufferedReader for reading user input (unused but kept for API compatibility)
     * @return The edited commit message, or null if editing was cancelled
     */
    private String editCommitMessage(String originalMessage, BufferedReader reader) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Edit Commit Message");
        System.out.println("=".repeat(60));

        Path tempFile = null;
        try {
            // Create a temporary file for editing
            tempFile = Files.createTempFile("joco-commit-", ".txt");
            logger.debug("Created temporary file: {}", tempFile);

            // Write the original message to the temp file
            Files.writeString(tempFile, originalMessage, StandardCharsets.UTF_8);
            logger.debug("Wrote original message to temp file");

            // Determine which editor to use
            String editor = getEditor();
            logger.info("Using editor: {}", editor);
            System.out.println("Opening editor: " + editor);

            // Launch the editor
            ProcessBuilder pb = new ProcessBuilder(editor, tempFile.toString());
            pb.inheritIO(); // Connect the editor to the current terminal
            Process process = pb.start();

            // Wait for the editor to close
            int exitCode = process.waitFor();
            logger.debug("Editor exited with code: {}", exitCode);

            // Check the exit code
            if (exitCode != 0) {
                System.out.println("\nEditor exited with error code " + exitCode + ". Edit cancelled.");
                logger.warn("Editor exited with non-zero code: {}", exitCode);
                return null;
            }

            // Read the edited message back
            String editedMessage = Files.readString(tempFile, StandardCharsets.UTF_8).trim();
            logger.debug("Read edited message: {} characters", editedMessage.length());

            if (editedMessage.isEmpty()) {
                System.out.println("\nEmpty message. Edit cancelled.");
                logger.info("Edit cancelled - empty message");
                return null;
            }

            logger.info("Message edited successfully");
            System.out.println("\nMessage updated.");
            displayCommitMessage(editedMessage);

            return editedMessage;

        } catch (IOException e) {
            logger.error("Error during file operations", e);
            System.out.println("\nError accessing temporary file. Edit cancelled.");
            return null;
        } catch (InterruptedException e) {
            logger.error("Editor process was interrupted", e);
            System.out.println("\nEditor process was interrupted. Edit cancelled.");
            Thread.currentThread().interrupt(); // Restore interrupt status
            return null;
        } finally {
            // Clean up the temporary file
            if (tempFile != null) {
                try {
                    Files.deleteIfExists(tempFile);
                    logger.debug("Deleted temporary file: {}", tempFile);
                } catch (IOException e) {
                    logger.warn("Failed to delete temporary file: {}", tempFile, e);
                }
            }
        }
    }

    /**
     * Determines which editor to use based on the EDITOR environment variable.
     * Falls back to nano if available, then vim, then vi.
     *
     * @return The editor command to use
     */
    private String getEditor() {
        // Check EDITOR environment variable first
        String editor = System.getenv("EDITOR");
        if (editor != null && !editor.trim().isEmpty()) {
            logger.debug("Using EDITOR from environment: {}", editor);
            return editor.trim();
        }

        // Try common editors in order of preference
        String[] fallbackEditors = {"nano", "vim", "vi"};
        for (String candidate : fallbackEditors) {
            if (isCommandAvailable(candidate)) {
                logger.debug("Using fallback editor: {}", candidate);
                return candidate;
            }
        }

        // Ultimate fallback (vi should be on all Unix systems)
        logger.warn("No preferred editor found, using vi as last resort");
        return "vi";
    }

    /**
     * Checks if a command is available on the system PATH.
     *
     * @param command The command to check
     * @return true if the command is available, false otherwise
     */
    private boolean isCommandAvailable(String command) {
        try {
            ProcessBuilder pb = new ProcessBuilder("which", command);
            Process process = pb.start();
            int exitCode = process.waitFor();
            boolean available = (exitCode == 0);
            logger.debug("Command '{}' available: {}", command, available);
            return available;
        } catch (IOException | InterruptedException e) {
            logger.debug("Error checking command availability for '{}': {}", command, e.getMessage());
            return false;
        }
    }
}
