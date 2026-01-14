package org.example.cli;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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
     *
     * @param originalMessage The original commit message
     * @param reader BufferedReader for reading user input
     * @return The edited commit message, or null if editing was cancelled
     */
    private String editCommitMessage(String originalMessage, BufferedReader reader) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("Edit Commit Message");
        System.out.println("=".repeat(60));
        System.out.println("Enter your commit message (end with Ctrl+D on a new line):");
        System.out.println("Current message:");
        System.out.println(originalMessage);
        System.out.println("\nNew message:");

        StringBuilder editedMessage = new StringBuilder();

        try {
            String line;
            while ((line = reader.readLine()) != null) {
                if (editedMessage.length() > 0) {
                    editedMessage.append("\n");
                }
                editedMessage.append(line);
            }

            // Reset the reader since we've hit EOF
            // Note: In a real scenario, we'd need to handle this differently
            // For now, we'll work with what we have

            String result = editedMessage.toString().trim();

            if (result.isEmpty()) {
                System.out.println("\nEmpty message. Edit cancelled.");
                logger.info("Edit cancelled - empty message");
                return null;
            }

            logger.info("Message edited successfully");
            System.out.println("\nMessage updated.");
            displayCommitMessage(result);

            return result;

        } catch (IOException e) {
            logger.error("Error reading edited message", e);
            System.out.println("\nError reading input. Edit cancelled.");
            return null;
        }
    }
}
