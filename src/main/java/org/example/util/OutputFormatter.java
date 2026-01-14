package org.example.util;

/**
 * Centralized output formatting with ANSI color support.
 * Automatically detects terminal capabilities and respects NO_COLOR.
 */
public final class OutputFormatter {
    // ANSI codes
    private static final String RESET = "\u001B[0m";
    private static final String RED = "\u001B[31m";
    private static final String GREEN = "\u001B[32m";
    private static final String YELLOW = "\u001B[33m";
    private static final String BLUE = "\u001B[34m";
    private static final String BOLD = "\u001B[1m";

    private static final boolean colorEnabled = TerminalUtils.supportsColor();

    private OutputFormatter() {}

    /**
     * Prints an error message in red with ERROR prefix.
     * @param message the error message
     */
    public static void error(String message) {
        if (colorEnabled) {
            System.out.println(RED + BOLD + "ERROR: " + RESET + RED + message + RESET);
        } else {
            System.out.println("ERROR: " + message);
        }
    }

    /**
     * Prints a success message in green with checkmark.
     * @param message the success message
     */
    public static void success(String message) {
        if (colorEnabled) {
            System.out.println(GREEN + "✓ " + message + RESET);
        } else {
            System.out.println("OK: " + message);
        }
    }

    /**
     * Prints a warning message in yellow with WARNING prefix.
     * @param message the warning message
     */
    public static void warning(String message) {
        if (colorEnabled) {
            System.out.println(YELLOW + "WARNING: " + message + RESET);
        } else {
            System.out.println("WARNING: " + message);
        }
    }

    /**
     * Prints a hint message in blue with Hint prefix.
     * @param message the hint message
     */
    public static void hint(String message) {
        if (colorEnabled) {
            System.out.println(BLUE + "Hint: " + message + RESET);
        } else {
            System.out.println("Hint: " + message);
        }
    }

    /**
     * Prints a command suggestion with bold formatting and indentation.
     * @param command the suggested command
     */
    public static void suggestion(String command) {
        if (colorEnabled) {
            System.out.println("  " + BOLD + command + RESET);
        } else {
            System.out.println("  " + command);
        }
    }

    /**
     * Prints an informational message without special formatting.
     * @param message the info message
     */
    public static void info(String message) {
        System.out.println(message);
    }
}
