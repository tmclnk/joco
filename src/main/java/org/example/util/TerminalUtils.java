package org.example.util;

/**
 * Utility class for terminal detection and capability checks.
 */
public final class TerminalUtils {
    private TerminalUtils() {}

    /**
     * Checks if the program is running in an interactive terminal.
     * @return true if running in a terminal with console access
     */
    public static boolean isTerminal() {
        return System.console() != null;
    }

    /**
     * Checks if the terminal supports ANSI color codes.
     * Respects the NO_COLOR environment variable convention.
     * @return true if colors should be used
     */
    public static boolean supportsColor() {
        if (!isTerminal()) return false;
        String noColor = System.getenv("NO_COLOR");
        return noColor == null || noColor.isEmpty();
    }
}
