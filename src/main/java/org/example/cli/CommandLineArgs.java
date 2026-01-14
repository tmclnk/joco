package org.example.cli;

import java.util.ArrayList;
import java.util.List;

/**
 * Command-line argument parser for joco.
 * Supports flags: --help, --version, --model, --dry-run
 */
public class CommandLineArgs {
    private boolean help = false;
    private boolean version = false;
    private boolean dryRun = false;
    private String modelOverride = null;
    private final List<String> errors = new ArrayList<>();

    /**
     * Parses command-line arguments.
     *
     * @param args command-line arguments from main()
     * @return parsed CommandLineArgs instance
     */
    public static CommandLineArgs parse(String[] args) {
        CommandLineArgs result = new CommandLineArgs();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];

            switch (arg) {
                case "--help", "-h" -> result.help = true;
                case "--version", "-v" -> result.version = true;
                case "--dry-run", "-d" -> result.dryRun = true;
                case "--model", "-m" -> {
                    if (i + 1 < args.length) {
                        result.modelOverride = args[++i];
                    } else {
                        result.errors.add("--model requires a value");
                    }
                }
                default -> {
                    if (arg.startsWith("--model=")) {
                        result.modelOverride = arg.substring("--model=".length());
                        if (result.modelOverride.isEmpty()) {
                            result.errors.add("--model= requires a value");
                        }
                    } else if (arg.startsWith("-")) {
                        result.errors.add("Unknown option: " + arg);
                    } else {
                        result.errors.add("Unexpected argument: " + arg);
                    }
                }
            }
        }

        return result;
    }

    /**
     * Checks if --help flag was provided.
     */
    public boolean isHelp() {
        return help;
    }

    /**
     * Checks if --version flag was provided.
     */
    public boolean isVersion() {
        return version;
    }

    /**
     * Checks if --dry-run flag was provided.
     */
    public boolean isDryRun() {
        return dryRun;
    }

    /**
     * Gets the model override if --model was provided.
     *
     * @return model name or null if not provided
     */
    public String getModelOverride() {
        return modelOverride;
    }

    /**
     * Checks if there were any parsing errors.
     */
    public boolean hasErrors() {
        return !errors.isEmpty();
    }

    /**
     * Gets the list of parsing errors.
     */
    public List<String> getErrors() {
        return new ArrayList<>(errors);
    }

    /**
     * Prints help message to stdout.
     */
    public static void printHelp() {
        System.out.println("joco - AI-powered commit message generator");
        System.out.println();
        System.out.println("USAGE:");
        System.out.println("    joco [OPTIONS]");
        System.out.println();
        System.out.println("OPTIONS:");
        System.out.println("    -h, --help              Show this help message");
        System.out.println("    -v, --version           Show version information");
        System.out.println("    -m, --model <MODEL>     Override the model specified in config");
        System.out.println("    -d, --dry-run           Generate commit message without creating commit");
        System.out.println();
        System.out.println("DESCRIPTION:");
        System.out.println("    joco generates commit messages from staged Git changes using a local");
        System.out.println("    LLM via Ollama. It analyzes your diff and suggests a conventional");
        System.out.println("    commit message following best practices.");
        System.out.println();
        System.out.println("EXAMPLES:");
        System.out.println("    joco                           # Generate and commit with default settings");
        System.out.println("    joco --dry-run                 # Generate message without committing");
        System.out.println("    joco --model qwen2.5-coder:7b  # Use a different model");
        System.out.println("    joco --help                    # Show this help");
        System.out.println();
        System.out.println("CONFIGURATION:");
        System.out.println("    Configuration file: ~/.joco.config");
        System.out.println("    See documentation for configuration options.");
    }

    /**
     * Prints version information to stdout.
     */
    public static void printVersion() {
        System.out.println("joco version 1.0-SNAPSHOT");
        System.out.println("A lightweight AI-powered commit message generator");
    }
}
