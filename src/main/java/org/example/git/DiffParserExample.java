package org.example.git;

import java.util.List;

/**
 * Example usage of DiffParser and DiffFormatter.
 * This class demonstrates how to use the diff parsing and formatting functionality.
 */
public class DiffParserExample {

    public static void main(String[] args) {
        // Example git diff output
        String sampleDiff = """
            diff --git a/src/Main.java b/src/Main.java
            index 1234567..89abcdef 100644
            --- a/src/Main.java
            +++ b/src/Main.java
            @@ -1,5 +1,6 @@
             public class Main {
                 public static void main(String[] args) {
            +        System.out.println("Hello World");
                     // TODO: implement
                 }
             }
            diff --git a/README.md b/README.md
            new file mode 100644
            index 0000000..1234567
            --- /dev/null
            +++ b/README.md
            @@ -0,0 +1,3 @@
            +# My Project
            +
            +This is a sample project.
            diff --git a/old.txt b/new.txt
            similarity index 100%
            rename from old.txt
            rename to new.txt
            diff --git a/image.png b/image.png
            Binary files differ
            """;

        // Parse the diff
        DiffParser parser = new DiffParser();
        List<DiffParser.FileChange> changes = parser.parse(sampleDiff);

        System.out.println("Parsed " + changes.size() + " file changes:");
        for (DiffParser.FileChange change : changes) {
            System.out.println("  - " + change);
        }

        System.out.println("\n" + "=".repeat(70) + "\n");

        // Format the diff for LLM consumption
        DiffFormatter formatter = new DiffFormatter();
        String formatted = formatter.format(changes);

        System.out.println("Formatted output:");
        System.out.println(formatted);

        // Check size
        int estimatedTokens = DiffFormatter.estimateTokens(formatted);
        System.out.println("\nEstimated tokens: " + estimatedTokens);
    }
}
