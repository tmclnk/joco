package org.example.harness.prompt.templates;

import org.example.harness.prompt.PromptTemplate;

/**
 * Verbose prompt template with detailed instructions.
 * Tests whether more guidance improves output quality.
 */
public class VerboseTemplate implements PromptTemplate {

    private static final String TEMPLATE = """
        You are an expert software developer writing a git commit message.

        TASK: Analyze the following git diff and write a single-line commit message.

        COMMIT MESSAGE FORMAT (Conventional Commits specification):
        - Structure: type(scope): description
        - The type MUST be one of:
          * feat: A new feature
          * fix: A bug fix
          * docs: Documentation only changes
          * style: Changes that do not affect the meaning of the code (white-space, formatting)
          * refactor: A code change that neither fixes a bug nor adds a feature
          * test: Adding missing tests or correcting existing tests
          * chore: Changes to the build process or auxiliary tools
          * perf: A code change that improves performance
        - The scope is optional but recommended - it should be the module/component affected
        - The description MUST:
          * Use imperative mood ("add" not "added" or "adds")
          * Start with lowercase
          * Not end with a period
          * Be under 50 characters

        INSTRUCTIONS:
        1. Read the diff carefully
        2. Identify what changed (files, functions, logic)
        3. Determine the type of change
        4. Identify the scope (module or component)
        5. Write a concise description of WHY the change was made

        OUTPUT: Return ONLY the commit message, nothing else.

        GIT DIFF:
        %s

        COMMIT MESSAGE:""";

    @Override
    public String getId() {
        return "verbose-v1";
    }

    @Override
    public String getDescription() {
        return "Detailed instructions with full conventional commit spec explanation";
    }

    @Override
    public String generatePrompt(String diff) {
        return TEMPLATE.formatted(diff.trim());
    }

    @Override
    public String getTemplateText() {
        return TEMPLATE;
    }
}
