package org.example.harness.prompt.templates;

import org.example.harness.prompt.PromptTemplate;

/**
 * File pattern matching template for improved type selection accuracy.
 *
 * Uses explicit file glob patterns to guide type selection and places
 * output constraints at the top for better format compliance with small models.
 */
public class FilePatternTemplate implements PromptTemplate {

    private static final String TEMPLATE = """
        Complete the commit message. Output ONLY the message after "Commit:".

        Rules:
        1. Format: type: description OR type(scope): description
        2. Types: docs, ci, build, test, chore, feat, fix, refactor
        3. One line, under 72 characters
        4. Start with lowercase after colon

        Type guide:
        - docs = .md files, README, documentation
        - ci = .github/, workflows, GitHub Actions
        - build = package.json, pom.xml, dependencies
        - test = test files
        - feat = new feature
        - fix = bug fix
        - refactor = restructure without behavior change

        Example diffs and commits:

        Diff: README.md changed
        Commit: docs: update readme

        Diff: .github/workflows/ci.yml changed
        Commit: ci: update workflow

        Diff: package.json dependency updated
        Commit: build(deps): update dependency

        Diff: src/auth.ts error handling added
        Commit: fix(auth): handle errors

        Now your turn. Output ONLY the commit message, nothing else.

        Diff:
        %s

        Commit:""";

    @Override
    public String getId() {
        return "file-pattern-v1";
    }

    @Override
    public String getDescription() {
        return "File pattern matching for type selection with terse few-shot examples";
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
