package org.example.harness.prompt.templates;

import org.example.harness.prompt.PromptTemplate;

/**
 * Minimal prompt template with very few tokens.
 * Tests how well the model performs with minimal guidance.
 */
public class MinimalTemplate implements PromptTemplate {

    private static final String TEMPLATE = """
        Write a conventional commit message (type(scope): description) for:

        %s

        Commit:""";

    @Override
    public String getId() {
        return "minimal-v1";
    }

    @Override
    public String getDescription() {
        return "Ultra-minimal prompt to test model's built-in knowledge";
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
