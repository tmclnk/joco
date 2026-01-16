package org.example.harness.prompt.templates;

import org.example.harness.prompt.PromptTemplate;

/**
 * Passthrough template that returns the raw diff without any formatting.
 * Used by multi-step generators that handle their own prompting.
 */
public class RawDiffTemplate implements PromptTemplate {

    @Override
    public String getId() {
        return "raw-diff";
    }

    @Override
    public String getDescription() {
        return "Passthrough template for generators with built-in prompts (e.g., multi-step)";
    }

    @Override
    public String generatePrompt(String diff) {
        return diff;  // Just return the raw diff
    }

    @Override
    public String getTemplateText() {
        return "%s";  // Identity template
    }
}
