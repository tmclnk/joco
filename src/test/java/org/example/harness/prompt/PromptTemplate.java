package org.example.harness.prompt;

/**
 * Interface for prompt templates used in commit message generation.
 * Each template represents a different prompting strategy.
 */
public interface PromptTemplate {

    /**
     * Unique identifier for this template.
     */
    String getId();

    /**
     * Human-readable description of this template.
     */
    String getDescription();

    /**
     * Generates the complete prompt from a git diff.
     *
     * @param diff The git diff content
     * @return The formatted prompt to send to the LLM
     */
    String generatePrompt(String diff);

    /**
     * Returns the raw template text for inspection.
     */
    String getTemplateText();
}
