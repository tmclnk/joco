package org.example.harness.prompt;

import org.example.ollama.CommitMessagePrompt;

/**
 * Baseline prompt template that wraps the production CommitMessagePrompt.
 * Used as the reference point for comparing other templates.
 */
public class BasePromptTemplate implements PromptTemplate {

    @Override
    public String getId() {
        return "baseline-v1";
    }

    @Override
    public String getDescription() {
        return "Current joco production prompt - CommitMessagePrompt.createCompletePrompt()";
    }

    @Override
    public String generatePrompt(String diff) {
        return CommitMessagePrompt.createCompletePrompt(diff);
    }

    @Override
    public String getTemplateText() {
        return CommitMessagePrompt.getSystemPrompt();
    }
}
