package org.example.harness.prompt;

import org.example.harness.prompt.templates.FewShotTemplate;
import org.example.harness.prompt.templates.MinimalTemplate;
import org.example.harness.prompt.templates.StrictFormatTemplate;
import org.example.harness.prompt.templates.TypeAccuracyTemplate;
import org.example.harness.prompt.templates.VerboseTemplate;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Registry of all available prompt templates.
 */
public class PromptTemplateRegistry {

    private final Map<String, PromptTemplate> templates = new HashMap<>();

    public PromptTemplateRegistry() {
        // Register all available templates
        register(new BasePromptTemplate());
        register(new FewShotTemplate());
        register(new MinimalTemplate());
        register(new VerboseTemplate());
        register(new TypeAccuracyTemplate());
        register(new StrictFormatTemplate());
    }

    /**
     * Registers a prompt template.
     */
    public void register(PromptTemplate template) {
        templates.put(template.getId(), template);
    }

    /**
     * Gets a template by ID.
     */
    public PromptTemplate get(String id) {
        PromptTemplate template = templates.get(id);
        if (template == null) {
            throw new IllegalArgumentException("Unknown template: " + id +
                ". Available: " + templates.keySet());
        }
        return template;
    }

    /**
     * Returns all registered templates.
     */
    public Collection<PromptTemplate> getAll() {
        return templates.values();
    }

    /**
     * Lists all template IDs.
     */
    public Collection<String> listIds() {
        return templates.keySet();
    }
}
