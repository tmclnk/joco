package org.example.harness.prompt.templates;

import org.example.harness.prompt.PromptTemplate;

/**
 * Prompt template focused on improving type accuracy.
 * Provides explicit decision criteria for choosing commit types.
 */
public class TypeAccuracyTemplate implements PromptTemplate {

    private static final String TEMPLATE = """
        Generate a git commit message in conventional commit format.

        FORMAT: type(scope): description

        IMPORTANT - Choose the correct type based on WHAT changed:

        docs  - README, CHANGELOG, comments, documentation files, release notes
        ci    - CI/CD configs, GitHub Actions, build scripts, renovate configs
        build - package.json dependencies, pom.xml, build configs, tooling updates
        chore - maintenance tasks, config files, non-code changes
        test  - test files only, no production code changes
        refactor - code restructuring without behavior change
        fix   - bug fixes, error corrections
        feat  - NEW functionality or features (use sparingly)

        TYPE DECISION GUIDE:
        - If it touches .md files or docs/ -> docs
        - If it touches .github/, CI configs, scripts/ -> ci
        - If it only updates dependencies or build tools -> build
        - If it's cleanup/maintenance with no new features -> chore
        - If it adds NEW user-facing capability -> feat
        - If it fixes a bug or error -> fix

        RULES:
        - Subject line under 50 chars
        - Lowercase description
        - Imperative mood (add, update, fix)
        - No period at end

        EXAMPLES:
        docs: update changelog for v2.1.0 release
        docs(readme): add installation instructions
        ci: update GitHub Actions to node 20
        ci(renovate): configure dependency update schedule
        build: update dependency typescript to 5.0
        build(deps): bump eslint to 8.50.0
        chore: remove unused configuration files
        test(auth): add unit tests for login flow
        refactor(api): simplify error handling logic
        fix(parser): handle null input correctly
        feat(auth): add OAuth2 support

        OUTPUT: Return ONLY the commit message, nothing else.

        DIFF:
        %s

        COMMIT MESSAGE:""";

    @Override
    public String getId() {
        return "type-accuracy-v1";
    }

    @Override
    public String getDescription() {
        return "Focused on type accuracy with explicit decision criteria and more non-feat examples";
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
