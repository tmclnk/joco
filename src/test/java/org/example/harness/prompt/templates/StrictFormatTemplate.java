package org.example.harness.prompt.templates;

import org.example.harness.prompt.PromptTemplate;

/**
 * Strict format template that constrains output to single-line commit message only.
 * Designed to prevent verbose explanations and enforce format compliance.
 */
public class StrictFormatTemplate implements PromptTemplate {

    private static final String TEMPLATE = """
        Task: Output a single-line git commit message. Nothing else.

        Format: type(scope): description

        Types (choose ONE based on what files changed):
        - docs = documentation, README, CHANGELOG, .md files, release notes
        - ci = GitHub Actions, CI configs, scripts/, .github/
        - build = package.json deps, pom.xml, tooling, version updates
        - chore = configs, maintenance, non-code
        - test = test files
        - refactor = restructuring code
        - fix = bug fix
        - feat = new feature (only if truly new functionality)

        Scope rules:
        - Use short module name (1 word): core, http, compiler, forms, router
        - NOT file paths
        - Optional - omit if unclear

        Examples:
        docs: update release notes for v2.1.0
        ci: update node version in GitHub Actions
        build: update typescript to 5.0
        build(deps): bump eslint version
        chore: remove unused config
        fix(http): handle timeout errors
        feat(auth): add SSO support

        CRITICAL: Output ONLY the commit message. No explanation. No quotes. One line.

        Diff:
        %s

        Commit:""";

    @Override
    public String getId() {
        return "strict-format-v1";
    }

    @Override
    public String getDescription() {
        return "Strict single-line output with short scopes and emphasis on non-feat types";
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
