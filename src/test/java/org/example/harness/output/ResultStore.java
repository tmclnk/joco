package org.example.harness.output;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.example.harness.evaluation.EvaluationMetrics;
import org.example.harness.evaluation.EvaluationResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Persists test results to disk.
 */
public class ResultStore {

    private static final Logger logger = LoggerFactory.getLogger(ResultStore.class);
    private static final Path RESULTS_DIR = Path.of("test-results");

    private final Gson gson;
    private final Gson compactGson;

    public ResultStore() {
        this.gson = new GsonBuilder().setPrettyPrinting().create();
        this.compactGson = new Gson();
    }

    /**
     * Saves test run results to a timestamped directory.
     *
     * @return Path to the output directory
     */
    public Path saveResults(String runId, List<EvaluationResult> results, EvaluationMetrics metrics)
            throws IOException {
        Path outputDir = RESULTS_DIR.resolve(runId);
        Files.createDirectories(outputDir);

        // Save individual results as JSONL
        Path resultsFile = outputDir.resolve("results.jsonl");
        List<String> lines = new ArrayList<>();
        for (EvaluationResult result : results) {
            lines.add(compactGson.toJson(result));
        }
        Files.write(resultsFile, lines);
        logger.info("Saved {} results to {}", results.size(), resultsFile);

        // Save aggregated metrics as JSON
        Path metricsFile = outputDir.resolve("metrics.json");
        Files.writeString(metricsFile, gson.toJson(metrics));
        logger.info("Saved metrics to {}", metricsFile);

        // Save summary text
        Path summaryFile = outputDir.resolve("summary.txt");
        Files.writeString(summaryFile, metrics.toSummary());

        return outputDir;
    }

    /**
     * Saves the prompt template used for a test run.
     */
    public void savePromptTemplate(String runId, String templateId, String templateDescription,
                                   String templateText) throws IOException {
        Path outputDir = RESULTS_DIR.resolve(runId);
        Files.createDirectories(outputDir);

        Path promptFile = outputDir.resolve("prompt-template.md");
        String content = String.format("""
            # Prompt Template: %s

            **Description:** %s

            ## Template Text

            ```
            %s
            ```
            """, templateId, templateDescription, templateText);

        Files.writeString(promptFile, content);
        logger.info("Saved prompt template to {}", promptFile);
    }

    /**
     * Loads metrics from a previous run.
     */
    public EvaluationMetrics loadMetrics(String runId) throws IOException {
        Path metricsFile = RESULTS_DIR.resolve(runId).resolve("metrics.json");
        if (!Files.exists(metricsFile)) {
            throw new IOException("Metrics file not found: " + metricsFile);
        }
        return gson.fromJson(Files.readString(metricsFile), EvaluationMetrics.class);
    }

    /**
     * Lists all available run IDs.
     */
    public List<String> listRuns() throws IOException {
        if (!Files.isDirectory(RESULTS_DIR)) {
            return List.of();
        }

        List<String> runs = new ArrayList<>();
        try (var stream = Files.list(RESULTS_DIR)) {
            stream.filter(Files::isDirectory)
                .map(p -> p.getFileName().toString())
                .forEach(runs::add);
        }
        return runs;
    }
}
