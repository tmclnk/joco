package org.example.harness.dataset;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.example.harness.data.TestCase;
import org.example.harness.data.TestCaseLoader;
import org.example.harness.evaluation.StructuralValidator;
import org.example.harness.evaluation.StructuralValidator.ValidationResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Curates a high-quality dataset from test cases for HuggingFace finetuning.
 *
 * Produces instruction-tuning format compatible with:
 * - HuggingFace datasets library
 * - Axolotl
 * - Unsloth
 * - LLaMA-Factory
 */
public class DatasetCurator {

    private static final Logger logger = LoggerFactory.getLogger(DatasetCurator.class);

    private final StructuralValidator validator = new StructuralValidator();
    private final Gson gson = new Gson();
    private final Random random = new Random(42); // Fixed seed for reproducibility

    // Quality thresholds
    private int minDiffLength = 50;
    private int maxDiffLength = 8000;
    private int minValidationScore = 70;
    private double validationSplitRatio = 0.1;

    /**
     * Dataset record in instruction-tuning format.
     */
    public record DatasetRecord(
        String instruction,
        String input,
        String output,
        Map<String, Object> metadata
    ) {}

    /**
     * Statistics about the curated dataset.
     */
    public record DatasetStats(
        int totalSourceRecords,
        int filteredRecords,
        int trainRecords,
        int validationRecords,
        Map<String, Integer> typeDistribution,
        Map<String, Integer> sourceDistribution,
        int avgDiffLength,
        int avgOutputLength
    ) {}

    public DatasetCurator setMinDiffLength(int minDiffLength) {
        this.minDiffLength = minDiffLength;
        return this;
    }

    public DatasetCurator setMaxDiffLength(int maxDiffLength) {
        this.maxDiffLength = maxDiffLength;
        return this;
    }

    public DatasetCurator setMinValidationScore(int minValidationScore) {
        this.minValidationScore = minValidationScore;
        return this;
    }

    public DatasetCurator setValidationSplitRatio(double ratio) {
        this.validationSplitRatio = ratio;
        return this;
    }

    /**
     * Curates dataset from multiple JSONL source files.
     *
     * @param sourceFiles List of JSONL files containing test cases
     * @param outputDir Directory to write train.jsonl and validation.jsonl
     * @return Statistics about the curated dataset
     */
    public DatasetStats curate(List<Path> sourceFiles, Path outputDir) throws IOException {
        logger.info("Loading test cases from {} source files", sourceFiles.size());

        // Load all test cases
        TestCaseLoader loader = new TestCaseLoader();
        List<TestCase> allTestCases = new ArrayList<>();

        for (Path sourceFile : sourceFiles) {
            if (Files.exists(sourceFile)) {
                try {
                    List<TestCase> cases = loader.loadFromFile(sourceFile);
                    allTestCases.addAll(cases);
                    logger.info("Loaded {} test cases from {}", cases.size(), sourceFile.getFileName());
                } catch (Exception e) {
                    logger.warn("Failed to load {}: {}", sourceFile, e.getMessage());
                }
            }
        }

        logger.info("Total loaded: {} test cases", allTestCases.size());

        // Filter for quality
        List<DatasetRecord> qualityRecords = allTestCases.stream()
            .filter(this::passesQualityFilter)
            .map(this::convertToRecord)
            .collect(Collectors.toList());

        logger.info("After quality filtering: {} records", qualityRecords.size());

        // Deduplicate by output (commit message)
        Map<String, DatasetRecord> deduped = new LinkedHashMap<>();
        for (DatasetRecord record : qualityRecords) {
            String key = record.output().toLowerCase().trim();
            if (!deduped.containsKey(key)) {
                deduped.put(key, record);
            }
        }

        List<DatasetRecord> finalRecords = new ArrayList<>(deduped.values());
        logger.info("After deduplication: {} records", finalRecords.size());

        // Shuffle for random split
        Collections.shuffle(finalRecords, random);

        // Split into train/validation
        int splitIndex = (int) (finalRecords.size() * (1 - validationSplitRatio));
        List<DatasetRecord> trainRecords = finalRecords.subList(0, splitIndex);
        List<DatasetRecord> validationRecords = finalRecords.subList(splitIndex, finalRecords.size());

        // Write output files
        Files.createDirectories(outputDir);
        writeRecords(trainRecords, outputDir.resolve("train.jsonl"));
        writeRecords(validationRecords, outputDir.resolve("validation.jsonl"));

        // Calculate statistics
        DatasetStats stats = calculateStats(allTestCases.size(), finalRecords, trainRecords, validationRecords);

        logger.info("Dataset created: {} train, {} validation",
            trainRecords.size(), validationRecords.size());

        return stats;
    }

    private boolean passesQualityFilter(TestCase tc) {
        // Check diff length
        int diffLength = tc.diff().length();
        if (diffLength < minDiffLength || diffLength > maxDiffLength) {
            return false;
        }

        // Validate commit message structure
        ValidationResult result = validator.validate(tc.expectedMessage());
        if (result.score() < minValidationScore) {
            return false;
        }

        // Must have valid type
        if (!result.hasValidType()) {
            return false;
        }

        // Must have description
        if (!result.hasDescription()) {
            return false;
        }

        // Subject must be within limit
        if (!result.subjectWithinLimit()) {
            return false;
        }

        // Must not end with period
        if (result.endsWithPeriod()) {
            return false;
        }

        return true;
    }

    private DatasetRecord convertToRecord(TestCase tc) {
        String instruction = buildInstruction();
        String input = tc.diff();
        String output = tc.expectedMessage();

        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("id", tc.id());
        if (tc.repository() != null) {
            metadata.put("repository", tc.repository());
        }
        if (tc.commitHash() != null) {
            metadata.put("commit_hash", tc.commitHash());
        }
        if (tc.metadata() != null) {
            metadata.putAll(tc.metadata());
        }

        // Add validation info
        ValidationResult validation = validator.validate(tc.expectedMessage());
        metadata.put("type", validation.type());
        if (validation.scope() != null) {
            metadata.put("scope", validation.scope());
        }
        metadata.put("quality_score", validation.score());

        return new DatasetRecord(instruction, input, output, metadata);
    }

    private String buildInstruction() {
        return """
            Generate a conventional commit message for the following git diff.

            Follow the conventional commits format: type(scope): description

            Valid types: feat, fix, docs, style, refactor, perf, test, build, ci, chore

            Rules:
            - Keep the subject line under 72 characters
            - Start description with lowercase
            - Do not end with a period
            - Be concise and specific""";
    }

    private void writeRecords(List<DatasetRecord> records, Path outputFile) throws IOException {
        List<String> lines = records.stream()
            .map(gson::toJson)
            .collect(Collectors.toList());
        Files.write(outputFile, lines);
        logger.info("Wrote {} records to {}", records.size(), outputFile);
    }

    private DatasetStats calculateStats(int totalSource, List<DatasetRecord> all,
                                        List<DatasetRecord> train, List<DatasetRecord> validation) {
        // Type distribution
        Map<String, Integer> typeDistribution = new TreeMap<>();
        for (DatasetRecord record : all) {
            String type = (String) record.metadata().get("type");
            typeDistribution.merge(type, 1, Integer::sum);
        }

        // Source distribution
        Map<String, Integer> sourceDistribution = new TreeMap<>();
        for (DatasetRecord record : all) {
            String repo = (String) record.metadata().get("repository");
            if (repo != null) {
                // Extract base repo name
                String baseName = repo.contains("-") ? repo.split("-")[0] : repo;
                sourceDistribution.merge(baseName, 1, Integer::sum);
            }
        }

        // Average lengths
        int totalDiffLength = all.stream()
            .mapToInt(r -> r.input().length())
            .sum();
        int totalOutputLength = all.stream()
            .mapToInt(r -> r.output().length())
            .sum();

        int avgDiffLength = all.isEmpty() ? 0 : totalDiffLength / all.size();
        int avgOutputLength = all.isEmpty() ? 0 : totalOutputLength / all.size();

        return new DatasetStats(
            totalSource,
            all.size(),
            train.size(),
            validation.size(),
            typeDistribution,
            sourceDistribution,
            avgDiffLength,
            avgOutputLength
        );
    }

    /**
     * Main entry point for dataset curation.
     */
    public static void main(String[] args) throws IOException {
        Path projectRoot = Path.of(System.getProperty("user.dir"));

        // Find all source JSONL files
        List<Path> sourceFiles = new ArrayList<>();

        // Format correctness (conventional commits from Angular)
        sourceFiles.add(projectRoot.resolve("benchmark/format-correctness/angular-commits.jsonl"));

        // Content quality (various expert authors)
        Path contentQuality = projectRoot.resolve("benchmark/content-quality");
        if (Files.exists(contentQuality)) {
            Files.walk(contentQuality, 2)
                .filter(p -> p.toString().endsWith(".jsonl"))
                .forEach(sourceFiles::add);
        }

        // Test cases
        Path testCases = projectRoot.resolve("test-cases");
        if (Files.exists(testCases)) {
            Files.walk(testCases, 1)
                .filter(p -> p.toString().endsWith(".jsonl"))
                .forEach(sourceFiles::add);
        }

        logger.info("Found {} source files", sourceFiles.size());
        sourceFiles.forEach(f -> logger.info("  - {}", f));

        // Curate dataset
        DatasetCurator curator = new DatasetCurator()
            .setMinDiffLength(50)
            .setMaxDiffLength(8000)
            .setMinValidationScore(70)
            .setValidationSplitRatio(0.1);

        Path outputDir = projectRoot.resolve("dataset");
        DatasetStats stats = curator.curate(sourceFiles, outputDir);

        // Print statistics
        System.out.println("\n=== Dataset Statistics ===");
        System.out.println("Total source records: " + stats.totalSourceRecords());
        System.out.println("After filtering: " + stats.filteredRecords());
        System.out.println("Train records: " + stats.trainRecords());
        System.out.println("Validation records: " + stats.validationRecords());
        System.out.println("Average diff length: " + stats.avgDiffLength() + " chars");
        System.out.println("Average output length: " + stats.avgOutputLength() + " chars");

        System.out.println("\nType distribution:");
        stats.typeDistribution().forEach((type, count) ->
            System.out.println("  " + type + ": " + count));

        System.out.println("\nSource distribution:");
        stats.sourceDistribution().forEach((source, count) ->
            System.out.println("  " + source + ": " + count));

        // Write stats to file
        Gson prettyGson = new GsonBuilder().setPrettyPrinting().create();
        Files.writeString(outputDir.resolve("stats.json"), prettyGson.toJson(stats));

        System.out.println("\nDataset written to: " + outputDir);
    }
}
