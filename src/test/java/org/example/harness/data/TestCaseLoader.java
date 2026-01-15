package org.example.harness.data;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Loads and saves test cases in JSONL format.
 */
public class TestCaseLoader {

    private final Gson gson;

    public TestCaseLoader() {
        this.gson = new GsonBuilder()
            .setPrettyPrinting()
            .create();
    }

    /**
     * Loads test cases from a JSONL file.
     * Each line is a JSON object representing a TestCase.
     */
    public List<TestCase> loadFromFile(Path jsonlFile) throws IOException {
        if (!Files.exists(jsonlFile)) {
            throw new IOException("Test cases file not found: " + jsonlFile);
        }

        List<TestCase> testCases = new ArrayList<>();
        List<String> lines = Files.readAllLines(jsonlFile);

        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i).trim();
            if (line.isEmpty()) {
                continue;
            }

            try {
                TestCaseDto dto = gson.fromJson(line, TestCaseDto.class);
                testCases.add(dto.toTestCase());
            } catch (Exception e) {
                throw new IOException("Failed to parse line " + (i + 1) + ": " + e.getMessage(), e);
            }
        }

        return testCases;
    }

    /**
     * Saves test cases to a JSONL file.
     */
    public void saveToFile(List<TestCase> testCases, Path outputFile) throws IOException {
        Files.createDirectories(outputFile.getParent());

        List<String> lines = new ArrayList<>();
        Gson compactGson = new Gson(); // No pretty printing for JSONL

        for (TestCase tc : testCases) {
            TestCaseDto dto = TestCaseDto.fromTestCase(tc);
            lines.add(compactGson.toJson(dto));
        }

        Files.write(outputFile, lines);
    }

    /**
     * DTO for JSON serialization of TestCase.
     */
    private static class TestCaseDto {
        String id;
        String diff;
        String expectedMessage;
        String repository;
        String commitHash;
        Map<String, String> metadata;

        TestCase toTestCase() {
            return new TestCase(
                id,
                diff,
                expectedMessage,
                repository,
                commitHash,
                metadata != null ? metadata : Map.of()
            );
        }

        static TestCaseDto fromTestCase(TestCase tc) {
            TestCaseDto dto = new TestCaseDto();
            dto.id = tc.id();
            dto.diff = tc.diff();
            dto.expectedMessage = tc.expectedMessage();
            dto.repository = tc.repository();
            dto.commitHash = tc.commitHash();
            dto.metadata = tc.metadata();
            return dto;
        }
    }
}
