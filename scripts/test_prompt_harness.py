#!/usr/bin/env python3
"""
Unit tests for the prompt harness.

Tests the evaluation logic without requiring Ollama to be running.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from prompt_harness import CommitMessageEvaluator, EvaluationResult


def test_parse_commit_message():
    """Test commit message parsing."""
    evaluator = CommitMessageEvaluator()

    # Test with scope
    commit_type, scope, desc = evaluator.parse_commit_message("feat(auth): add SSO support")
    assert commit_type == "feat"
    assert scope == "auth"
    assert desc == "add SSO support"

    # Test without scope
    commit_type, scope, desc = evaluator.parse_commit_message("fix: handle null pointer")
    assert commit_type == "fix"
    assert scope is None
    assert desc == "handle null pointer"

    # Test invalid format
    commit_type, scope, desc = evaluator.parse_commit_message("this is not valid")
    assert commit_type is None
    assert scope is None
    assert desc is None

    print("✓ parse_commit_message tests passed")


def test_check_format_compliance():
    """Test format compliance checking."""
    evaluator = CommitMessageEvaluator()

    # Valid with scope
    is_valid, reason = evaluator.check_format_compliance("feat(auth): add SSO support")
    assert is_valid
    assert reason == "valid_with_scope"

    # Valid without scope
    is_valid, reason = evaluator.check_format_compliance("fix: handle errors")
    assert is_valid
    assert reason == "valid_no_scope"

    # Invalid format
    is_valid, reason = evaluator.check_format_compliance("this is not valid")
    assert not is_valid
    assert reason == "format_mismatch"

    # Invalid type
    is_valid, reason = evaluator.check_format_compliance("invalid: this type doesn't exist")
    assert not is_valid
    assert "invalid_type" in reason

    # Too short description
    is_valid, reason = evaluator.check_format_compliance("fix: ab")
    assert not is_valid
    assert reason == "description_too_short"

    print("✓ check_format_compliance tests passed")


def test_compute_score():
    """Test score computation."""
    evaluator = CommitMessageEvaluator()

    # Perfect match
    generated = "fix(auth): handle null token"
    expected = "fix(auth): handle null token"
    score = evaluator.compute_score(generated, expected, True, True)
    assert score >= 90  # Should be very high

    # Valid format, correct type, different description
    generated = "fix(auth): add error handling"
    expected = "fix(auth): handle null token"
    score = evaluator.compute_score(generated, expected, True, True)
    assert 60 <= score <= 80  # Format + type correct

    # Valid format, wrong type (but similar description, so score includes description similarity)
    generated = "feat(auth): handle null token"
    expected = "fix(auth): handle null token"
    score = evaluator.compute_score(generated, expected, True, False)
    assert 40 <= score <= 80  # Format + description similarity

    # Invalid format
    generated = "this is invalid"
    expected = "fix(auth): handle null token"
    score = evaluator.compute_score(generated, expected, False, False)
    assert score < 40  # No format compliance

    print("✓ compute_score tests passed")


def test_evaluate():
    """Test full evaluation."""
    evaluator = CommitMessageEvaluator()

    result = evaluator.evaluate(
        example_id=1,
        diff_input="diff --git a/auth.py...",
        expected_commit="fix(auth): handle null token",
        generated_commit="fix(auth): add error handling",
        generation_time_ms=1000,
        prompt_tokens=100,
        completion_tokens=20
    )

    assert result.example_id == 1
    assert result.is_valid_format
    assert result.expected_type == "fix"
    assert result.generated_type == "fix"
    assert result.type_correct
    assert result.expected_scope == "auth"
    assert result.generated_scope == "auth"
    assert result.has_scope
    assert result.score >= 60
    assert result.generation_time_ms == 1000

    print("✓ evaluate tests passed")


def test_load_examples():
    """Test loading examples from JSONL."""
    from prompt_harness import PromptHarness
    import json
    import tempfile

    harness = PromptHarness()

    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Write test data
        json.dump({
            "instruction": "Generate commit message",
            "input": "diff --git a/test.py...",
            "output": "test: add unit tests"
        }, f)
        f.write('\n')
        json.dump({
            "instruction": "Generate commit message",
            "input": "diff --git a/docs.md...",
            "output": "docs: update README"
        }, f)
        f.write('\n')
        temp_file = f.name

    try:
        # Load examples
        examples = harness.load_examples(temp_file, num_examples=2)

        assert len(examples) == 2
        assert examples[0]['diff'] == "diff --git a/test.py..."
        assert examples[0]['expected_commit'] == "test: add unit tests"
        assert examples[1]['diff'] == "diff --git a/docs.md..."
        assert examples[1]['expected_commit'] == "docs: update README"

        print("✓ load_examples tests passed")

    finally:
        # Clean up
        Path(temp_file).unlink()


def main():
    """Run all tests."""
    print("Running prompt harness unit tests...")
    print()

    test_parse_commit_message()
    test_check_format_compliance()
    test_compute_score()
    test_evaluate()
    test_load_examples()

    print()
    print("="*60)
    print("All tests passed! ✓")
    print("="*60)
    print()
    print("The prompt harness is ready to use.")
    print("To test with Ollama, make sure it's running and try:")
    print("  ./scripts/prompt_harness.py --builtin baseline --num-examples 5 --verbose")


if __name__ == '__main__':
    main()
