#!/usr/bin/env python3
"""
Test Harness for LLM Prompt Iteration

This script provides a comprehensive test harness for iterating on prompts for
commit message generation. It loads test examples, runs the LLM with different
prompts and temperatures, evaluates outputs, and generates comparison reports.

Features:
- Load test examples from JSONL dataset
- Run Ollama models with different prompts and temperatures
- Evaluate format compliance, type accuracy, and quality scores
- Generate detailed comparison reports
- Support multiple prompt variations in a single run

Usage:
    # Test a single prompt
    python prompt_harness.py --prompt my_prompt.txt

    # Test multiple prompts
    python prompt_harness.py --prompts baseline.txt,strict-v1.txt,strict-v2.txt

    # Test with different temperatures
    python prompt_harness.py --prompt strict-v1.txt --temperatures 0.1,0.3,0.5,0.7

    # Use custom dataset
    python prompt_harness.py --prompt my_prompt.txt --dataset dataset/validation.jsonl --num-examples 20
"""

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import requests


@dataclass
class PromptConfig:
    """Configuration for a prompt variation."""
    name: str
    system_prompt: str
    temperature: float
    model: str = "qwen2.5-coder:1.5b"
    max_tokens: int = 100


@dataclass
class EvaluationResult:
    """Result of evaluating a single test example."""
    example_id: int
    diff_input: str
    expected_commit: str
    generated_commit: str
    generation_time_ms: int

    # Format compliance
    is_valid_format: bool
    format_reason: str

    # Type accuracy
    expected_type: str
    generated_type: str
    type_correct: bool

    # Scope analysis
    expected_scope: Optional[str]
    generated_scope: Optional[str]
    has_scope: bool

    # Quality score (0-100)
    score: int

    # Token stats
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass
class PromptEvaluationSummary:
    """Summary statistics for a prompt variation."""
    config: PromptConfig
    results: List[EvaluationResult]

    # Aggregate metrics
    total_examples: int = 0
    format_compliance_rate: float = 0.0
    type_accuracy_rate: float = 0.0
    scope_inclusion_rate: float = 0.0
    avg_score: float = 0.0
    avg_generation_time_ms: float = 0.0
    avg_completion_tokens: float = 0.0

    # Type distribution
    type_distribution: Dict[str, int] = field(default_factory=dict)

    # Failure analysis
    format_failures: List[EvaluationResult] = field(default_factory=list)
    type_failures: List[EvaluationResult] = field(default_factory=list)

    def compute_summary(self):
        """Compute summary statistics from results."""
        self.total_examples = len(self.results)

        if self.total_examples == 0:
            return

        # Format compliance
        format_valid = sum(1 for r in self.results if r.is_valid_format)
        self.format_compliance_rate = format_valid / self.total_examples

        # Type accuracy
        type_correct = sum(1 for r in self.results if r.type_correct)
        self.type_accuracy_rate = type_correct / self.total_examples

        # Scope inclusion
        has_scope = sum(1 for r in self.results if r.has_scope)
        self.scope_inclusion_rate = has_scope / self.total_examples

        # Average score
        self.avg_score = sum(r.score for r in self.results) / self.total_examples

        # Average generation time
        self.avg_generation_time_ms = sum(r.generation_time_ms for r in self.results) / self.total_examples

        # Average completion tokens
        token_counts = [r.completion_tokens for r in self.results if r.completion_tokens is not None]
        if token_counts:
            self.avg_completion_tokens = sum(token_counts) / len(token_counts)

        # Type distribution
        self.type_distribution = Counter(r.generated_type for r in self.results)

        # Failure analysis
        self.format_failures = [r for r in self.results if not r.is_valid_format]
        self.type_failures = [r for r in self.results if not r.type_correct and r.is_valid_format]


class OllamaClient:
    """Client for communicating with Ollama API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        stream: bool = False
    ) -> Dict:
        """
        Generate text using Ollama API.

        Returns dict with:
            - response: generated text
            - prompt_eval_count: number of prompt tokens
            - eval_count: number of completion tokens
            - total_duration: total time in nanoseconds
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        start_time = time.time()
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()

        result = response.json()
        result['_generation_time_ms'] = int((time.time() - start_time) * 1000)

        return result

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False


class CommitMessageEvaluator:
    """Evaluates commit message quality."""

    VALID_TYPES = ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'ci', 'build', 'perf']

    @staticmethod
    def parse_commit_message(commit_msg: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse a commit message into (type, scope, description).

        Returns:
            (type, scope, description) or (None, None, None) if invalid
        """
        if not commit_msg:
            return None, None, None

        # Take first line only
        first_line = commit_msg.strip().split('\n')[0].strip()

        # Pattern with scope: type(scope): description
        match_scope = re.match(r'^([a-z]+)\(([a-z0-9_/-]+)\):\s+(.+)$', first_line, re.IGNORECASE)
        if match_scope:
            return (
                match_scope.group(1).lower(),
                match_scope.group(2).lower(),
                match_scope.group(3)
            )

        # Pattern without scope: type: description
        match_no_scope = re.match(r'^([a-z]+):\s+(.+)$', first_line, re.IGNORECASE)
        if match_no_scope:
            return (
                match_no_scope.group(1).lower(),
                None,
                match_no_scope.group(2)
            )

        return None, None, None

    @classmethod
    def check_format_compliance(cls, commit_msg: str) -> Tuple[bool, str]:
        """
        Check if commit message follows conventional commit format.

        Returns:
            (is_valid, reason)
        """
        commit_type, scope, description = cls.parse_commit_message(commit_msg)

        if commit_type is None:
            return False, "format_mismatch"

        if commit_type not in cls.VALID_TYPES:
            return False, f"invalid_type_{commit_type}"

        if not description or len(description.strip()) < 3:
            return False, "description_too_short"

        return True, "valid_with_scope" if scope else "valid_no_scope"

    @classmethod
    def compute_score(
        cls,
        generated: str,
        expected: str,
        is_valid_format: bool,
        type_correct: bool
    ) -> int:
        """
        Compute quality score (0-100).

        Scoring:
        - Format compliance: 40 points
        - Type accuracy: 30 points
        - Description similarity: 30 points
        """
        score = 0

        # Format compliance (40 points)
        if is_valid_format:
            score += 40

        # Type accuracy (30 points)
        if type_correct:
            score += 30

        # Description similarity (30 points)
        # Simple word overlap metric
        gen_type, gen_scope, gen_desc = cls.parse_commit_message(generated)
        exp_type, exp_scope, exp_desc = cls.parse_commit_message(expected)

        if gen_desc and exp_desc:
            gen_words = set(gen_desc.lower().split())
            exp_words = set(exp_desc.lower().split())

            if gen_words and exp_words:
                overlap = len(gen_words & exp_words)
                union = len(gen_words | exp_words)
                similarity = overlap / union if union > 0 else 0
                score += int(similarity * 30)

        return min(100, score)

    @classmethod
    def evaluate(
        cls,
        example_id: int,
        diff_input: str,
        expected_commit: str,
        generated_commit: str,
        generation_time_ms: int,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None
    ) -> EvaluationResult:
        """Evaluate a single generated commit message."""

        # Check format compliance
        is_valid_format, format_reason = cls.check_format_compliance(generated_commit)

        # Parse both commits
        gen_type, gen_scope, gen_desc = cls.parse_commit_message(generated_commit)
        exp_type, exp_scope, exp_desc = cls.parse_commit_message(expected_commit)

        # Type accuracy
        type_correct = (gen_type == exp_type) if gen_type and exp_type else False

        # Compute score
        score = cls.compute_score(generated_commit, expected_commit, is_valid_format, type_correct)

        return EvaluationResult(
            example_id=example_id,
            diff_input=diff_input,
            expected_commit=expected_commit,
            generated_commit=generated_commit,
            generation_time_ms=generation_time_ms,
            is_valid_format=is_valid_format,
            format_reason=format_reason,
            expected_type=exp_type or "unknown",
            generated_type=gen_type or "unknown",
            type_correct=type_correct,
            expected_scope=exp_scope,
            generated_scope=gen_scope,
            has_scope=gen_scope is not None,
            score=score,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )


class PromptHarness:
    """Main test harness for prompt iteration."""

    def __init__(self, ollama_base_url: str = "http://localhost:11434"):
        self.client = OllamaClient(ollama_base_url)
        self.evaluator = CommitMessageEvaluator()

    def load_examples(self, dataset_path: str, num_examples: int = 10) -> List[Dict]:
        """Load test examples from JSONL dataset."""
        examples = []

        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break

                data = json.loads(line.strip())

                # Extract from messages format or direct format
                if 'messages' in data:
                    # Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
                    user_msg = next((m['content'] for m in data['messages'] if m['role'] == 'user'), '')
                    assistant_msg = next((m['content'] for m in data['messages'] if m['role'] == 'assistant'), '')

                    # Extract diff from user message (usually after "Diff:" or similar)
                    diff_match = re.search(r'(?:Diff:|diff:)(.*)', user_msg, re.DOTALL | re.IGNORECASE)
                    diff = diff_match.group(1).strip() if diff_match else user_msg

                    examples.append({
                        'diff': diff,
                        'expected_commit': assistant_msg.strip()
                    })
                else:
                    # Direct format
                    examples.append({
                        'diff': data.get('input', ''),
                        'expected_commit': data.get('output', '')
                    })

        return examples

    def run_prompt_variation(
        self,
        config: PromptConfig,
        examples: List[Dict],
        verbose: bool = False
    ) -> PromptEvaluationSummary:
        """Run evaluation for a single prompt variation."""

        print(f"\n{'='*80}")
        print(f"Testing: {config.name}")
        print(f"Model: {config.model} | Temperature: {config.temperature}")
        print(f"{'='*80}")

        results = []

        for i, example in enumerate(examples, 1):
            if verbose:
                print(f"Example {i}/{len(examples)}...", end=" ", flush=True)

            # Build prompt
            prompt = config.system_prompt + "\n\nDiff:\n" + example['diff']

            # Generate
            try:
                response = self.client.generate(
                    model=config.model,
                    prompt=prompt,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )

                generated = response['response'].strip()
                generation_time = response['_generation_time_ms']
                prompt_tokens = response.get('prompt_eval_count')
                completion_tokens = response.get('eval_count')

            except Exception as e:
                print(f"ERROR: {e}")
                generated = ""
                generation_time = 0
                prompt_tokens = None
                completion_tokens = None

            # Evaluate
            result = self.evaluator.evaluate(
                example_id=i,
                diff_input=example['diff'],
                expected_commit=example['expected_commit'],
                generated_commit=generated,
                generation_time_ms=generation_time,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )

            results.append(result)

            if verbose:
                status = "✓" if result.is_valid_format else "✗"
                print(f"{status} {result.generated_type} (score: {result.score})")

        # Create summary
        summary = PromptEvaluationSummary(config=config, results=results)
        summary.compute_summary()

        return summary

    def compare_prompts(
        self,
        configs: List[PromptConfig],
        examples: List[Dict],
        verbose: bool = False
    ) -> List[PromptEvaluationSummary]:
        """Run and compare multiple prompt variations."""

        summaries = []

        for config in configs:
            summary = self.run_prompt_variation(config, examples, verbose)
            summaries.append(summary)

        return summaries

    def print_comparison_report(self, summaries: List[PromptEvaluationSummary]):
        """Print a detailed comparison report."""

        print("\n" + "="*80)
        print("PROMPT COMPARISON REPORT")
        print("="*80)

        # Summary table
        print("\n{:<30} {:>10} {:>10} {:>10} {:>10}".format(
            "Prompt", "Format%", "Type%", "Scope%", "Avg Score"
        ))
        print("-" * 80)

        for summary in summaries:
            print("{:<30} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>10.1f}".format(
                summary.config.name[:28],
                summary.format_compliance_rate * 100,
                summary.type_accuracy_rate * 100,
                summary.scope_inclusion_rate * 100,
                summary.avg_score
            ))

        # Detailed metrics for each prompt
        for summary in summaries:
            print(f"\n{'='*80}")
            print(f"DETAILED RESULTS: {summary.config.name}")
            print(f"{'='*80}")

            print(f"\nConfiguration:")
            print(f"  Model: {summary.config.model}")
            print(f"  Temperature: {summary.config.temperature}")
            print(f"  Max tokens: {summary.config.max_tokens}")

            print(f"\nMetrics:")
            print(f"  Total examples: {summary.total_examples}")
            print(f"  Format compliance: {summary.format_compliance_rate*100:.1f}% ({int(summary.format_compliance_rate*summary.total_examples)}/{summary.total_examples})")
            print(f"  Type accuracy: {summary.type_accuracy_rate*100:.1f}% ({int(summary.type_accuracy_rate*summary.total_examples)}/{summary.total_examples})")
            print(f"  Scope inclusion: {summary.scope_inclusion_rate*100:.1f}% ({int(summary.scope_inclusion_rate*summary.total_examples)}/{summary.total_examples})")
            print(f"  Average score: {summary.avg_score:.1f}/100")
            print(f"  Average generation time: {summary.avg_generation_time_ms:.0f}ms")

            if summary.avg_completion_tokens > 0:
                print(f"  Average completion tokens: {summary.avg_completion_tokens:.1f}")

            print(f"\nType distribution:")
            for commit_type, count in sorted(summary.type_distribution.items(), key=lambda x: x[1], reverse=True):
                print(f"  {commit_type}: {count}")

            # Show format failures
            if summary.format_failures:
                print(f"\nFormat failures ({len(summary.format_failures)}):")
                for i, result in enumerate(summary.format_failures[:3], 1):
                    print(f"  {i}. Expected: {result.expected_commit[:60]}")
                    print(f"     Generated: {result.generated_commit[:60]}")
                    print(f"     Reason: {result.format_reason}")

                if len(summary.format_failures) > 3:
                    print(f"  ... and {len(summary.format_failures) - 3} more")

            # Show type mismatches (but valid format)
            if summary.type_failures:
                print(f"\nType mismatches ({len(summary.type_failures)}):")
                for i, result in enumerate(summary.type_failures[:3], 1):
                    print(f"  {i}. Expected: {result.expected_type} | Generated: {result.generated_type}")
                    print(f"     Expected: {result.expected_commit[:60]}")
                    print(f"     Generated: {result.generated_commit[:60]}")

                if len(summary.type_failures) > 3:
                    print(f"  ... and {len(summary.type_failures) - 3} more")

        # Best prompt recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}")

        best = max(summaries, key=lambda s: s.avg_score)
        print(f"\nBest performing prompt: {best.config.name}")
        print(f"  Average score: {best.avg_score:.1f}/100")
        print(f"  Format compliance: {best.format_compliance_rate*100:.1f}%")
        print(f"  Type accuracy: {best.type_accuracy_rate*100:.1f}%")


def load_prompt_from_file(file_path: str) -> str:
    """Load prompt text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_builtin_prompts() -> Dict[str, str]:
    """Create built-in prompt variations for testing."""

    prompts = {}

    # Baseline prompt (current joco prompt)
    prompts['baseline'] = """Task: Output a single-line git commit message. Nothing else.

Format: type(scope): description

Types (choose ONE based on what files changed):
- docs = documentation, README, CHANGELOG, .md files, release notes
- ci = GitHub Actions, CI configs, scripts/, .github/
- build = package.json deps, pom.xml, build configs, tooling updates
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

CRITICAL: Output ONLY the commit message. No explanation. No quotes. One line."""

    # Strict format v1 (from joco-8hs)
    prompts['strict-format-v1'] = """Generate ONLY a commit message in this exact format:
type(scope): description

Valid types: feat, fix, docs, style, refactor, test, chore, ci, build, perf

Rules:
1. ONE line only
2. No quotes, no explanation
3. Lowercase description
4. Scope is optional

Examples:
build(deps): bump eslint to 8.50
fix(auth): handle null token
docs: update README installation steps

Output the commit message now:"""

    # Minimal prompt
    prompts['minimal'] = """Output a conventional commit message (type: description or type(scope): description) for this diff.

Valid types: feat, fix, docs, style, refactor, test, chore, ci, build, perf

ONE line only. No explanation."""

    # Verbose prompt with more guidance
    prompts['verbose'] = """You are a commit message generator. Analyze the git diff and create a conventional commit message.

CONVENTIONAL COMMIT FORMAT:
type(scope): description

COMMIT TYPES:
- feat: A new feature or capability
- fix: A bug fix
- docs: Documentation changes only
- style: Code style/formatting (no logic changes)
- refactor: Code restructuring (no feature or fix)
- test: Adding or updating tests
- chore: Maintenance tasks, configs
- ci: CI/CD changes (GitHub Actions, etc.)
- build: Build system, dependencies (package.json, pom.xml)
- perf: Performance improvements

SCOPE (optional):
- Short module/component name (1 word)
- Examples: auth, http, router, compiler

DESCRIPTION:
- Lowercase, imperative mood
- Start with verb: add, fix, update, remove
- Be concise and specific

CRITICAL INSTRUCTIONS:
- Output ONLY the commit message
- ONE line only
- NO explanations or quotes
- NO markdown formatting

Generate the commit message:"""

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Test harness for LLM prompt iteration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Dataset options
    parser.add_argument(
        '--dataset',
        default='dataset/validation.jsonl',
        help='Path to JSONL dataset file (default: dataset/validation.jsonl)'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=10,
        help='Number of examples to test (default: 10)'
    )

    # Prompt options
    parser.add_argument(
        '--prompt',
        help='Path to prompt file to test'
    )
    parser.add_argument(
        '--prompts',
        help='Comma-separated paths to multiple prompt files'
    )
    parser.add_argument(
        '--builtin',
        choices=['baseline', 'strict-format-v1', 'minimal', 'verbose', 'all'],
        help='Use a built-in prompt variation'
    )
    parser.add_argument(
        '--prompt-name',
        help='Name for the prompt (used in reports)'
    )

    # Model options
    parser.add_argument(
        '--model',
        default='qwen2.5-coder:1.5b',
        help='Ollama model to use (default: qwen2.5-coder:1.5b)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for generation (default: 0.3)'
    )
    parser.add_argument(
        '--temperatures',
        help='Comma-separated temperatures to test (e.g., 0.1,0.3,0.5)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=100,
        help='Maximum tokens to generate (default: 100)'
    )

    # Other options
    parser.add_argument(
        '--ollama-url',
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress for each example'
    )

    args = parser.parse_args()

    # Initialize harness
    harness = PromptHarness(args.ollama_url)

    # Check Ollama availability
    if not harness.client.is_available():
        print(f"ERROR: Cannot connect to Ollama at {args.ollama_url}")
        print("Make sure Ollama is running: ollama serve")
        return 1

    print(f"Connected to Ollama at {args.ollama_url}")

    # Load examples
    print(f"Loading {args.num_examples} examples from {args.dataset}...")
    examples = harness.load_examples(args.dataset, args.num_examples)
    print(f"Loaded {len(examples)} examples")

    # Build prompt configurations
    configs = []
    builtin_prompts = create_builtin_prompts()

    if args.builtin:
        # Use built-in prompts
        if args.builtin == 'all':
            for name, prompt_text in builtin_prompts.items():
                configs.append(PromptConfig(
                    name=name,
                    system_prompt=prompt_text,
                    temperature=args.temperature,
                    model=args.model,
                    max_tokens=args.max_tokens
                ))
        else:
            prompt_text = builtin_prompts[args.builtin]
            configs.append(PromptConfig(
                name=args.builtin,
                system_prompt=prompt_text,
                temperature=args.temperature,
                model=args.model,
                max_tokens=args.max_tokens
            ))

    elif args.prompts:
        # Multiple prompt files
        for prompt_path in args.prompts.split(','):
            prompt_path = prompt_path.strip()
            prompt_text = load_prompt_from_file(prompt_path)
            name = args.prompt_name or Path(prompt_path).stem

            configs.append(PromptConfig(
                name=name,
                system_prompt=prompt_text,
                temperature=args.temperature,
                model=args.model,
                max_tokens=args.max_tokens
            ))

    elif args.prompt:
        # Single prompt file
        prompt_text = load_prompt_from_file(args.prompt)
        name = args.prompt_name or Path(args.prompt).stem

        # Test with multiple temperatures if specified
        if args.temperatures:
            for temp in args.temperatures.split(','):
                temp = float(temp.strip())
                configs.append(PromptConfig(
                    name=f"{name}_t{temp}",
                    system_prompt=prompt_text,
                    temperature=temp,
                    model=args.model,
                    max_tokens=args.max_tokens
                ))
        else:
            configs.append(PromptConfig(
                name=name,
                system_prompt=prompt_text,
                temperature=args.temperature,
                model=args.model,
                max_tokens=args.max_tokens
            ))

    else:
        # Default: test baseline prompt
        configs.append(PromptConfig(
            name='baseline',
            system_prompt=builtin_prompts['baseline'],
            temperature=args.temperature,
            model=args.model,
            max_tokens=args.max_tokens
        ))

    # Run evaluations
    summaries = harness.compare_prompts(configs, examples, args.verbose)

    # Print report
    harness.print_comparison_report(summaries)

    return 0


if __name__ == '__main__':
    exit(main())
