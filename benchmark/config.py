"""
Benchmark Configuration for Path Coverage Experiments

This module contains all configuration settings for the benchmarking pipeline
comparing Zero-shot, AutoExe, and Agentic-Z3 approaches.
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_ROOT = Path(__file__).parent

# TestEval paths
TESTEVAL_ROOT = PROJECT_ROOT / "TestEval"
TESTEVAL_DATA = TESTEVAL_ROOT / "data"
LEETCODE_DATA = TESTEVAL_DATA / "leetcode-py.jsonl"
LEETCODE_INSTRUMENTED = TESTEVAL_DATA / "leetcode-py-instrumented.jsonl"
TARGET_PATHS_DATA = TESTEVAL_DATA / "tgt_paths.jsonl"
TESTEVAL_PROMPTS = TESTEVAL_ROOT / "prompt"

# AutoExe paths
AUTOEXE_ROOT = PROJECT_ROOT / "AutoExe-Artifact"
AUTOEXE_EXECUTOR = AUTOEXE_ROOT / "executables" / "executor-py"

# Agentic-Z3 paths
AGENTIC_Z3_ROOT = PROJECT_ROOT / "agentic_z3"

# Benchmark output paths
RESULTS_DIR = BENCHMARK_ROOT / "results"
SELECTED_TASKS_FILE = BENCHMARK_ROOT / "selected_tasks.json"

# Task selection configuration
NUM_MEDIUM_TASKS = 0   # difficulty == 2 (disabled by default)
NUM_HARD_TASKS = 15    # difficulty == 3 (hard-only benchmark)
RANDOM_SEED = 42

# Difficulty mapping (LeetCode style)
DIFFICULTY_EASY = 1
DIFFICULTY_MEDIUM = 2
DIFFICULTY_HARD = 3

# LLM Configuration
DEFAULT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 512

# OpenAI API (from environment)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Evaluation settings
EXECUTION_TIMEOUT = 5  # seconds per test
MAX_RETRIES = 3

# Output file names
def get_output_filename(approach: str, model: str = DEFAULT_MODEL) -> str:
    """Generate output filename for a given approach and model."""
    safe_model = model.replace("/", "-").replace(":", "-")
    return f"pathcov_{approach}_{safe_model}.jsonl"


# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


