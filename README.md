# Agentic-Z3

**An Autonomous SMT Solving Framework with Hierarchical Planning, Type-Aware Probing, and Evolutionary Memory**

## Overview

Agentic-Z3 is a multi-agent framework that uses an LLM + Z3 to solve natural-language SMT-style problems, and also includes a **path-coverage benchmark pipeline** (TestEval) to compare multiple approaches.

Agentic-Z3 targets three recurring failure modes:

1. **Type error sensitivity** → mitigated via **type-aware interactive probing**
2. **Stagnation on timeouts / UNKNOWN** → mitigated via **TTRL + soft reset**
3. **Lack of reuse** → mitigated via **skill crystallization** into a **ChromaDB-backed skill library**

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Engine (Orchestrator)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Architect  │  │    Worker    │  │    Coach     │          │
│  │   (Planner)  │  │   (Coder)    │  │ (Diagnose)   │          │
│  │              │  │              │  │              │          │
│  │ • Blueprint  │  │ • Type Probe │  │ • Unsat Core │          │
│  │ • Curriculum │  │ • Code Gen   │  │   Analysis   │          │
│  │              │  │ • Soft Reset │  │ • Skill      │          │
│  │              │  │              │  │   Extraction │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Skill Library│  │  TTRL Cache  │  │ Z3 Executor  │          │
│  │  (ChromaDB)  │  │ (Ephemeral)  │  │ (Subprocess) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
python -m pip install -r requirements.txt
```

### Environment

Agentic-Z3 and the benchmark runners use the OpenAI API by default.

```bash
export OPENAI_API_KEY="your-key"
```

You can also configure via a `.env` file (see **Configuration**).

## Quickstart (solve a single problem)

```bash
# Direct problem input
python main.py "Find integers x, y where x + y = 10 and x > y"

# From file
python main.py --file problem.txt

# From stdin
cat problem.txt | python main.py --stdin
```

### Useful CLI flags

```bash
# Show verbose agent logs (also bumps log level to DEBUG)
python main.py --verbose "Prove x + y >= x for all non-negative y"

# Override Z3 timeout and retry budget
python main.py --timeout 10000 --max-retries 5 "Complex scheduling problem..."

# JSON logs (helpful for collecting runs)
python main.py --json-output "..."

# Disable curriculum warmup (useful for speed / benchmarking)
python main.py --no-warmup "..."
```

## Configuration

Configuration is defined in `config.py` (Pydantic Settings). Values can be set via environment variables or a `.env` file.

Common settings:

```bash
OPENAI_API_KEY=...
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=4096

Z3_TIMEOUT=5000
Z3_PROBE_TIMEOUT=1000
MAX_TTRL_RETRIES=5
SOFT_RESET_THRESHOLD=3

CHROMA_PERSIST_PATH=./.agentic_z3_data/chroma_db
SKILL_RETRIEVAL_TOP_K=3

ENABLE_CURRICULUM_WARMUP=true
MAX_PLANNING_ITERATIONS=3
```

## Benchmark: Path Coverage (TestEval)

This repo includes a unified pipeline under `benchmark/` to run and compare:

- **Zero-shot**: direct prompting baseline (`benchmark/runners/zero_shot_runner.py`)
- **AutoExe**: LLM-powered symbolic execution (`benchmark/runners/autoexe_runner.py`)
- **Agentic-Z3**: SMT-based solving (full engine or direct Z3) (`benchmark/runners/agentic_z3_runner.py`)
- **Vanilla Z3**: single “text-to-Z3 script + execute” baseline (`benchmark/runners/vanilla_z3_runner.py`)

### Run the whole pipeline

```bash
# Run task selection (if needed), all approaches, then evaluate
python benchmark/run_benchmark.py --all
```

### Run specific approaches

```bash
python benchmark/run_benchmark.py --run zero_shot
python benchmark/run_benchmark.py --run autoexe

# Agentic-Z3 modes:
# - engine: full multi-agent loop (Architect/Worker/Coach)
# - direct: direct Z3 solving without the multi-agent engine
# - llm: LLM-only SMT-style reasoning baseline
python benchmark/run_benchmark.py --run agentic_z3 --agentic-mode engine
python benchmark/run_benchmark.py --run agentic_z3 --agentic-mode direct
python benchmark/run_benchmark.py --run agentic_z3 --agentic-mode llm

python benchmark/run_benchmark.py --run vanilla_z3
```

### Evaluate existing results

```bash
python benchmark/run_benchmark.py --evaluate

# Or evaluate with more control
python benchmark/evaluate.py --list
python benchmark/evaluate.py --approaches agentic_z3_engine vanilla_z3
python benchmark/evaluate.py --no-errors
```

### Results

- Results are written to `benchmark/results/*.jsonl`
- Evaluation summaries are written to `benchmark/results/summary_*.json`
- Task selection is stored at `benchmark/selected_tasks.json` (recreate with `python benchmark/run_benchmark.py --select-tasks`)

### Rate limiting (important for benchmarking)

Benchmark runners use `benchmark/rate_limiter.py` for cross-runner throttling and 429 retries.
If you run runners directly, you can choose an API tier and adjust concurrency:

```bash
python benchmark/runners/zero_shot_runner.py --rate-limit-tier tier1 --max-workers 2
python benchmark/runners/vanilla_z3_runner.py --rate-limit-tier tier1 --max-workers 2
python benchmark/runners/agentic_z3_runner.py --mode engine --rate-limit-tier tier1 --max-workers 2
```

## Project Structure

```
agentic_z3/
  agents/                # Architect / Worker / Coach
  core/                  # Engine state machine + shared state
  memory/                # Chroma skill library + TTRL cache
  tools/                 # Z3 execution, type checking
  utils/                 # logging, prompting

benchmark/
  run_benchmark.py       # Orchestrates selection + runs + evaluation
  evaluate.py            # Unified evaluator for generated tests
  task_selector.py       # Reproducible selection from TestEval
  rate_limiter.py        # Thread-safe TPM/RPM limiter with retries
  adapters/              # Convert TestEval paths → SMT / AutoExe formats
  runners/               # zero-shot / autoexe / agentic_z3 / vanilla_z3
  results/               # Output JSONL + summary JSON

TestEval/                # Dataset + prompt templates + evaluator utilities
AutoExe-Artifact/        # AutoExe executables/artifact (optional for AutoExe runs)
```

## Notes / Implementation Details

### Unsat-core tracking

Agentic-Z3’s internal Z3 execution path is designed to support unsat-core-driven diagnosis.
Some baselines (e.g., Vanilla Z3) intentionally avoid any preprocessing that would rewrite
LLM-generated solver code.

### Soft reset behavior

Soft reset clears the agent conversation history to force strategy change, while preserving failure summaries.

## License

MIT License