# Agentic-Z3

**An Autonomous SMT Solving Framework with Hierarchical Planning, Type-Aware Probing, and Evolutionary Memory**

## Overview

Agentic-Z3 is a multi-agent system that addresses three major pain points of SMT solvers like Z3:

1. **Type Error Sensitivity** → Solved by Type-Aware Interactive Probing
2. **Difficulty in Quantifier Instantiation** → Solved by TTRL with Soft Reset
3. **Lack of Knowledge Reusability** → Solved by Curriculum-Guided Skill Evolution

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Engine (Orchestrator)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Architect  │  │    Worker    │  │    Coach     │          │
│  │   (Planner)  │  │   (Coder)    │  │ (Diagnostician)│        │
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

## Key Mechanisms

### 1. Type-Aware Interactive Probing

Before generating full Z3 code, the Worker creates a minimal "probe" script that verifies variable types are correctly declared. This catches 90% of `TypeError` issues before they cause execution failures.

```python
# Probe script (generated before full code)
from z3 import *
x = Int('x')  # Verify Int declaration works
y = Real('y')  # Verify Real declaration works
solver = Solver()
solver.add(x >= 0)  # Trivial constraint
print(solver.check())  # Quick validation
```

### 2. TTRL with Soft Reset

When Z3 returns `UNKNOWN` (timeout) multiple times, instead of just retrying, the system performs a "soft reset":

- **CLEARS** conversation history (forgets failed strategies)
- **PRESERVES** failure summaries (knows what NOT to do)
- **BOOSTS** temperature (0.2 → 0.7) for exploration diversity

This prevents the LLM from getting stuck generating similar failing code.

### 3. Curriculum-Guided Skill Evolution

Successful solutions are crystallized into parameterized templates:

```python
# Original: solver.assert_and_track(x <= 100, "c_bound")
# Template: solver.assert_and_track(x <= {{UPPER_BOUND}}, "c_bound")
```

These templates are stored in ChromaDB and retrieved for similar future problems.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `z3-solver>=4.12.0` - The SMT solver
- `openai>=1.0.0` - LLM API
- `chromadb>=0.4.0` - Vector database for skill library
- `pydantic>=2.0` - Configuration management

## Usage

```bash
# Direct problem input
python main.py "Find integers x, y where x + y = 10 and x > y"

# From file
python main.py --file problem.txt

# With verbose output
python main.py --verbose "Prove x + y >= x for all non-negative y"

# With custom timeout
python main.py --timeout 10000 "Complex scheduling problem..."
```

## Configuration

Set environment variables or use a `.env` file:

```bash
OPENAI_API_KEY=your-api-key
LLM_MODEL=gpt-4o
Z3_TIMEOUT=5000
MAX_TTRL_RETRIES=5
SOFT_RESET_THRESHOLD=3
```

## Project Structure

```
Agentic-Z3/
├── main.py                 # Entry point
├── config.py               # Configuration (Pydantic Settings)
├── requirements.txt        # Dependencies
└── agentic_z3/
    ├── core/
    │   ├── state.py        # SMTState shared memory
    │   └── engine.py       # Main orchestration loop
    ├── agents/
    │   ├── base_agent.py   # LLM interaction base
    │   ├── architect.py    # Hierarchical planning
    │   ├── worker.py       # Code generation + TTRL
    │   └── coach.py        # Diagnosis + skill extraction
    ├── memory/
    │   ├── skill_library.py # ChromaDB skill storage
    │   └── ttrl_cache.py   # Soft reset context
    ├── tools/
    │   ├── z3_executor.py  # Safe Z3 execution
    │   └── type_checker.py # Static type analysis
    └── utils/
        ├── logger.py       # Structured logging
        └── prompter.py     # Prompt templates
```

## Critical Implementation Details

### assert_and_track Enforcement

All constraints must use `solver.assert_and_track()` instead of `solver.add()` for unsat core tracking to work. The `z3_executor.py` includes a preprocessor that auto-converts if needed:

```python
# LLM generates:
solver.add(x > 0)

# Auto-converted to:
solver.assert_and_track(x > 0, "c_auto_1")
```

### Soft Reset Context Preservation

When soft reset clears conversation history, it carefully reconstructs:

1. System prompt (agent role)
2. Original problem (ground truth)
3. Failure summary (what NOT to do)
4. Instruction to try different strategy

## State Machine Flow

```
INIT → Planning → Probing → Coding → Solving
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
                   SAT              UNSAT            UNKNOWN
                    ↓                 ↓                 ↓
              Crystallize        Diagnose          SoftReset?
                    ↓                 ↓                 ↓
                  DONE           Re-plan?           Retry
                                    ↓
                               ←───────────────────────┘
```

## References

This system integrates concepts from:

- **BFS-Prover-V2**: Soft Reset mechanism for exploration diversity
- **LLM-Sym**: Type inference before code generation
- **LEGO-Prover**: Skill library and template extraction
- **SolSearch**: Curriculum learning for warmup
- **AlphaProof**: Interactive probing strategy

## License

MIT License


