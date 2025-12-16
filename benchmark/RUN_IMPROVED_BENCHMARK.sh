#!/bin/bash
# Quick script to run the improved agentic_z3_engine benchmark
#
# Usage: 
#   export OPENAI_API_KEY="your-key"
#   bash benchmark/RUN_IMPROVED_BENCHMARK.sh

set -e

echo "======================================================================="
echo "Running Improved agentic_z3_engine Benchmark"
echo "======================================================================="
echo

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set"
    echo "Please run: export OPENAI_API_KEY=\"your-key\""
    exit 1
fi

echo "✓ API key detected"
echo

# Verify test suite passes
echo "Step 1: Verifying improvements..."
python benchmark/test_improvements.py
if [ $? -ne 0 ]; then
    echo "ERROR: Test suite failed"
    exit 1
fi
echo

# Backup old results (optional)
OLD_RESULT="benchmark/results/pathcov_agentic_z3_engine_gpt-5.2.jsonl"
if [ -f "$OLD_RESULT" ]; then
    BACKUP="${OLD_RESULT}.backup_$(date +%Y%m%d_%H%M%S)"
    echo "Step 2: Backing up old results to $BACKUP"
    cp "$OLD_RESULT" "$BACKUP"
    rm "$OLD_RESULT"
    echo "✓ Old results backed up and removed"
    echo
fi

# Run benchmark
echo "Step 3: Running agentic_z3_engine benchmark..."
echo "This will take several minutes (15 tasks, multiple paths each)"
echo
python benchmark/run_benchmark.py --run agentic_z3 --agentic-mode engine

if [ $? -ne 0 ]; then
    echo "ERROR: Benchmark run failed"
    exit 1
fi
echo

# Evaluate
echo "Step 4: Evaluating results..."
python benchmark/run_benchmark.py --evaluate

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed"
    exit 1
fi
echo

echo "======================================================================="
echo "✓ Benchmark Complete!"
echo "======================================================================="
echo
echo "Check the comparison table above."
echo "Expected improvements:"
echo "  - Exec: 8.9% → ~100%"
echo "  - Exact: 1.8% → ~70%"
echo "  - Similarity: 0.0286 → ~0.90"
echo
echo "Results saved to: benchmark/results/"





