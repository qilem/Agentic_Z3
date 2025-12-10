#!/usr/bin/env python3
"""
Path Coverage Benchmark Runner

Main orchestration script that runs the complete benchmarking pipeline:
1. Select tasks (if not already done)
2. Run all three approaches
3. Evaluate and compare results

Usage:
    python run_benchmark.py --all                    # Run everything
    python run_benchmark.py --run zero_shot autoexe # Run specific approaches
    python run_benchmark.py --evaluate              # Evaluate existing results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add benchmark directory to path
benchmark_dir = str(Path(__file__).parent)
if benchmark_dir not in sys.path:
    sys.path.insert(0, benchmark_dir)

# Import config
import importlib.util
config_path = Path(__file__).parent / "config.py"
spec = importlib.util.spec_from_file_location("benchmark_config", config_path)
benchmark_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_config)

RESULTS_DIR = benchmark_config.RESULTS_DIR
SELECTED_TASKS_FILE = benchmark_config.SELECTED_TASKS_FILE
DEFAULT_MODEL = benchmark_config.DEFAULT_MODEL


def ensure_task_selection():
    """Ensure tasks are selected, run selection if not."""
    if not SELECTED_TASKS_FILE.exists():
        print("=" * 60)
        print("Step 1: Selecting Tasks")
        print("=" * 60)
        
        from task_selector import select_tasks, save_selection, print_selection_summary
        
        selection = select_tasks()
        save_selection(selection)
        print_selection_summary(selection)
        
        return selection
    else:
        print(f"Using existing task selection: {SELECTED_TASKS_FILE}")
        with open(SELECTED_TASKS_FILE, 'r') as f:
            return json.load(f)


def run_zero_shot(model: str = DEFAULT_MODEL):
    """Run zero-shot baseline."""
    print("\n" + "=" * 60)
    print("Running Zero-Shot Baseline")
    print("=" * 60)
    
    from runners.zero_shot_runner import ZeroShotRunner
    
    try:
        runner = ZeroShotRunner(model=model)
        results = runner.run_selected_tasks()
        print(f"Zero-shot completed: {len(results)} tasks")
        return True
    except Exception as e:
        print(f"Zero-shot failed: {e}")
        return False


def run_autoexe(model: str = DEFAULT_MODEL, use_llm_mode: bool = True):
    """Run AutoExe approach."""
    print("\n" + "=" * 60)
    print("Running AutoExe")
    print("=" * 60)
    
    from runners.autoexe_runner import AutoExeRunner, AutoExeLLMRunner
    
    try:
        if use_llm_mode:
            runner = AutoExeLLMRunner(model=model)
        else:
            runner = AutoExeRunner(model=model)
        
        results = runner.run_selected_tasks()
        print(f"AutoExe completed: {len(results)} tasks")
        return True
    except Exception as e:
        print(f"AutoExe failed: {e}")
        return False


def run_agentic_z3(model: str = DEFAULT_MODEL, mode: str = 'llm'):
    """Run Agentic-Z3 approach."""
    print("\n" + "=" * 60)
    print("Running Agentic-Z3")
    print("=" * 60)
    
    from runners.agentic_z3_runner import AgenticZ3Runner, AgenticZ3LLMRunner
    
    try:
        if mode == 'llm':
            runner = AgenticZ3LLMRunner(model=model)
        else:
            runner = AgenticZ3Runner(
                model=model,
                use_engine=(mode == 'engine')
            )
        
        results = runner.run_selected_tasks()
        print(f"Agentic-Z3 completed: {len(results)} tasks")
        return True
    except Exception as e:
        print(f"Agentic-Z3 failed: {e}")
        return False


def run_evaluation():
    """Run evaluation on all available results."""
    print("\n" + "=" * 60)
    print("Running Evaluation")
    print("=" * 60)
    
    from evaluate import Evaluator
    
    # Discover all result files
    approaches = {}
    for f in RESULTS_DIR.glob("*.jsonl"):
        name = f.stem.replace("pathcov_", "")
        approaches[name] = f
    
    if not approaches:
        print("No results found to evaluate")
        return False
    
    print(f"Found {len(approaches)} result files")
    
    evaluator = Evaluator()
    results = evaluator.evaluate_all(approaches)
    Evaluator.print_comparison(results)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = RESULTS_DIR / f"summary_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "results": {name: r.to_dict() for name, r in results.items()}
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")
    return True


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           PATH COVERAGE BENCHMARK PIPELINE                    ║
║                                                               ║
║  Comparing:                                                   ║
║    1. Zero-Shot Baseline (Direct LLM prompting)              ║
║    2. AutoExe (LLM-powered symbolic execution)               ║
║    3. Agentic-Z3 (SMT constraint solving)                    ║
║                                                               ║
║  Dataset: 15 Medium + 15 Hard tasks from TestEval            ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_status():
    """Print current status of the benchmark."""
    print("\n" + "=" * 60)
    print("BENCHMARK STATUS")
    print("=" * 60)
    
    # Check task selection
    if SELECTED_TASKS_FILE.exists():
        with open(SELECTED_TASKS_FILE, 'r') as f:
            selection = json.load(f)
        print(f"✓ Tasks selected: {selection['total_count']} tasks")
    else:
        print("✗ Tasks not selected yet")
    
    # Check results
    print("\nResult files:")
    result_files = list(RESULTS_DIR.glob("*.jsonl"))
    if result_files:
        for f in result_files:
            data = []
            with open(f, 'r') as fp:
                for line in fp:
                    data.append(json.loads(line))
            print(f"  ✓ {f.name}: {len(data)} tasks")
    else:
        print("  (no results yet)")
    
    print("=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Path Coverage Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --all              # Run complete pipeline
  python run_benchmark.py --run zero_shot    # Run only zero-shot
  python run_benchmark.py --evaluate         # Evaluate existing results
  python run_benchmark.py --status           # Show current status
        """
    )
    
    parser.add_argument("--all", action="store_true",
                        help="Run complete pipeline (all approaches + evaluation)")
    parser.add_argument("--run", nargs="+", 
                        choices=["zero_shot", "autoexe", "agentic_z3"],
                        help="Run specific approaches")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation on existing results")
    parser.add_argument("--select-tasks", action="store_true",
                        help="Select/re-select benchmark tasks")
    parser.add_argument("--status", action="store_true",
                        help="Show current benchmark status")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--llm-mode", action="store_true", default=True,
                        help="Use LLM-based runners (default)")
    parser.add_argument("--no-llm-mode", action="store_false", dest="llm_mode",
                        help="Use non-LLM runners where available")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print_banner()
    
    # Status check
    if args.status:
        print_status()
        return
    
    # Task selection
    if args.select_tasks:
        if SELECTED_TASKS_FILE.exists():
            SELECTED_TASKS_FILE.unlink()
        ensure_task_selection()
        return
    
    # Run specific approaches
    if args.run:
        ensure_task_selection()
        
        for approach in args.run:
            if approach == "zero_shot":
                run_zero_shot(model=args.model)
            elif approach == "autoexe":
                run_autoexe(model=args.model, use_llm_mode=args.llm_mode)
            elif approach == "agentic_z3":
                run_agentic_z3(model=args.model, mode='llm' if args.llm_mode else 'direct')
        
        if args.evaluate or args.all:
            run_evaluation()
        return
    
    # Evaluate only
    if args.evaluate:
        run_evaluation()
        return
    
    # Run all
    if args.all:
        print("Running complete benchmark pipeline...")
        
        selection = ensure_task_selection()
        
        success = {
            "zero_shot": run_zero_shot(model=args.model),
            "autoexe": run_autoexe(model=args.model, use_llm_mode=args.llm_mode),
            "agentic_z3": run_agentic_z3(model=args.model, mode='llm' if args.llm_mode else 'direct'),
        }
        
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        for approach, ok in success.items():
            status = "✓ Completed" if ok else "✗ Failed"
            print(f"  {approach}: {status}")
        
        # Run evaluation
        run_evaluation()
        return
    
    # Default: show status
    print_status()
    print("\nUse --help for usage information")
    print("Use --all to run the complete pipeline")


if __name__ == "__main__":
    main()
