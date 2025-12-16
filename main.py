#!/usr/bin/env python3
"""
Agentic-Z3: An Autonomous SMT Solving Framework

Entry point for the multi-agent system that combines:
- Hierarchical Planning (Architect agent)
- Type-Aware Probing & TTRL with Soft Reset (Worker agent)  
- Deep Diagnosis & Skill Crystallization (Coach agent)

Usage:
    python main.py "Find integers x, y where x + y = 10 and x > y"
    python main.py --file problem.txt
    echo "problem description" | python main.py --stdin

The system will:
1. Decompose the problem into a structured blueprint
2. Generate type-safe Z3 code with interactive probing
3. Execute and diagnose results
4. Learn from successes (skill crystallization) or failures (soft reset)
"""

import argparse
import sys
from pathlib import Path

from config import settings
from agentic_z3.core.engine import Engine
from agentic_z3.core.state import ExecutionStatus
from agentic_z3.utils.logger import get_logger, LogCategory

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for problem input."""
    parser = argparse.ArgumentParser(
        description="Agentic-Z3: Autonomous SMT Solving Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Find x where x^2 = 4"
  %(prog)s --file problems/scheduling.txt
  %(prog)s --stdin < problem.txt
  %(prog)s --verbose "Prove x + y >= x for all non-negative y"
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "problem",
        nargs="?",
        help="Problem description as a string"
    )
    input_group.add_argument(
        "--file", "-f",
        type=Path,
        help="Path to file containing problem description"
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read problem from stdin"
    )
    
    # Configuration overrides
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=settings.Z3_TIMEOUT,
        help=f"Z3 solver timeout in ms (default: {settings.Z3_TIMEOUT})"
    )
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=settings.MAX_TTRL_RETRIES,
        help=f"Maximum TTRL retries (default: {settings.MAX_TTRL_RETRIES})"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose agent thought logging"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable curriculum warmup for complex problems"
    )
    
    return parser.parse_args()


def load_problem(args: argparse.Namespace) -> str:
    """
    Load problem description from the specified source.
    
    Supports three input modes:
    1. Direct string argument
    2. File path
    3. Standard input (stdin)
    
    Returns:
        The problem description as a string.
        
    Raises:
        SystemExit: If no input is provided or file cannot be read.
    """
    if args.problem:
        return args.problem
    
    if args.file:
        if not args.file.exists():
            logger.error(f"File not found: {args.file}", category=LogCategory.SYSTEM)
            sys.exit(1)
        try:
            return args.file.read_text(encoding="utf-8").strip()
        except IOError as e:
            logger.error(f"Failed to read file: {e}", category=LogCategory.SYSTEM)
            sys.exit(1)
    
    if args.stdin:
        return sys.stdin.read().strip()
    
    # No input provided
    logger.error(
        "No problem provided. Use positional argument, --file, or --stdin",
        category=LogCategory.SYSTEM
    )
    sys.exit(1)


def main() -> int:
    """
    Main entry point for Agentic-Z3.
    
    Orchestrates the solving pipeline:
    1. Parse arguments and load problem
    2. Initialize the Engine with configuration
    3. Execute the solve loop
    4. Report results
    
    Returns:
        Exit code: 0 for SAT, 1 for UNSAT, 2 for UNKNOWN/ERROR
    """
    args = parse_args()
    
    # Apply configuration overrides
    if args.verbose:
        settings.LOG_SHOW_AGENT_THOUGHTS = True
        settings.LOG_LEVEL = "DEBUG"
    if args.json_output:
        settings.LOG_JSON_MODE = True
    if args.no_warmup:
        settings.ENABLE_CURRICULUM_WARMUP = False
    
    # Load the problem
    problem = load_problem(args)
    
    logger.info("=" * 60, category=LogCategory.SYSTEM)
    logger.info("Agentic-Z3 Autonomous SMT Solver", category=LogCategory.SYSTEM)
    logger.info("=" * 60, category=LogCategory.SYSTEM)
    logger.info(f"Problem: {problem[:100]}{'...' if len(problem) > 100 else ''}", 
                category=LogCategory.SYSTEM)
    
    # Initialize and run the engine
    try:
        engine = Engine(
            z3_timeout=args.timeout,
            max_retries=args.max_retries
        )
        
        final_state = engine.solve(problem)
        
        # Report results
        logger.info("=" * 60, category=LogCategory.SYSTEM)
        logger.info(f"Final Status: {final_state.execution_status.name}", 
                    category=LogCategory.SYSTEM)
        
        if final_state.execution_status == ExecutionStatus.SAT:
            logger.info("Solution found!", category=LogCategory.SYSTEM)
            if final_state.model:
                logger.info(f"Model: {final_state.model}", category=LogCategory.SYSTEM)
            return 0
        
        elif final_state.execution_status == ExecutionStatus.UNSAT:
            logger.info("Problem is unsatisfiable (no solution exists)", 
                        category=LogCategory.SYSTEM)
            if final_state.unsat_core_dump:
                logger.info(f"Conflicting constraints: {final_state.unsat_core_dump}",
                            category=LogCategory.SYSTEM)
            return 1
        
        else:
            logger.warning(
                f"Could not determine satisfiability: {final_state.execution_status.name}",
                category=LogCategory.SYSTEM
            )
            if final_state.failure_summaries:
                logger.info(f"Failure summary: {final_state.failure_summaries[-1]}",
                            category=LogCategory.SYSTEM)
            return 2
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user", category=LogCategory.SYSTEM)
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", category=LogCategory.SYSTEM)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())









