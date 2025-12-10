#!/usr/bin/env python3
"""
Agentic-Z3 Runner for Path Coverage Benchmark

Uses Agentic-Z3's SMT solving approach to generate test cases
by formulating path conditions as Z3 constraints.
"""

import os
import sys
import json
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import re

# Add benchmark directory to path first (for our config)
benchmark_dir = str(Path(__file__).parent.parent)
if benchmark_dir not in sys.path:
    sys.path.insert(0, benchmark_dir)
# Add project root for agentic_z3 imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from benchmark config (explicit relative import)
import importlib.util
config_path = Path(__file__).parent.parent / "config.py"
spec = importlib.util.spec_from_file_location("benchmark_config", config_path)
benchmark_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_config)

# Get config values
LEETCODE_DATA = benchmark_config.LEETCODE_DATA
TARGET_PATHS_DATA = benchmark_config.TARGET_PATHS_DATA
SELECTED_TASKS_FILE = benchmark_config.SELECTED_TASKS_FILE
RESULTS_DIR = benchmark_config.RESULTS_DIR
AGENTIC_Z3_ROOT = benchmark_config.AGENTIC_Z3_ROOT
DEFAULT_MODEL = benchmark_config.DEFAULT_MODEL
get_output_filename = benchmark_config.get_output_filename

# Import adapters
adapters_dir = str(Path(__file__).parent.parent / "adapters")
if adapters_dir not in sys.path:
    sys.path.insert(0, adapters_dir)

from path_to_smt import (
    convert_path_to_smt,
    generate_smt_problem_for_path,
    parse_path_condition,
    extract_variables_from_condition,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], path: Path):
    """Write data to JSONL file."""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


class AgenticZ3Runner:
    """
    Runner for Agentic-Z3 approach.
    
    This runner formulates path conditions as SMT constraints and uses
    the Agentic-Z3 framework to solve them, generating test inputs.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        z3_timeout: int = 5000,
        max_retries: int = 3,
        use_engine: bool = True
    ):
        """
        Initialize Agentic-Z3 runner.
        
        Args:
            model: LLM model for Agentic-Z3 agents
            z3_timeout: Z3 solver timeout in milliseconds
            max_retries: Maximum retries for TTRL
            use_engine: If True, use full Agentic-Z3 engine; otherwise use direct Z3
        """
        self.model = model
        self.z3_timeout = z3_timeout
        self.max_retries = max_retries
        self.use_engine = use_engine
        
        # Try to import Agentic-Z3 components
        self.engine = None
        if use_engine:
            try:
                from agentic_z3.core.engine import Engine
                self.engine = Engine(
                    z3_timeout=z3_timeout,
                    max_retries=max_retries
                )
                print("Agentic-Z3 Engine initialized")
            except ImportError as e:
                print(f"Warning: Could not import Agentic-Z3 Engine: {e}")
                print("Falling back to direct Z3 mode")
                self.use_engine = False
        
        # Try to import Z3
        try:
            import z3
            self.z3 = z3
            print("Z3 solver available")
        except ImportError:
            print("Warning: Z3 not available. Install with: pip install z3-solver")
            self.z3 = None
    
    def _extract_input_params(self, code: str, func_name: str) -> List[Dict[str, str]]:
        """
        Extract input parameters from function signature.
        
        Returns:
            List of {name, type} dicts
        """
        # Match function definition
        pattern = rf"def\s+{func_name}\s*\(self,?\s*([^)]*)\)"
        match = re.search(pattern, code)
        
        if not match:
            return []
        
        params_str = match.group(1).strip()
        if not params_str:
            return []
        
        params = []
        for param in params_str.split(','):
            param = param.strip()
            if ':' in param:
                name, type_hint = param.split(':', 1)
                params.append({
                    'name': name.strip(),
                    'type': type_hint.strip()
                })
            else:
                params.append({
                    'name': param.strip(),
                    'type': 'Any'
                })
        
        return params
    
    def _infer_z3_type(self, type_hint: str) -> str:
        """Map Python type hints to Z3 types."""
        type_hint = type_hint.strip()
        
        # Handle List types
        if type_hint.startswith('List['):
            return 'Array'
        
        # Handle basic types
        type_map = {
            'int': 'Int',
            'float': 'Real',
            'bool': 'Bool',
            'str': 'String',
            'List': 'Array',
            'Dict': 'Array',
        }
        
        for py_type, z3_type in type_map.items():
            if py_type in type_hint:
                return z3_type
        
        return 'Int'  # Default
    
    def _solve_with_engine(
        self,
        task_data: Dict[str, Any],
        path_conditions: List[str]
    ) -> Dict[str, Any]:
        """
        Use Agentic-Z3 Engine to solve path constraints.
        
        Returns:
            Dict with status and model (if SAT)
        """
        if self.engine is None:
            return {'status': 'error', 'error': 'Engine not available'}
        
        # Generate problem description for Agentic-Z3
        func_name = task_data['func_name']
        description = task_data['description']
        
        # Build problem statement
        path_desc = '\n'.join(f"  - {c}" for c in path_conditions)
        
        problem = f"""
Find input values for Python function '{func_name}' that will execute through
the following path:

Function description: {description}

Path constraints (must ALL be satisfied in sequence):
{path_desc}

Generate test input values that will cause all these conditions to be true.
"""
        
        try:
            state = self.engine.solve(problem)
            
            return {
                'status': state.execution_status.name.lower(),
                'model': state.model,
                'code': state.current_code
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _solve_direct_z3(
        self,
        task_data: Dict[str, Any],
        path_conditions: List[str]
    ) -> Dict[str, Any]:
        """
        Directly use Z3 to solve path constraints.
        
        This is a simplified approach that converts conditions to Z3.
        """
        if self.z3 is None:
            return {'status': 'error', 'error': 'Z3 not available'}
        
        z3 = self.z3
        
        # Convert path to SMT constraints
        constraint_set = convert_path_to_smt(path_conditions)
        
        # Create solver
        solver = z3.Solver()
        solver.set("timeout", self.z3_timeout)
        
        # Declare variables
        vars_dict = {}
        for var_name, var_type in constraint_set.all_variables.items():
            if var_type == 'Int':
                vars_dict[var_name] = z3.Int(var_name)
            elif var_type == 'Bool':
                vars_dict[var_name] = z3.Bool(var_name)
            elif var_type == 'Real':
                vars_dict[var_name] = z3.Real(var_name)
            else:
                vars_dict[var_name] = z3.Int(var_name)
        
        # Add constraints
        for constraint in constraint_set.constraints:
            try:
                # Evaluate Z3 code in context with our variables
                z3_expr = eval(constraint.z3_code, {'__builtins__': {}}, 
                              {**vars_dict, 'And': z3.And, 'Or': z3.Or, 
                               'Not': z3.Not, 'True': True, 'False': False})
                if z3_expr is not None and z3_expr is not True:
                    solver.add(z3_expr)
            except Exception as e:
                # Skip constraints that can't be translated
                pass
        
        # Solve
        result = solver.check()
        
        if result == z3.sat:
            model = solver.model()
            model_dict = {}
            for var_name, var in vars_dict.items():
                try:
                    val = model.evaluate(var)
                    model_dict[var_name] = str(val)
                except:
                    pass
            return {
                'status': 'sat',
                'model': model_dict
            }
        elif result == z3.unsat:
            return {'status': 'unsat'}
        else:
            return {'status': 'unknown'}
    
    def _generate_test_from_model(
        self,
        func_name: str,
        input_params: List[Dict[str, str]],
        model: Optional[Dict[str, Any]],
        status: str
    ) -> str:
        """Generate test case from Z3 model."""
        
        if status != 'sat' or not model:
            return f'''def test_{func_name}():
    solution = Solution()
    # Agentic-Z3 status: {status}
    # Could not find satisfying input
    pass
'''
        
        # Try to map model values to function parameters
        param_values = []
        for param in input_params:
            name = param['name']
            type_hint = param['type']
            
            # Look for matching variable in model
            value = model.get(name)
            
            if value is not None:
                param_values.append(f"{name}={value}")
            elif 'List' in type_hint:
                param_values.append(f"{name}=[]")
            elif 'int' in type_hint.lower():
                param_values.append(f"{name}=0")
            elif 'str' in type_hint.lower():
                param_values.append(f"{name}=''")
            else:
                param_values.append(f"{name}=None")
        
        params_str = ', '.join(param_values) if param_values else ''
        
        return f'''def test_{func_name}():
    solution = Solution()
    # Generated by Agentic-Z3
    # Model: {model}
    result = solution.{func_name}({params_str})
'''
    
    def run_task(
        self,
        task_data: Dict[str, Any],
        paths_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Agentic-Z3 test generation for a single task.
        """
        func_name = task_data['func_name']
        code = task_data['python_solution']
        difficulty = task_data['difficulty']
        task_num = task_data['task_num']
        task_title = task_data.get('task_title', '')
        
        # Extract input parameters
        input_params = self._extract_input_params(code, func_name)
        
        condition_paths = paths_data.get('sampled_condition_paths', [])
        
        generated_tests = []
        
        for path_idx, condition_path in enumerate(condition_paths):
            # Solve using appropriate method
            if self.use_engine and self.engine is not None:
                result = self._solve_with_engine(task_data, condition_path)
            else:
                result = self._solve_direct_z3(task_data, condition_path)
            
            # Generate test from result
            test = self._generate_test_from_model(
                func_name,
                input_params,
                result.get('model'),
                result.get('status', 'error')
            )
            generated_tests.append(test)
            
            status = result.get('status', 'error')
            print(f"  Path {path_idx + 1}/{len(condition_paths)}: {status}")
        
        return {
            'task_num': task_num,
            'task_title': task_title,
            'func_name': func_name,
            'difficulty': difficulty,
            'code': code,
            'tests': generated_tests
        }
    
    def run_selected_tasks(
        self,
        output_path: Optional[Path] = None,
        save_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run Agentic-Z3 on all selected tasks.
        """
        # Load selection
        with open(SELECTED_TASKS_FILE, 'r') as f:
            selection = json.load(f)
        
        # Load datasets
        leetcode_data = read_jsonl(LEETCODE_DATA)
        paths_data = read_jsonl(TARGET_PATHS_DATA)
        
        # Get output path
        mode = "engine" if self.use_engine else "direct"
        if output_path is None:
            output_path = RESULTS_DIR / get_output_filename(f"agentic_z3_{mode}", self.model)
        
        # Load existing progress
        results = []
        completed_indices = set()
        
        if output_path.exists():
            existing = read_jsonl(output_path)
            results = existing
            completed_indices = {r['task_num'] for r in existing}
            print(f"Resuming from {len(existing)} completed tasks")
        
        # Process each selected task
        all_indices = selection['all_indices']
        
        for idx in tqdm(all_indices, desc="Agentic-Z3 generation"):
            task_data = leetcode_data[idx]
            task_paths = paths_data[idx]
            
            # Skip if already completed
            if task_data['task_num'] in completed_indices:
                continue
            
            print(f"\nTask {task_data['task_num']}: {task_data['task_title']}")
            
            result = self.run_task(task_data, task_paths)
            results.append(result)
            
            # Save progress
            if save_progress:
                write_jsonl(results, output_path)
        
        # Final save
        write_jsonl(results, output_path)
        print(f"\nResults saved to {output_path}")
        
        return results


class AgenticZ3LLMRunner:
    """
    Alternative Agentic-Z3 runner using LLM for SMT-style reasoning.
    
    This approach prompts the LLM to think like an SMT solver,
    following Agentic-Z3's multi-agent approach.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package required")
        return self._client
    
    def _build_smt_prompt(
        self,
        code: str,
        func_name: str,
        description: str,
        path_conditions: List[str]
    ) -> str:
        """Build an SMT-style prompt for the LLM."""
        # Parse conditions to extract structure
        constraint_set = convert_path_to_smt(path_conditions)
        
        vars_info = '\n'.join(f"  - {k}: {v}" for k, v in constraint_set.all_variables.items())
        constraints_info = '\n'.join(f"  {i+1}. {c.description}" for i, c in enumerate(constraint_set.constraints))
        
        return f"""You are an SMT (Satisfiability Modulo Theories) solver. Your task is to find concrete values that satisfy a set of constraints.

## Problem Context
Function: {func_name}
Description: {description}

## Variables (with inferred types)
{vars_info}

## Path Constraints (must ALL be satisfied)
{constraints_info}

## Your Task
1. Analyze each constraint carefully
2. Find values for the INPUT PARAMETERS that will make ALL constraints true
3. Consider edge cases and boundary conditions
4. The constraints form an execution path - they must be satisfiable in sequence

## Output Format
Generate a Python test function with concrete input values:

```python
def test_{func_name}():
    solution = Solution()
    # Reasoning: [explain your constraint solving]
    result = solution.{func_name}(...)  # Fill in actual input values
```

Think step-by-step about what values would satisfy each constraint."""
    
    def generate_test(
        self,
        code: str,
        func_name: str,
        description: str,
        path_conditions: List[str]
    ) -> str:
        """Generate test using LLM with SMT-style reasoning."""
        prompt = self._build_smt_prompt(
            code, func_name, description, path_conditions
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SMT solver that finds satisfying assignments for logical constraints."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"# Error: {e}"
    
    def run_task(
        self,
        task_data: Dict[str, Any],
        paths_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run LLM-based SMT solving for a single task."""
        func_name = task_data['func_name']
        code = task_data['python_solution']
        description = task_data['description']
        difficulty = task_data['difficulty']
        task_num = task_data['task_num']
        task_title = task_data.get('task_title', '')
        
        condition_paths = paths_data.get('sampled_condition_paths', [])
        
        generated_tests = []
        
        for path_idx, condition_path in enumerate(condition_paths):
            test = self.generate_test(
                code, func_name, description, condition_path
            )
            generated_tests.append(test)
            print(f"  Path {path_idx + 1}/{len(condition_paths)}: Generated")
        
        return {
            'task_num': task_num,
            'task_title': task_title,
            'func_name': func_name,
            'difficulty': difficulty,
            'code': code,
            'tests': generated_tests
        }
    
    def run_selected_tasks(
        self,
        output_path: Optional[Path] = None,
        save_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Run on all selected tasks."""
        with open(SELECTED_TASKS_FILE, 'r') as f:
            selection = json.load(f)
        
        leetcode_data = read_jsonl(LEETCODE_DATA)
        paths_data = read_jsonl(TARGET_PATHS_DATA)
        
        if output_path is None:
            output_path = RESULTS_DIR / get_output_filename("agentic_z3_llm", self.model)
        
        results = []
        completed_indices = set()
        
        if output_path.exists():
            existing = read_jsonl(output_path)
            results = existing
            completed_indices = {r['task_num'] for r in existing}
        
        all_indices = selection['all_indices']
        
        for idx in tqdm(all_indices, desc="Agentic-Z3 LLM generation"):
            task_data = leetcode_data[idx]
            task_paths = paths_data[idx]
            
            if task_data['task_num'] in completed_indices:
                continue
            
            print(f"\nTask {task_data['task_num']}: {task_data['task_title']}")
            
            result = self.run_task(task_data, task_paths)
            results.append(result)
            
            if save_progress:
                write_jsonl(results, output_path)
        
        write_jsonl(results, output_path)
        print(f"\nResults saved to {output_path}")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Agentic-Z3 runner for path coverage")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--mode", type=str, choices=['engine', 'direct', 'llm'],
                        default='direct',
                        help="Mode: engine (full Agentic-Z3), direct (Z3 only), llm (LLM reasoning)")
    parser.add_argument("--timeout", type=int, default=5000,
                        help="Z3 timeout in ms (default: 5000)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max TTRL retries for engine mode (default: 3)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Agentic-Z3 Runner")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    
    if args.dry_run:
        print("\nDry run mode - checking configuration")
        
        # Check Z3
        try:
            import z3
            print(f"Z3 version: {z3.get_version_string()}")
        except ImportError:
            print("Z3: Not installed")
        
        # Check Agentic-Z3 engine
        try:
            from agentic_z3.core.engine import Engine
            print("Agentic-Z3 Engine: Available")
        except ImportError as e:
            print(f"Agentic-Z3 Engine: Not available ({e})")
        
        return
    
    if args.mode == 'llm':
        runner = AgenticZ3LLMRunner(model=args.model)
    else:
        runner = AgenticZ3Runner(
            model=args.model,
            z3_timeout=args.timeout,
            max_retries=args.max_retries,
            use_engine=(args.mode == 'engine')
        )
    
    results = runner.run_selected_tasks(output_path=args.output)
    print(f"\nCompleted: {len(results)} tasks")


if __name__ == "__main__":
    main()
