#!/usr/bin/env python3
"""
AutoExe Runner for Path Coverage Benchmark

Uses AutoExe's LLM-powered symbolic execution to generate test cases
that satisfy target path conditions.
"""

import os
import sys
import json
import subprocess
import tempfile
import threading
import time
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Set

# Add benchmark directory to path first (for our config)
benchmark_dir = str(Path(__file__).parent.parent)
if benchmark_dir not in sys.path:
    sys.path.insert(0, benchmark_dir)

# Import from benchmark config (explicit)
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
AUTOEXE_EXECUTOR = benchmark_config.AUTOEXE_EXECUTOR
DEFAULT_MODEL = benchmark_config.DEFAULT_MODEL
get_output_filename = benchmark_config.get_output_filename

# Import adapters
adapters_dir = str(Path(__file__).parent.parent / "adapters")
if adapters_dir not in sys.path:
    sys.path.insert(0, adapters_dir)

from path_to_precondition import (
    create_autoexe_wrapper,
    create_simple_constraint_program,
    path_conditions_to_assertion,
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


def compute_selection_hash(selection: Dict[str, Any]) -> str:
    """Compute a short hash of the task selection for identification."""
    task_nums = sorted([t.get('task_num', 0) for t in selection.get('tasks', [])])
    key = f"{selection.get('seed', 0)}_{task_nums}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def get_selected_task_nums(selection: Dict[str, Any]) -> Set[int]:
    """Extract set of task_nums from selection."""
    return {t.get('task_num') for t in selection.get('tasks', []) if 'task_num' in t}


class AutoExeRunner:
    """Runner for AutoExe approach."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        executor_path: Path = AUTOEXE_EXECUTOR,
        timeout: int = 120,
        use_baseline: bool = False,
        max_workers: int = 4
    ):
        """
        Initialize AutoExe runner.
        
        Args:
            model: LLM model to use (format: "openapi::model-name" or local)
            executor_path: Path to AutoExe executor binary
            timeout: Timeout per execution in seconds
            use_baseline: If True, use --skip-slice (baseline mode)
            max_workers: Max parallel workers (default: 4)
        """
        self.model = model
        self.executor_path = executor_path
        self.timeout = timeout
        self.use_baseline = use_baseline
        self.max_workers = max_workers
        
        # Check if executor exists
        if not executor_path.exists():
            print(f"Warning: AutoExe executor not found at {executor_path}")
            print("AutoExe runs will be simulated.")
            self.executor_available = False
        else:
            self.executor_available = True
    
    def _format_model_name(self, model: str) -> str:
        """Format model name for AutoExe."""
        # AutoExe uses namespaced model names
        if "::" in model:
            return model
        # Assume OpenAI for common models
        if model.startswith("gpt-"):
            return f"openapi::{model}"
        return model
    
    def _run_autoexe(
        self,
        source_file: Path,
        source_dir: Path,
        output_file: Path,
        result_json_file: Path
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Run AutoExe on a source file.
        
        Returns:
            Tuple of (result_status, elapsed_time, parsed_result_dict)
        """
        if not self.executor_available:
            return "skipped", 0.0, {}
        
        model_arg = self._format_model_name(self.model)
        
        cmd = [
            str(self.executor_path),
            str(source_dir),
            str(source_file),
            "--auto-entry",
            "--output", str(output_file),
            "--result-output", str(result_json_file),  # Get detailed JSON output
            "--model", model_arg
        ]
        
        if self.use_baseline:
            cmd.append("--skip-slice")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                timeout=self.timeout,
                capture_output=True,
                text=True
            )
            elapsed = time.time() - start_time
            
            status = "unknown"
            result_data = {}
            
            # Read status from output file
            if output_file.exists():
                with open(output_file, 'r') as f:
                    content = f.read().strip()
                    lines = content.split('\n')
                    if lines:
                        status = lines[0].lower()
            
            # Read detailed results from JSON file
            if result_json_file.exists():
                try:
                    with open(result_json_file, 'r') as f:
                        result_data = json.load(f)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try reading as text
                    with open(result_json_file, 'r') as f:
                        result_data = {"raw_output": f.read()}
            
            # Also capture stdout/stderr for debugging
            result_data['stdout'] = result.stdout
            result_data['stderr'] = result.stderr
            result_data['returncode'] = result.returncode
            
            return status, elapsed, result_data
            
        except subprocess.TimeoutExpired:
            return "timeout", self.timeout, {"error": "timeout"}
        except Exception as e:
            print(f"AutoExe error: {e}")
            return "error", 0.0, {"error": str(e)}
    
    def _parse_assignments_from_output(
        self,
        result_data: Dict[str, Any],
        input_params: List[Tuple[str, str]]
    ) -> Dict[str, str]:
        """
        Parse variable assignments from AutoExe's output.
        
        Looks for assignments in:
        1. The JSON result data (queries/responses)
        2. Stdout/stderr for patterns like "x = 5"
        
        Returns:
            Dict mapping parameter names to their values
        """
        assignments = {}
        param_names = [name for name, _ in input_params]
        
        # Combine all text output sources
        text_sources = []
        
        # Check for queries and responses in JSON
        if isinstance(result_data, dict):
            # Look for query/response pairs
            for key in ['queries', 'responses', 'query', 'response']:
                if key in result_data:
                    value = result_data[key]
                    if isinstance(value, list):
                        text_sources.extend(str(v) for v in value)
                    else:
                        text_sources.append(str(value))
            
            # Add stdout/stderr
            if 'stdout' in result_data:
                text_sources.append(result_data['stdout'])
            if 'stderr' in result_data:
                text_sources.append(result_data['stderr'])
            if 'raw_output' in result_data:
                text_sources.append(result_data['raw_output'])
        
        combined_text = '\n'.join(text_sources)
        
        # Parse assignments using multiple patterns
        for param_name in param_names:
            # Pattern 1: param = value (with various formats)
            patterns = [
                rf'{re.escape(param_name)}\s*=\s*([^\n,;]+)',  # Simple assignment
                rf'\b{re.escape(param_name)}\s*:\s*([^\n,;]+)',  # Colon notation
                rf'"{re.escape(param_name)}"\s*:\s*([^\n,]+)',  # JSON-style
            ]
            
            for pattern in patterns:
                match = re.search(pattern, combined_text)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    value = value.rstrip(',;')
                    # Remove trailing comments
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    if '//' in value:
                        value = value.split('//')[0].strip()
                    if value:
                        assignments[param_name] = value
                        break
        
        return assignments
    
    def _get_safe_default(self, type_hint: str) -> str:
        """Get a safe default value for a type hint."""
        type_hint_lower = type_hint.lower().strip()
        
        if 'list[list[' in type_hint_lower:
            return '[[0]]'
        elif 'list[' in type_hint_lower:
            return '[0]'
        elif 'str' in type_hint_lower:
            return '"a"'
        elif 'bool' in type_hint_lower:
            return 'True'
        elif 'float' in type_hint_lower:
            return '1.0'
        elif 'int' in type_hint_lower:
            return '1'
        elif 'dict' in type_hint_lower:
            return '{}'
        elif 'optional' in type_hint_lower:
            return 'None'
        else:
            return '0'
    
    def _generate_test_from_result(
        self,
        func_name: str,
        result_status: str,
        input_params: List[Tuple[str, str]],
        assignments: Dict[str, str]
    ) -> str:
        """
        Generate a test case from AutoExe result.
        
        Args:
            func_name: Function name to test
            result_status: AutoExe result status (sat/unsat/etc)
            input_params: List of (param_name, type_hint) tuples
            assignments: Parsed variable assignments from AutoExe
        
        Returns:
            A Python test function string
        """
        if result_status in ["sat", "success"]:
            # Build argument list
            args = []
            missing_params = []
            
            for param_name, type_hint in input_params:
                if param_name in assignments:
                    value = assignments[param_name]
                    # Validate that the value looks reasonable
                    try:
                        # Simple validation - try to compile as expression
                        compile(value, '<string>', 'eval')
                        args.append(value)
                    except SyntaxError:
                        # Use safe default if value is malformed
                        args.append(self._get_safe_default(type_hint))
                        missing_params.append(f"{param_name} (malformed: {value[:20]})")
                else:
                    # Use safe default based on type hint
                    args.append(self._get_safe_default(type_hint))
                    missing_params.append(param_name)
            
            args_str = ', '.join(args)
            
            test_code = f'''def test_{func_name}():
    solution = Solution()
    # Generated by AutoExe (status: {result_status})'''
            
            if assignments:
                model_comment = ', '.join(f"{k}={v}" for k, v in list(assignments.items())[:5])
                test_code += f'\n    # Parsed assignments: {model_comment}'
            
            if missing_params:
                test_code += f'\n    # Using defaults for: {", ".join(missing_params[:5])}'
            
            test_code += f'''
    result = solution.{func_name}({args_str})
'''
            return test_code
        else:
            return f'''def test_{func_name}():
    solution = Solution()
    # AutoExe status: {result_status}
    # Could not find satisfying input
    pass
'''
    
    def run_task(
        self,
        task_data: Dict[str, Any],
        paths_data: Dict[str, Any],
        temp_dir: Path
    ) -> Dict[str, Any]:
        """
        Run AutoExe test generation for a single task.
        """
        func_name = task_data['func_name']
        code = task_data['python_solution']
        difficulty = task_data['difficulty']
        task_num = task_data['task_num']
        task_title = task_data.get('task_title', '')
        
        condition_paths = paths_data.get('sampled_condition_paths', [])
        
        # Extract input parameters once for the task
        from path_to_precondition import extract_function_inputs
        input_params = extract_function_inputs(code, func_name)
        
        generated_tests = []
        debug_info = []
        
        for path_idx, condition_path in enumerate(condition_paths):
            # Create wrapper program for AutoExe
            wrapper = create_autoexe_wrapper(task_data, condition_path)
            
            # Write to temp file
            source_file = temp_dir / f"task_{task_num}_path_{path_idx}.py"
            output_file = temp_dir / f"result_{task_num}_path_{path_idx}.txt"
            result_json_file = temp_dir / f"result_{task_num}_path_{path_idx}.json"
            
            with open(source_file, 'w') as f:
                f.write(wrapper.code)
            
            # Run AutoExe with detailed output
            status, elapsed, result_data = self._run_autoexe(
                source_file, temp_dir, output_file, result_json_file
            )
            
            # Parse assignments from AutoExe output
            assignments = self._parse_assignments_from_output(result_data, input_params)
            
            # Generate test from result with parsed assignments
            test = self._generate_test_from_result(
                func_name, status, input_params, assignments
            )
            generated_tests.append(test)
            
            # Store debug info
            debug_info.append({
                'path_idx': path_idx,
                'status': status,
                'elapsed': elapsed,
                'assignments': assignments,
                'wrapper_code': wrapper.code[:1000] if wrapper.code else None,  # Truncate for storage
            })
            
            print(f"  Path {path_idx + 1}/{len(condition_paths)}: {status} ({elapsed:.1f}s)")
        
        return {
            'task_num': task_num,
            'task_title': task_title,
            'func_name': func_name,
            'difficulty': difficulty,
            'code': code,
            'tests': generated_tests,
            'debug': debug_info
        }
    
    def run_selected_tasks(
        self,
        output_path: Optional[Path] = None,
        save_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run AutoExe on all selected tasks.
        
        Filters existing results to only include tasks in current selection,
        preventing stale/mixed results from different selections.
        """
        # Load selection
        with open(SELECTED_TASKS_FILE, 'r') as f:
            selection = json.load(f)
        
        # Get valid task_nums for current selection
        valid_task_nums = get_selected_task_nums(selection)
        selection_hash = compute_selection_hash(selection)
        print(f"Selection hash: {selection_hash}, valid tasks: {sorted(valid_task_nums)}")
        
        # Load datasets
        leetcode_data = read_jsonl(LEETCODE_DATA)
        paths_data = read_jsonl(TARGET_PATHS_DATA)
        
        # Get output path
        approach = "autoexe_baseline" if self.use_baseline else "autoexe"
        if output_path is None:
            output_path = RESULTS_DIR / get_output_filename(approach, self.model)
        
        # Load existing progress, filtering to only include entries in current selection
        results = []
        completed_indices = set()
        
        if output_path.exists():
            existing = read_jsonl(output_path)
            original_count = len(existing)
            
            # Filter to only keep entries that are in the current selection
            results = [r for r in existing if r.get('task_num') in valid_task_nums]
            filtered_count = original_count - len(results)
            
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} entries not in current selection")
            
            completed_indices = {r['task_num'] for r in results}
            print(f"Resuming from {len(results)} completed tasks (in current selection)")
        
        # Filter tasks to process (exclude already completed)
        all_indices = selection['all_indices']
        tasks_to_process = [
            (leetcode_data[idx], paths_data[idx])
            for idx in all_indices
            if leetcode_data[idx]['task_num'] not in completed_indices
        ]
        
        if not tasks_to_process:
            print("All tasks already completed")
            return results
        
        print(f"Processing {len(tasks_to_process)} tasks with {self.max_workers} workers")
        
        # Thread-safe progress saving
        write_lock = threading.Lock()
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            def process_task(task_tuple):
                task_data, task_paths = task_tuple
                return self.run_task(task_data, task_paths, temp_path)
            
            # Process tasks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(process_task, task_tuple): task_tuple[0]['task_num']
                    for task_tuple in tasks_to_process
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="AutoExe generation"):
                    task_num = futures[future]
                    try:
                        result = future.result()
                        with write_lock:
                            results.append(result)
                            if save_progress:
                                write_jsonl(results, output_path)
                    except Exception as e:
                        print(f"\nTask {task_num} failed: {e}")
        
        # Final save
        write_jsonl(results, output_path)
        print(f"\nResults saved to {output_path}")
        
        return results


class AutoExeLLMRunner:
    """
    Alternative AutoExe runner using LLM for constraint solving.
    
    This approach uses the same prompting strategy as AutoExe's
    LLM-based symbolic execution but without the actual executor.
    Good for when AutoExe binary is not available.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_workers: int = 4,
        api_key: Optional[str] = None
    ):
        self.model = model
        self.max_workers = max_workers
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
                raise ImportError("openai package required for LLM mode")
        return self._client
    
    def _build_autoexe_prompt(
        self,
        code: str,
        func_name: str,
        description: str,
        path_conditions: List[str]
    ) -> str:
        """Build a prompt mimicking AutoExe's approach."""
        path_desc = '\n'.join(f"  {i+1}. {c}" for i, c in enumerate(path_conditions))
        
        return f"""You are a symbolic execution engine. Your task is to find concrete input values that will cause the following Python function to execute through a specific path.

Function: {func_name}
Description: {description}

Code:
```python
{code}
```

Target execution path (the conditions that must be satisfied in order):
{path_desc}

Analyze the path conditions and find concrete input values that will satisfy ALL conditions.
Then generate a test case.

Your response should be a Python test function:
def test_{func_name}():
    solution = Solution()
    # Explain your reasoning in comments
    result = solution.{func_name}(...)  # Fill in actual values
"""
    
    def generate_test(
        self,
        code: str,
        func_name: str,
        description: str,
        path_conditions: List[str]
    ) -> str:
        """Generate test using LLM."""
        prompt = self._build_autoexe_prompt(
            code, func_name, description, path_conditions
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise symbolic execution engine that finds inputs to satisfy path conditions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_completion_tokens=512
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"# Error: {e}"
    
    def run_task(
        self,
        task_data: Dict[str, Any],
        paths_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run LLM-based test generation for a single task."""
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
        """Run on all selected tasks with selection filtering."""
        with open(SELECTED_TASKS_FILE, 'r') as f:
            selection = json.load(f)
        
        # Get valid task_nums for current selection
        valid_task_nums = get_selected_task_nums(selection)
        selection_hash = compute_selection_hash(selection)
        print(f"Selection hash: {selection_hash}, valid tasks: {sorted(valid_task_nums)}")
        
        leetcode_data = read_jsonl(LEETCODE_DATA)
        paths_data = read_jsonl(TARGET_PATHS_DATA)
        
        if output_path is None:
            output_path = RESULTS_DIR / get_output_filename("autoexe_llm", self.model)
        
        # Load existing progress, filtering to only include entries in current selection
        results = []
        completed_indices = set()
        
        if output_path.exists():
            existing = read_jsonl(output_path)
            original_count = len(existing)
            
            # Filter to only keep entries that are in the current selection
            results = [r for r in existing if r.get('task_num') in valid_task_nums]
            filtered_count = original_count - len(results)
            
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} entries not in current selection")
            
            completed_indices = {r['task_num'] for r in results}
            print(f"Resuming from {len(results)} completed tasks (in current selection)")
        
        # Filter tasks to process (exclude already completed)
        all_indices = selection['all_indices']
        tasks_to_process = [
            (leetcode_data[idx], paths_data[idx])
            for idx in all_indices
            if leetcode_data[idx]['task_num'] not in completed_indices
        ]
        
        if not tasks_to_process:
            print("All tasks already completed")
            return results
        
        print(f"Processing {len(tasks_to_process)} tasks with {self.max_workers} workers")
        
        # Thread-safe progress saving
        write_lock = threading.Lock()
        
        def process_task(task_tuple):
            task_data, task_paths = task_tuple
            return self.run_task(task_data, task_paths)
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_task, task_tuple): task_tuple[0]['task_num']
                for task_tuple in tasks_to_process
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="AutoExe LLM generation"):
                task_num = futures[future]
                try:
                    result = future.result()
                    with write_lock:
                        results.append(result)
                        if save_progress:
                            write_jsonl(results, output_path)
                except Exception as e:
                    print(f"\nTask {task_num} failed: {e}")
        
        write_jsonl(results, output_path)
        print(f"\nResults saved to {output_path}")
        
        return results


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="AutoExe runner for path coverage")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--baseline", action="store_true",
                        help="Use baseline mode (--skip-slice)")
    parser.add_argument("--llm-mode", action="store_true",
                        help="Use LLM-only mode (no AutoExe binary)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per execution (default: 120s)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel workers (default: 4)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("AutoExe Runner")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Mode: {'LLM-only' if args.llm_mode else 'Binary' + (' (baseline)' if args.baseline else '')}")
    
    if args.dry_run:
        print("\nDry run mode - showing configuration")
        print(f"AutoExe binary: {AUTOEXE_EXECUTOR}")
        print(f"Binary exists: {AUTOEXE_EXECUTOR.exists()}")
        return
    
    if args.llm_mode:
        runner = AutoExeLLMRunner(model=args.model, max_workers=args.max_workers)
    else:
        runner = AutoExeRunner(
            model=args.model,
            timeout=args.timeout,
            use_baseline=args.baseline,
            max_workers=args.max_workers
        )
    
    results = runner.run_selected_tasks(output_path=args.output)
    print(f"\nCompleted: {len(results)} tasks")


if __name__ == "__main__":
    main()
