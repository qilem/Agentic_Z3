#!/usr/bin/env python3
"""
Vanilla Z3 Baseline Runner for Path Coverage Benchmark

A naive "Text-to-Z3" baseline: prompts an LLM once to translate Python code
and path constraints into a Z3 script, executes it, and extracts a model.

This represents the simplest possible approach - no iterative refinement,
no probing, no diagnosis loop, no TTRL. Just one prompt and one execution.
"""

import os
import sys
import json
import re
import threading
import subprocess
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple, Set

# Import rate limiter
try:
    from rate_limiter import RateLimiter, RateLimitConfig, TIER_CONFIGS
except ImportError:
    import importlib.util
    rl_path = Path(__file__).parent.parent / "rate_limiter.py"
    spec = importlib.util.spec_from_file_location("rate_limiter", rl_path)
    rate_limiter_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rate_limiter_module)
    RateLimiter = rate_limiter_module.RateLimiter
    RateLimitConfig = rate_limiter_module.RateLimitConfig
    TIER_CONFIGS = rate_limiter_module.TIER_CONFIGS

# Add benchmark directory to path first (for our config)
benchmark_dir = str(Path(__file__).parent.parent)
if benchmark_dir not in sys.path:
    sys.path.insert(0, benchmark_dir)

# Add project root for agentic_z3 imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
DEFAULT_MODEL = benchmark_config.DEFAULT_MODEL
DEFAULT_TEMPERATURE = benchmark_config.DEFAULT_TEMPERATURE
# Use higher max tokens for Z3 script generation (complex code)
DEFAULT_MAX_TOKENS = 2048  # Increased from benchmark_config default
OPENAI_API_KEY = benchmark_config.OPENAI_API_KEY
get_output_filename = benchmark_config.get_output_filename

# Try to import Z3Executor from agentic_z3
Z3Executor = None
ExecutionResult = None

def get_z3_executor():
    """Get Z3Executor, importing if needed."""
    global Z3Executor, ExecutionResult
    if Z3Executor is None:
        try:
            from agentic_z3.tools.z3_executor import Z3Executor as _Z3Executor
            from agentic_z3.tools.z3_executor import ExecutionResult as _ExecutionResult
            Z3Executor = _Z3Executor
            ExecutionResult = _ExecutionResult
        except ImportError:
            pass
    return Z3Executor

# Import OpenAI client (deferred to allow dry-run without API)
OpenAI = None

def get_openai_client():
    """Get OpenAI client, importing if needed."""
    global OpenAI
    if OpenAI is None:
        try:
            from openai import OpenAI as _OpenAI
            OpenAI = _OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    return OpenAI


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


def add_lineno(code: str) -> str:
    """Add line numbers to code."""
    lines = code.split('\n')
    new_code = ''
    for i, line in enumerate(lines):
        new_code += f'{i+1}. {line}\n'
    return new_code


def format_path_constraints(condition_path: List[str]) -> str:
    """Format path constraints for the prompt."""
    formatted = []
    for i, cond in enumerate(condition_path):
        formatted.append(f"{i+1}. {cond}")
    return '\n'.join(formatted)


def extract_function_inputs(code: str, func_name: str) -> List[Tuple[str, str]]:
    """
    Extract input parameters from a function definition.
    
    Returns:
        List of (name, type_hint) tuples
    """
    # Look for the function definition
    pattern = rf"def\s+{func_name}\s*\((.*?)\)"
    match = re.search(pattern, code, re.DOTALL)
    
    if not match:
        return []
    
    params_str = match.group(1)
    params = []
    
    # Parse parameters (handle complex type hints)
    current_param = ""
    bracket_depth = 0
    
    for char in params_str + ",":
        if char in "([{":
            bracket_depth += 1
            current_param += char
        elif char in ")]}":
            bracket_depth -= 1
            current_param += char
        elif char == "," and bracket_depth == 0:
            param = current_param.strip()
            if param and param != "self":
                # Parse name: type
                if ":" in param:
                    name, type_hint = param.split(":", 1)
                    params.append((name.strip(), type_hint.strip()))
                else:
                    params.append((param, "Any"))
            current_param = ""
        else:
            current_param += char
    
    return params


def parse_model_from_output(output: str) -> Dict[str, str]:
    """
    Parse variable assignments from Z3 execution output.
    
    Handles formats like:
    - Model: x=5, y=3
    - x = 5
    - [x = 5, y = 3]
    """
    model = {}
    
    # Try to find "Model:" line
    model_match = re.search(r'Model:\s*(.+?)(?:\n|$)', output, re.IGNORECASE)
    if model_match:
        model_str = model_match.group(1).strip()
        # Parse comma-separated assignments: x=5, y=3
        for part in model_str.split(','):
            part = part.strip()
            if '=' in part:
                name, val = part.split('=', 1)
                model[name.strip()] = val.strip()
        if model:
            return model
    
    # Try to find assignments in [x = 5, y = 3] format
    bracket_match = re.search(r'\[([^\]]+)\]', output)
    if bracket_match:
        content = bracket_match.group(1)
        for part in content.split(','):
            part = part.strip()
            if '=' in part:
                name, val = part.split('=', 1)
                model[name.strip()] = val.strip()
        if model:
            return model
    
    # Try to find individual lines with x = value
    for line in output.split('\n'):
        line = line.strip()
        # Match patterns like: x = 5, var_name = 123
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
        if match:
            name = match.group(1)
            val = match.group(2).strip()
            # Skip if it looks like Python code (contains operators, parentheses for calls)
            if not any(op in val for op in ['(', ')', '+', '-', '*', '/', 'import', 'from']):
                model[name] = val
    
    return model


def get_default_value(type_hint: str) -> str:
    """Get a safe default value for a given type hint."""
    type_hint = type_hint.strip().lower()
    
    if 'list' in type_hint:
        return '[]'
    elif 'str' in type_hint:
        return "''"
    elif 'bool' in type_hint:
        return 'False'
    elif 'float' in type_hint:
        return '0.0'
    elif 'int' in type_hint:
        return '0'
    elif 'dict' in type_hint:
        return '{}'
    elif 'optional' in type_hint:
        return 'None'
    else:
        return '0'  # Default fallback


def execute_z3_code_fallback(code: str, timeout_sec: int = 30) -> Tuple[str, str, str]:
    """
    Fallback execution of Z3 code via subprocess.
    
    Returns:
        Tuple of (status, stdout, stderr)
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_sec
            )
            
            stdout = result.stdout
            stderr = result.stderr
            
            # Determine status from output
            combined = (stdout + stderr).lower()
            if 'error' in combined or 'traceback' in combined:
                status = 'error'
            elif 'unsat' in combined:
                status = 'unsat'
            elif 'sat' in combined:
                status = 'sat'
            else:
                status = 'unknown'
            
            return status, stdout, stderr
            
        except subprocess.TimeoutExpired:
            return 'timeout', '', 'Execution timed out'
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
                
    except Exception as e:
        return 'error', '', str(e)


class VanillaZ3Runner:
    """
    Runner for Vanilla Z3 baseline approach.
    
    This baseline:
    1. Prompts an LLM once to translate code+path into a Z3 script
    2. Executes the Z3 script
    3. Parses the model and generates a test function
    
    No iterative refinement, no probing, no diagnosis - just one shot.
    """
    
    SYSTEM_PROMPT = """You are a translator. Your task is to convert Python code and path constraints into a standalone Z3 solver script.

Output ONLY valid Python code using the z3-solver library. No explanations.

The script must:
1. Import z3 (from z3 import *)
2. Declare symbolic variables for the function inputs
3. Add constraints corresponding to the path conditions
4. Call solver.check() and print the result
5. If sat, print the model values

Output format requirements:
- After solver.check(), print "Result: sat" or "Result: unsat" or "Result: unknown"
- If sat, print "Model: var1=value1, var2=value2, ..." on a single line"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_workers: int = 4,
        z3_timeout: int = 10000,
        api_key: Optional[str] = None,
        rate_limit_tier: str = 'tier1'
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.z3_timeout = z3_timeout
        
        # Setup rate limiting
        config = TIER_CONFIGS.get(rate_limit_tier, TIER_CONFIGS['tier1'])
        self.rate_limiter = RateLimiter.get_instance(config)
        
        # Use recommended max_workers based on rate limit tier
        if max_workers > config.recommended_max_workers:
            print(f"Warning: Reducing max_workers from {max_workers} to {config.recommended_max_workers} for {rate_limit_tier} rate limits")
            max_workers = config.recommended_max_workers
        self.max_workers = max_workers
        
        print(f"Rate limiting: {rate_limit_tier} ({config.tokens_per_minute} TPM, {max_workers} workers)")
        
        # Initialize OpenAI client
        api_key = api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        OpenAIClient = get_openai_client()
        self.client = OpenAIClient(api_key=api_key)
        
        # Try to initialize Z3Executor
        Z3ExecutorClass = get_z3_executor()
        if Z3ExecutorClass:
            self.executor = Z3ExecutorClass(default_timeout=z3_timeout)
        else:
            self.executor = None
            print("Warning: Z3Executor not available, using fallback subprocess execution")
    
    def build_prompt(
        self,
        code: str,
        func_name: str,
        description: str,
        condition_path: List[str],
        input_params: List[Tuple[str, str]]
    ) -> str:
        """Build the translation prompt for the LLM."""
        
        # Format input parameters
        params_desc = []
        for name, type_hint in input_params:
            params_desc.append(f"  - {name}: {type_hint}")
        params_str = '\n'.join(params_desc) if params_desc else "  (no parameters)"
        
        # Format path constraints
        path_str = format_path_constraints(condition_path)
        
        prompt = f"""Translate this Python function and target execution path into a Z3 solver script.

## Function: {func_name}

### Description:
{description}

### Code:
```python
{code}
```

### Input Parameters:
{params_str}

### Target Path Constraints (must ALL be satisfied):
{path_str}

Generate a Z3 script that finds input values satisfying all path constraints.
Focus on the INPUT PARAMETERS - find values for them that make the path constraints true."""

        return prompt
    
    def generate_z3_script(self, prompt: str) -> str:
        """Generate Z3 script using OpenAI API with rate limiting."""
        
        @self.rate_limiter.with_retry
        def _call_api():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens
            )
        
        try:
            response = _call_api()
            content = response.choices[0].message.content
            
            # Extract code from markdown if present
            code_match = re.search(r'```(?:python)?\s*(.*?)```', content, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            
            return content.strip()
            
        except Exception as e:
            print(f"API error: {e}")
            return f"# Error generating Z3 script: {e}"
    
    def execute_z3_script(self, code: str) -> Tuple[str, Dict[str, str], str, str]:
        """
        Execute the Z3 script and extract the model.
        
        Uses non-intrusive execution (no assert_and_track preprocessing).
        
        Returns:
            Tuple of (status, model_dict, stdout, stderr)
        """
        if self.executor:
            # Use check_sat_with_timeout which does NOT preprocess for tracking
            # This avoids breaking LLM-generated Z3 code that uses solver.add()
            result = self.executor.check_sat_with_timeout(code, timeout_ms=self.z3_timeout)
            
            # Parse model from output
            model = {}
            if result.status == 'sat':
                if result.model:
                    # Try to parse the model string
                    model = parse_model_from_output(f"Model: {result.model}")
                if not model:
                    model = parse_model_from_output(result.stdout)
            
            return result.status, model, result.stdout, result.stderr
        else:
            # Fallback execution
            status, stdout, stderr = execute_z3_code_fallback(code, timeout_sec=self.z3_timeout // 1000 + 5)
            
            model = {}
            if status == 'sat':
                model = parse_model_from_output(stdout)
            
            return status, model, stdout, stderr
    
    def generate_test_function(
        self,
        func_name: str,
        input_params: List[Tuple[str, str]],
        status: str,
        model: Dict[str, str],
        z3_script: str
    ) -> str:
        """
        Generate a test function from the Z3 result.
        
        Args:
            func_name: Name of the function to test
            input_params: List of (param_name, type_hint) tuples
            status: Z3 result status (sat/unsat/unknown/error)
            model: Dictionary of variable assignments from Z3
            z3_script: The generated Z3 script (for debugging)
            
        Returns:
            A test function string compatible with the evaluator
        """
        if status != 'sat' or not model:
            # Generate a test that notes the failure
            return f'''def test_{func_name}():
    solution = Solution()
    # Vanilla Z3 status: {status}
    # Could not find satisfying input - using defaults
    pass
'''
        
        # Build argument list from model
        args = []
        missing_params = []
        
        for param_name, type_hint in input_params:
            if param_name in model:
                value = model[param_name]
                args.append(value)
            else:
                # Try common variations
                found = False
                for model_var, model_val in model.items():
                    # Check if model var matches param (case-insensitive or partial)
                    if model_var.lower() == param_name.lower():
                        args.append(model_val)
                        found = True
                        break
                
                if not found:
                    # Use default value
                    default = get_default_value(type_hint)
                    args.append(default)
                    missing_params.append(param_name)
        
        args_str = ', '.join(args)
        
        # Build comment about model
        model_comment = ', '.join(f"{k}={v}" for k, v in model.items())
        
        test_code = f'''def test_{func_name}():
    solution = Solution()
    # Vanilla Z3 status: {status}
    # Model: {model_comment}'''
        
        if missing_params:
            test_code += f'\n    # Note: defaults used for: {", ".join(missing_params)}'
        
        test_code += f'''
    result = solution.{func_name}({args_str})
'''
        
        return test_code
    
    def run_task(
        self,
        task_data: Dict[str, Any],
        paths_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run Vanilla Z3 test generation for a single task.
        
        Returns:
            Dict with task_num, difficulty, func_name, code, tests, and debug info
        """
        func_name = task_data['func_name']
        description = task_data['description']
        code = task_data['python_solution']
        difficulty = task_data['difficulty']
        task_num = task_data['task_num']
        task_title = task_data.get('task_title', '')
        
        # Extract input parameters
        input_params = extract_function_inputs(code, func_name)
        
        # Get target paths
        condition_paths = paths_data.get('sampled_condition_paths', [])
        
        generated_tests = []
        debug_info = []  # Store debug info for each path
        
        for path_idx, condition_path in enumerate(condition_paths):
            # Build prompt
            prompt = self.build_prompt(
                code, func_name, description, condition_path, input_params
            )
            
            # Generate Z3 script (rate limiting and retry handled internally)
            z3_script = self.generate_z3_script(prompt)
            
            # Execute Z3 script
            status, model, stdout, stderr = self.execute_z3_script(z3_script)
            
            # Generate test function
            test = self.generate_test_function(
                func_name, input_params, status, model, z3_script
            )
            generated_tests.append(test)
            
            # Store debug info for this path
            debug_info.append({
                'path_idx': path_idx,
                'z3_status': status,
                'z3_model': model,
                'z3_script': z3_script[:2000] if z3_script else None,  # Truncate for storage
                'z3_stdout': stdout[:1000] if stdout else None,
                'z3_stderr': stderr[:500] if stderr else None
            })
            
            print(f"  Path {path_idx + 1}/{len(condition_paths)}: {status}")
        
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
        Run Vanilla Z3 generation on all selected tasks.
        
        Filters existing results to only include tasks in current selection,
        preventing stale/mixed results from different selections.
        
        Args:
            output_path: Path to save results
            save_progress: Whether to save after each task
            
        Returns:
            List of results for all tasks
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
        if output_path is None:
            output_path = RESULTS_DIR / get_output_filename("vanilla_z3", self.model)
        
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
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Vanilla Z3 generation"):
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


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Vanilla Z3 baseline for path coverage")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel workers (default: 4)")
    parser.add_argument("--z3-timeout", type=int, default=10000,
                        help="Z3 timeout in milliseconds (default: 10000)")
    parser.add_argument("--rate-limit-tier", type=str, default='tier1',
                        choices=['free', 'tier1', 'tier2', 'tier3'],
                        help="OpenAI API tier for rate limiting (default: tier1)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making API calls")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("Vanilla Z3 Baseline Runner")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Z3 timeout: {args.z3_timeout}ms")
    
    if args.dry_run:
        print("\nDry run mode - showing selected tasks:")
        with open(SELECTED_TASKS_FILE, 'r') as f:
            selection = json.load(f)
        for task in selection['tasks']:
            print(f"  [{task['index']}] {task['task_num']}: {task['task_title']} ({task['num_target_paths']} paths)")
        return
    
    runner = VanillaZ3Runner(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        z3_timeout=args.z3_timeout,
        rate_limit_tier=args.rate_limit_tier
    )
    
    results = runner.run_selected_tasks(output_path=args.output)
    
    print(f"\nCompleted: {len(results)} tasks")


if __name__ == "__main__":
    main()

