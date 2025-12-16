#!/usr/bin/env python3
"""
Zero-Shot Baseline Runner for Path Coverage Benchmark

Directly prompts LLM to generate test cases for target paths,
similar to TestEval's generate_pathcov_openai.py approach.
"""

import os
import sys
import json
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Set

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
TESTEVAL_PROMPTS = benchmark_config.TESTEVAL_PROMPTS
DEFAULT_MODEL = benchmark_config.DEFAULT_MODEL
DEFAULT_TEMPERATURE = benchmark_config.DEFAULT_TEMPERATURE
DEFAULT_MAX_TOKENS = benchmark_config.DEFAULT_MAX_TOKENS
OPENAI_API_KEY = benchmark_config.OPENAI_API_KEY
get_output_filename = benchmark_config.get_output_filename

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


def add_lineno(code: str) -> str:
    """Add line numbers to code."""
    lines = code.split('\n')
    new_code = ''
    for i, line in enumerate(lines):
        new_code += f'{i+1}. {line}\n'
    return new_code


def generate_path_prompt(condition_path: List[str]) -> str:
    """
    Convert condition path to prompt format.
    Similar to TestEval's prompt_utils.generate_path()
    """
    formatted = []
    for i, cond in enumerate(condition_path):
        formatted.append(f"'{cond}'")
    return ' -> '.join(formatted)


def load_prompt_templates() -> tuple:
    """Load prompt templates from TestEval."""
    template_path = TESTEVAL_PROMPTS / "template_path.txt"
    system_path = TESTEVAL_PROMPTS / "system.txt"
    
    with open(template_path, 'r') as f:
        prompt_template = f.read()
    
    with open(system_path, 'r') as f:
        system_template = f.read()
    
    return prompt_template, system_template


def compute_selection_hash(selection: Dict[str, Any]) -> str:
    """Compute a short hash of the task selection for identification."""
    task_nums = sorted([t.get('task_num', 0) for t in selection.get('tasks', [])])
    key = f"{selection.get('seed', 0)}_{task_nums}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def get_selected_task_nums(selection: Dict[str, Any]) -> Set[int]:
    """Extract set of task_nums from selection."""
    return {t.get('task_num') for t in selection.get('tasks', []) if 'task_num' in t}


class ZeroShotRunner:
    """Runner for zero-shot baseline approach."""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_workers: int = 4,
        api_key: Optional[str] = None,
        rate_limit_tier: str = 'tier1'
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
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
        
        # Load templates
        self.prompt_template, system_template = load_prompt_templates()
        self.system_message = system_template.format(lang='python')
    
    def generate_completion(self, prompt: str) -> str:
        """Generate completion using OpenAI API with rate limiting."""
        
        @self.rate_limiter.with_retry
        def _call_api():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens
            )
        
        try:
            response = _call_api()
            return response.choices[0].message.content
        except Exception as e:
            print(f"API error: {e}")
            return f"# Error: {e}"
    
    def run_task(
        self,
        task_data: Dict[str, Any],
        paths_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run zero-shot test generation for a single task.
        
        Returns:
            Dict with task_num, difficulty, func_name, code, and tests
        """
        func_name = task_data['func_name']
        description = task_data['description']
        code = task_data['python_solution']
        difficulty = task_data['difficulty']
        task_num = task_data['task_num']
        task_title = task_data.get('task_title', '')
        
        code_with_lineno = add_lineno(code)
        
        # Get target paths
        condition_paths = paths_data.get('sampled_condition_paths', [])
        
        generated_tests = []
        
        for path_idx, condition_path in enumerate(condition_paths):
            path_prompt = generate_path_prompt(condition_path)
            
            prompt = self.prompt_template.format(
                func_name=func_name,
                description=description,
                program=code_with_lineno,
                path=path_prompt
            )
            
            # Generate test
            generated_test = self.generate_completion(prompt)
            generated_tests.append(generated_test)
            
            print(f"  Path {path_idx + 1}/{len(condition_paths)}: Generated test")
        
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
        Run zero-shot generation on all selected tasks.
        
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
            output_path = RESULTS_DIR / get_output_filename("zero_shot", self.model)
        
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
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Zero-shot generation"):
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
    parser = ArgumentParser(description="Zero-shot baseline for path coverage")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Max parallel workers (default: 4)")
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
    print("Zero-Shot Baseline Runner")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    
    if args.dry_run:
        print("\nDry run mode - showing selected tasks:")
        with open(SELECTED_TASKS_FILE, 'r') as f:
            selection = json.load(f)
        for task in selection['tasks']:
            print(f"  [{task['index']}] {task['task_num']}: {task['task_title']} ({task['num_target_paths']} paths)")
        return
    
    runner = ZeroShotRunner(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_workers=args.max_workers,
        rate_limit_tier=args.rate_limit_tier
    )
    
    results = runner.run_selected_tasks(output_path=args.output)
    
    print(f"\nCompleted: {len(results)} tasks")


if __name__ == "__main__":
    main()
