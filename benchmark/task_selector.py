#!/usr/bin/env python3
"""
Task Selector for Path Coverage Benchmark

Selects Hard (difficulty=3) tasks from the TestEval dataset for benchmarking.
Default: 15 hard tasks only (configurable via NUM_MEDIUM_TASKS and NUM_HARD_TASKS).
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

from config import (
    LEETCODE_DATA,
    TARGET_PATHS_DATA,
    SELECTED_TASKS_FILE,
    NUM_MEDIUM_TASKS,
    NUM_HARD_TASKS,
    DIFFICULTY_MEDIUM,
    DIFFICULTY_HARD,
    RANDOM_SEED,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file and return list of dictionaries."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def select_tasks(
    num_medium: int = NUM_MEDIUM_TASKS,
    num_hard: int = NUM_HARD_TASKS,
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Select tasks for benchmarking.
    
    Args:
        num_medium: Number of medium difficulty tasks to select
        num_hard: Number of hard difficulty tasks to select
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with selected task indices and metadata
    """
    random.seed(seed)
    
    # Load datasets
    print(f"Loading data from {LEETCODE_DATA}...")
    leetcode_data = read_jsonl(LEETCODE_DATA)
    
    print(f"Loading target paths from {TARGET_PATHS_DATA}...")
    paths_data = read_jsonl(TARGET_PATHS_DATA)
    
    # Verify alignment
    assert len(leetcode_data) == len(paths_data), \
        f"Dataset mismatch: {len(leetcode_data)} vs {len(paths_data)}"
    
    # Categorize by difficulty
    medium_indices = []
    hard_indices = []
    
    for idx, task in enumerate(leetcode_data):
        difficulty = task.get('difficulty', 0)
        # Only include tasks that have target paths
        if idx < len(paths_data) and paths_data[idx].get('sampled_condition_paths'):
            if difficulty == DIFFICULTY_MEDIUM:
                medium_indices.append(idx)
            elif difficulty == DIFFICULTY_HARD:
                hard_indices.append(idx)
    
    print(f"Found {len(medium_indices)} Medium tasks")
    print(f"Found {len(hard_indices)} Hard tasks")
    
    # Sample tasks
    if len(medium_indices) < num_medium:
        print(f"Warning: Only {len(medium_indices)} Medium tasks available, using all")
        selected_medium = medium_indices
    else:
        selected_medium = random.sample(medium_indices, num_medium)
    
    if len(hard_indices) < num_hard:
        print(f"Warning: Only {len(hard_indices)} Hard tasks available, using all")
        selected_hard = hard_indices
    else:
        selected_hard = random.sample(hard_indices, num_hard)
    
    # Sort indices for reproducibility
    selected_medium.sort()
    selected_hard.sort()
    
    # Build selection result
    selection = {
        "seed": seed,
        "medium_count": len(selected_medium),
        "hard_count": len(selected_hard),
        "total_count": len(selected_medium) + len(selected_hard),
        "medium_indices": selected_medium,
        "hard_indices": selected_hard,
        "all_indices": sorted(selected_medium + selected_hard),
        "tasks": []
    }
    
    # Add task details
    for idx in selection["all_indices"]:
        task = leetcode_data[idx]
        paths = paths_data[idx]
        selection["tasks"].append({
            "index": idx,
            "task_num": task["task_num"],
            "task_title": task["task_title"],
            "difficulty": task["difficulty"],
            "func_name": task["func_name"],
            "num_target_paths": len(paths.get("sampled_condition_paths", []))
        })
    
    return selection


def save_selection(selection: Dict[str, Any], output_path: Path = SELECTED_TASKS_FILE):
    """Save the task selection to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(selection, f, indent=2)
    print(f"Saved selection to {output_path}")


def load_selection(input_path: Path = SELECTED_TASKS_FILE) -> Dict[str, Any]:
    """Load a previously saved task selection."""
    with open(input_path, 'r') as f:
        return json.load(f)


def print_selection_summary(selection: Dict[str, Any]):
    """Print a summary of the selected tasks."""
    print("\n" + "=" * 60)
    print("SELECTED TASKS SUMMARY")
    print("=" * 60)
    print(f"Random seed: {selection['seed']}")
    print(f"Medium tasks: {selection['medium_count']}")
    print(f"Hard tasks: {selection['hard_count']}")
    print(f"Total tasks: {selection['total_count']}")
    print()
    
    print("Medium Tasks:")
    print("-" * 40)
    for task in selection["tasks"]:
        if task["difficulty"] == DIFFICULTY_MEDIUM:
            print(f"  [{task['index']:3d}] {task['task_num']:4d}: {task['task_title'][:35]:<35} ({task['num_target_paths']} paths)")
    
    print()
    print("Hard Tasks:")
    print("-" * 40)
    for task in selection["tasks"]:
        if task["difficulty"] == DIFFICULTY_HARD:
            print(f"  [{task['index']:3d}] {task['task_num']:4d}: {task['task_title'][:35]:<35} ({task['num_target_paths']} paths)")
    
    print("=" * 60)


def main():
    """Main entry point for task selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select tasks for path coverage benchmark")
    parser.add_argument("--num-medium", type=int, default=NUM_MEDIUM_TASKS,
                        help=f"Number of medium tasks (default: {NUM_MEDIUM_TASKS})")
    parser.add_argument("--num-hard", type=int, default=NUM_HARD_TASKS,
                        help=f"Number of hard tasks (default: {NUM_HARD_TASKS})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed (default: {RANDOM_SEED})")
    parser.add_argument("--output", type=Path, default=SELECTED_TASKS_FILE,
                        help=f"Output file (default: {SELECTED_TASKS_FILE})")
    
    args = parser.parse_args()
    
    # Select tasks
    selection = select_tasks(
        num_medium=args.num_medium,
        num_hard=args.num_hard,
        seed=args.seed
    )
    
    # Save and display
    save_selection(selection, args.output)
    print_selection_summary(selection)


if __name__ == "__main__":
    main()


