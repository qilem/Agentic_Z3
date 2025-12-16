#!/usr/bin/env python3
"""
Path to Precondition Adapter for AutoExe

Converts TestEval path conditions to PRE/POST format expected by AutoExe.
AutoExe uses symbolic execution with assertions to find inputs.
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class AutoExeProgram:
    """Represents a program in AutoExe format."""
    code: str
    entry_function: str
    precondition: str
    postcondition: str


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


def path_conditions_to_assertion(conditions: List[str]) -> str:
    """
    Convert path conditions to a Python assertion string.
    
    Args:
        conditions: List of path condition strings
        
    Returns:
        Python assertion code
    """
    assertions = []
    
    for cond in conditions:
        # Parse "Line N: [NOT] (condition)"
        match = re.match(r"Line \d+:\s*(NOT\s+)?(.*)$", cond)
        if not match:
            continue
        
        negated = match.group(1) is not None
        condition = match.group(2).strip()
        
        # Clean up condition
        if condition.startswith("(") and condition.endswith(")"):
            # Check balance before stripping
            depth = 0
            balanced = True
            for i, c in enumerate(condition):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                if depth == 0 and i < len(condition) - 1:
                    balanced = False
                    break
            if balanced:
                condition = condition[1:-1]
        
        if not condition:
            continue
        
        # Skip complex conditions that can't be easily translated
        # (e.g., those involving iteration, method calls on containers)
        skip_patterns = [
            r'in range\(',
            r'in enumerate\(',
            r'\.items\(\)',
            r'for .* in',
        ]
        
        should_skip = any(re.search(p, condition) for p in skip_patterns)
        if should_skip:
            continue
        
        # Build assertion
        if negated:
            assertions.append(f"not ({condition})")
        else:
            assertions.append(f"({condition})")
    
    if not assertions:
        return "True"
    
    return " and ".join(assertions)


def create_autoexe_wrapper(
    task_data: Dict[str, Any],
    path_conditions: List[str],
    target_result: bool = True
) -> AutoExeProgram:
    """
    Create an AutoExe-compatible program wrapper.
    
    Args:
        task_data: Task data from TestEval
        path_conditions: Target path conditions
        target_result: Whether the postcondition should be SAT (True) or UNSAT (False)
        
    Returns:
        AutoExeProgram with PRE/POST comments
    """
    func_name = task_data.get('func_name', 'solution')
    code = task_data.get('python_solution', '')
    description = task_data.get('description', '')
    
    # Extract function inputs
    inputs = extract_function_inputs(code, func_name)
    
    # Convert path conditions to assertion
    path_assertion = path_conditions_to_assertion(path_conditions)
    
    # Build wrapper code
    # AutoExe expects:
    # - PRE: preconditions on inputs
    # - POST: postcondition to verify (the path we want to cover)
    
    # Create input declarations
    input_vars = [name for name, _ in inputs if name != 'self']
    
    wrapper_code = f'''# Auto-generated wrapper for AutoExe
# Function: {func_name}
# Description: {description[:100]}...

{code}

def entry({', '.join(input_vars)}):
    """
    Entry point for AutoExe symbolic execution.
    Find inputs that satisfy the target execution path.
    """
    # PRE: Input constraints (basic validity)
    # Types are inferred by AutoExe
    
    solution = Solution()
    result = solution.{func_name}({', '.join(input_vars)})
    
    # POST: Path condition assertion
    # AutoExe will find inputs that make this assertion hold
    assert {path_assertion}  # POST
'''
    
    return AutoExeProgram(
        code=wrapper_code,
        entry_function="entry",
        precondition="True",
        postcondition=path_assertion
    )


def create_simple_constraint_program(
    path_conditions: List[str],
    func_name: str = "check_path"
) -> str:
    """
    Create a simple Python program that encodes path constraints
    for AutoExe to solve.
    
    This is a simpler approach that just encodes the constraints
    directly without running the actual function.
    """
    # Parse conditions into variable constraints
    constraints = []
    variables = set()
    
    for cond in path_conditions:
        match = re.match(r"Line \d+:\s*(NOT\s+)?(.*)$", cond)
        if not match:
            continue
        
        negated = match.group(1) is not None
        condition = match.group(2).strip()
        
        # Clean parentheses
        if condition.startswith("(") and condition.endswith(")"):
            depth = 0
            balanced = True
            for i, c in enumerate(condition):
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                if depth == 0 and i < len(condition) - 1:
                    balanced = False
                    break
            if balanced:
                condition = condition[1:-1]
        
        if not condition:
            continue
        
        # Extract variable names (simple approach)
        var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        found_vars = re.findall(var_pattern, condition)
        keywords = {'in', 'and', 'or', 'not', 'True', 'False', 'None',
                   'range', 'len', 'enumerate', 'for', 'if', 'else'}
        for v in found_vars:
            if v not in keywords and not v[0].isupper():
                variables.add(v)
        
        # Build constraint
        if negated:
            constraints.append(f"not ({condition})")
        else:
            constraints.append(f"({condition})")
    
    # Generate program
    var_list = sorted(variables)
    program = f'''# Path constraint solver for AutoExe
# Variables: {', '.join(var_list)}

def {func_name}({', '.join(var_list)}):
    """
    Entry point encoding path constraints.
    PRE: inputs are integers (symbolic)
    POST: all path constraints satisfied
    """
    # POST: Path constraints
    assert {' and '.join(constraints) if constraints else 'True'}
'''
    
    return program


def save_autoexe_program(program: AutoExeProgram, output_path: str):
    """Save an AutoExe program to file."""
    with open(output_path, 'w') as f:
        f.write(program.code)


def test_conversion():
    """Test the path to precondition conversion."""
    test_conditions = [
        "Line 26: (l < r)",
        "Line 28: NOT (summ == 0)",
        "Line 36: (summ < 0)",
    ]
    
    print("Testing path to precondition conversion:")
    print("=" * 60)
    
    assertion = path_conditions_to_assertion(test_conditions)
    print(f"Conditions: {test_conditions}")
    print(f"Assertion: {assertion}")
    
    print("\n" + "=" * 60)
    print("Simple constraint program:")
    print(create_simple_constraint_program(test_conditions))


if __name__ == "__main__":
    test_conversion()







