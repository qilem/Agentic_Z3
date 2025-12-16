#!/usr/bin/env python3
"""
Path to SMT Adapter

Converts TestEval path conditions to Z3 constraints for Agentic-Z3.
This module parses Python-style path conditions and translates them
into Z3-compatible constraint representations.
"""

import ast
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class PathCondition:
    """Represents a single path condition."""
    line_number: int
    condition: str
    negated: bool = False
    original: str = ""


@dataclass
class SMTConstraint:
    """Represents an SMT constraint derived from path conditions."""
    z3_code: str
    variables: Dict[str, str] = field(default_factory=dict)  # name -> type
    description: str = ""


@dataclass
class PathConstraintSet:
    """Collection of constraints for a single execution path."""
    constraints: List[SMTConstraint] = field(default_factory=list)
    all_variables: Dict[str, str] = field(default_factory=dict)
    path_description: str = ""
    
    def to_z3_problem(self) -> str:
        """Generate a complete Z3 problem description."""
        var_decls = []
        for name, vtype in self.all_variables.items():
            var_decls.append(f"  {name} = {vtype}('{name}')")
        
        constraints_code = []
        for c in self.constraints:
            if c.z3_code:
                constraints_code.append(f"  solver.add({c.z3_code})")
        
        return f"""
from z3 import *

solver = Solver()

# Variable declarations
{chr(10).join(var_decls)}

# Path constraints
{chr(10).join(constraints_code)}

# Check satisfiability
result = solver.check()
if result == sat:
    model = solver.model()
    print("SAT")
    for var in [{', '.join(self.all_variables.keys())}]:
        print(f"{{var}} = {{model[var]}}")
else:
    print(result)
"""


def is_loop_header_condition(condition: str) -> bool:
    """
    Check if a condition is a loop header that should be skipped.
    
    Loop headers like "(i, c in enumerate(s))" or "(i in range(n))"
    are not constraints on INPUT variables - they just describe
    loop iteration and should be filtered out.
    """
    loop_patterns = [
        r'^\s*\w+\s+in\s+range\(',        # i in range(...)
        r'^\s*\w+\s*,\s*\w+\s+in\s+enumerate\(',  # i, c in enumerate(...)
        r'^\s*for\s+',                      # for ... in ...
        r'^\s*\w+\s+in\s+\w+\s*$',          # simple "x in y" (usually iteration)
        r'^\s*\w+\s+in\s+\w+\.\w+\(\)',     # x in dict.items()
    ]
    
    for pattern in loop_patterns:
        if re.search(pattern, condition):
            return True
    return False


def is_internal_variable_condition(condition: str, input_vars: List[str] = None) -> bool:
    """
    Check if a condition references only internal/loop variables.
    
    Conditions like "seenE or not seenNum" reference internal state
    variables that are not INPUT parameters, and cannot be directly
    constrained in the SMT problem.
    
    Also filters out conditions that use loop index variables (i, j, k, etc.)
    as these often cause "NameError: i is not defined" when the LLM tries
    to use them in ForAll without proper declaration.
    
    Args:
        condition: The condition string
        input_vars: Known input variable names (if available)
        
    Returns:
        True if the condition appears to use only internal variables
    """
    # Common internal variable patterns (loop indices, state flags)
    internal_patterns = [
        r'^seen\w*',    # seenE, seenNum, etc.
        r'^visited',
        r'^count',
        r'^flag',
        r'^result',
        r'^temp',
        r'^curr',
        r'^prev',
        r'^dp$',        # dynamic programming state
        r'^dp\d*$',
        r'^memo',
        r'^cache',
        r'^ans$',
        r'^ret$',
        r'^res$',
    ]
    
    # Loop index variables - these cause "NameError: i is not defined" 
    # when used with ForAll without proper declaration
    loop_index_patterns = [
        r'^[ijklmnpq]$',  # Common single-letter loop indices
        r'^lo$',          # lo, hi for binary search
        r'^hi$',
        r'^mid$',
        r'^left$',        # When used as loop index (not input param)
        r'^right$',
        r'^idx$',
        r'^index$',
        r'^pos$',
    ]
    
    # Extract variable names from condition
    var_names = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', condition))
    builtins = {'True', 'False', 'None', 'and', 'or', 'not', 'in',
                'len', 'range', 'enumerate', 'str', 'int', 'float', 'bool',
                'isdigit', 'isalpha', 'isalnum', 'isupper', 'islower',
                'ord', 'chr', 'abs', 'max', 'min', 'sum'}
    var_names = var_names - builtins
    
    if not var_names:
        return False
    
    # If input_vars is provided, check if ANY variable is NOT in input_vars
    # This is a strong filter: if condition uses variables we don't know about,
    # it's likely internal computation
    if input_vars:
        input_vars_set = set(input_vars)
        # If ALL variables in condition are unknown (not in input_vars), skip
        if not any(v in input_vars_set for v in var_names):
            return True
        # If condition contains loop indices AND unknown vars, it's likely internal
        has_loop_idx = any(
            any(re.match(p, v, re.IGNORECASE) for p in loop_index_patterns)
            for v in var_names
        )
        if has_loop_idx:
            # Check if there's at least one input var - if not, skip
            if not any(v in input_vars_set for v in var_names):
                return True
    
    # Check if all variables match internal patterns
    all_internal = all(
        any(re.match(p, v, re.IGNORECASE) for p in internal_patterns)
        for v in var_names
    )
    if all_internal:
        return True
    
    # Check if condition ONLY contains loop index variables (common failure mode)
    all_loop_idx = all(
        any(re.match(p, v, re.IGNORECASE) for p in loop_index_patterns)
        for v in var_names
    )
    if all_loop_idx and len(var_names) > 0:
        return True
    
    return False


def parse_path_condition(condition_str: str, input_vars: List[str] = None) -> PathCondition:
    """
    Parse a path condition string from TestEval format.
    
    Example inputs:
        "Line 26: (l < r)"
        "Line 28: NOT (summ == 0)"
        "Line 17: (c in s)"
    
    Also filters out:
        - Empty conditions: "Line 35: "
        - Loop headers: "Line 21: (i, c in enumerate(s))"
        - Pure internal variable conditions: "Line 27: (seenE or not seenNum)"
        
    Args:
        condition_str: The condition string to parse
        input_vars: Known input variable names for better filtering
    """
    # Pattern: "Line N: [NOT] (condition)" or "Line N: "
    match = re.match(r"Line (\d+):\s*(NOT\s+)?(.*)$", condition_str)
    
    if not match:
        return PathCondition(
            line_number=0,
            condition=condition_str,
            negated=False,
            original=condition_str
        )
    
    line_num = int(match.group(1))
    negated = match.group(2) is not None
    condition = match.group(3).strip() if match.group(3) else ""
    
    # Filter out empty conditions
    if not condition or condition.isspace():
        return PathCondition(
            line_number=line_num,
            condition="",  # Empty = will be skipped
            negated=negated,
            original=condition_str
        )
    
    # Remove outer parentheses if present (balanced)
    while condition.startswith("(") and condition.endswith(")"):
        # Check if parentheses are balanced
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
        if balanced and depth == 0:
            condition = condition[1:-1]
        else:
            break
    
    # Filter out loop headers (not real constraints)
    if is_loop_header_condition(condition):
        return PathCondition(
            line_number=line_num,
            condition="",  # Empty = will be skipped
            negated=negated,
            original=condition_str
        )
    
    # Filter out pure internal variable conditions
    if is_internal_variable_condition(condition, input_vars=input_vars):
        return PathCondition(
            line_number=line_num,
            condition="",  # Empty = will be skipped
            negated=negated,
            original=condition_str
        )
    
    return PathCondition(
        line_number=line_num,
        condition=condition,
        negated=negated,
        original=condition_str
    )


def infer_variable_type(var_name: str, context: str = "") -> str:
    """
    Infer Z3 type for a variable based on naming conventions and context.
    
    Supports: Int, Real, Bool, String
    """
    # Common integer variable names
    int_patterns = [
        r"^[ijklmn]$",  # loop indices
        r"^num",  # numbers
        r"^count",
        r"^len",
        r"^size",
        r"^idx",
        r"^index",
        r"^sum",
        r"^total",
        r"^max",
        r"^min",
        r"^left",
        r"^right",
        r"^[lr]$",  # l, r for left/right
        r"^start",
        r"^end",
        r"^row",
        r"^col",
        r"^x$",
        r"^y$",
        r"^dx",
        r"^dy",
    ]
    
    # Check if integer
    for pattern in int_patterns:
        if re.match(pattern, var_name, re.IGNORECASE):
            return "Int"
    
    # Boolean patterns
    bool_patterns = [
        r"^is",
        r"^has",
        r"^can",
        r"^should",
        r"^flag",
        r"^seen",
        r"^visited",
    ]
    
    for pattern in bool_patterns:
        if re.match(pattern, var_name, re.IGNORECASE):
            return "Bool"
    
    # String/char patterns
    string_patterns = [
        r"^s$",         # common string variable
        r"^str",
        r"^text",
        r"^word",
        r"^char",
        r"^[c]$",       # single char variable
        r"^prefix",
        r"^suffix",
        r"^pattern",
    ]
    
    for pattern in string_patterns:
        if re.match(pattern, var_name, re.IGNORECASE):
            return "String"
    
    # Check context for string-related operations
    if context:
        string_ops = ["isdigit", "isalpha", "isalnum", "isupper", "islower",
                      "StringVal", "charAt", ".at("]
        if any(op in context for op in string_ops):
            return "String"
    
    # Default to Int for unknown
    return "Int"


def extract_variables_from_condition(condition: str) -> Dict[str, str]:
    """
    Extract variable names and infer their types from a condition string.
    """
    variables = {}
    
    # Skip if condition is empty or just whitespace
    if not condition or not condition.strip():
        return variables
    
    try:
        # Try to parse as Python AST
        tree = ast.parse(condition, mode='eval')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name = node.id
                # Skip built-in names and common function names
                if var_name not in ['True', 'False', 'None', 'len', 'range', 
                                     'enumerate', 'list', 'dict', 'set', 'str',
                                     'int', 'float', 'bool', 'max', 'min', 'abs',
                                     'sum', 'sorted', 'reversed']:
                    variables[var_name] = infer_variable_type(var_name, condition)
    except SyntaxError:
        # Fall back to regex-based extraction
        # Match word characters that look like variable names
        potential_vars = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', condition)
        
        builtins = {'True', 'False', 'None', 'len', 'range', 'enumerate', 
                    'list', 'dict', 'set', 'str', 'int', 'float', 'bool',
                    'max', 'min', 'abs', 'sum', 'sorted', 'reversed', 'in',
                    'and', 'or', 'not', 'for', 'if', 'else', 'elif'}
        
        for var in potential_vars:
            if var not in builtins and not var[0].isupper():
                variables[var] = infer_variable_type(var, condition)
    
    return variables


def python_condition_to_z3(condition: str, negated: bool = False) -> str:
    """
    Convert a Python condition to Z3 syntax.
    
    This handles common patterns including:
    - Basic comparisons (==, !=, <, >, <=, >=)
    - String character checks (c == '.', c.isdigit(), etc.)
    - Boolean operations (and, or, not)
    - Range constraints
    
    Complex conditions may need manual adjustment.
    """
    if not condition or not condition.strip():
        return ""
    
    # Handle common patterns
    z3_condition = condition
    
    # =========================================================================
    # STRING PREDICATES - Common in TestEval string-heavy tasks
    # =========================================================================
    
    # Handle c.isdigit() -> CharIsDigit(c) (conceptual - needs Z3 string theory)
    # For now, translate to a constraint on char code range: '0' <= c <= '9'
    isdigit_pattern = r'(\w+)\.isdigit\(\)'
    if re.search(isdigit_pattern, z3_condition):
        match = re.search(isdigit_pattern, z3_condition)
        if match:
            char_var = match.group(1)
            # In Z3 String theory, we'd use StrToCode and compare to ASCII range
            # For now, express as a conceptual constraint the LLM can translate
            z3_condition = re.sub(
                isdigit_pattern,
                f'And(StrToCode({char_var}) >= 48, StrToCode({char_var}) <= 57)',
                z3_condition
            )
    
    # Handle c.isalpha() -> char is a letter
    isalpha_pattern = r'(\w+)\.isalpha\(\)'
    if re.search(isalpha_pattern, z3_condition):
        match = re.search(isalpha_pattern, z3_condition)
        if match:
            char_var = match.group(1)
            z3_condition = re.sub(
                isalpha_pattern,
                f'Or(And(StrToCode({char_var}) >= 65, StrToCode({char_var}) <= 90), '
                f'And(StrToCode({char_var}) >= 97, StrToCode({char_var}) <= 122))',
                z3_condition
            )
    
    # Handle character equality: c == '.' or c == 'e'
    # Convert to Z3 String comparison using s.at(i) pattern
    char_eq_pattern = r"(\w+)\s*(==|!=)\s*'([^'])'"
    for match in re.finditer(char_eq_pattern, z3_condition):
        var = match.group(1)
        op = match.group(2)
        char = match.group(3)
        # Replace with StringVal comparison (assuming var is a single char)
        replacement = f'{var} {op} StringVal("{char}")'
        z3_condition = z3_condition.replace(match.group(0), replacement)
    
    # Handle string 'in' for character sets: c in '+-'
    char_in_pattern = r"(\w+)\s+in\s+'([^']+)'"
    for match in re.finditer(char_in_pattern, z3_condition):
        var = match.group(1)
        chars = match.group(2)
        # Convert to Or of equalities
        char_checks = [f'{var} == StringVal("{c}")' for c in chars]
        replacement = f'Or({", ".join(char_checks)})'
        z3_condition = z3_condition.replace(match.group(0), replacement)
    
    # =========================================================================
    # RANGE AND LOOP PATTERNS
    # =========================================================================
    
    # Handle 'in range(...)' patterns - these are loop entry conditions
    range_pattern = r'\((\w+)\s+in\s+range\((.*?)\)\)'
    if re.search(range_pattern, z3_condition):
        # Convert "i in range(n)" to "And(i >= 0, i < n)"
        match = re.search(range_pattern, z3_condition)
        if match:
            var = match.group(1)
            args = match.group(2).split(',')
            if len(args) == 1:
                # range(stop) -> var >= 0 and var < stop
                z3_condition = f"And({var} >= 0, {var} < {args[0].strip()})"
            elif len(args) == 2:
                # range(start, stop) -> var >= start and var < stop
                z3_condition = f"And({var} >= {args[0].strip()}, {var} < {args[1].strip()})"
            else:
                # range(start, stop, step) - complex, simplify
                z3_condition = f"And({var} >= {args[0].strip()}, {var} < {args[1].strip()})"
    
    # Handle enumerate patterns
    enum_pattern = r'\((\w+),\s*(\w+)\s+in\s+enumerate\(.*?\)\)'
    if re.search(enum_pattern, z3_condition):
        match = re.search(enum_pattern, z3_condition)
        if match:
            idx_var = match.group(1)
            # Just constrain the index
            z3_condition = f"{idx_var} >= 0"
    
    # Handle 'in' for membership - convert to disjunction or just True for now
    simple_in_pattern = r'(\w+)\s+in\s+(\w+)'
    if ' in ' in z3_condition and 'range' not in z3_condition and 'StringVal' not in z3_condition:
        # For simple membership, we can't easily translate without knowing the collection
        # Just return True as a placeholder
        z3_condition = "True"
    
    # =========================================================================
    # ARRAY/LIST INDEXING
    # =========================================================================
    
    # Handle array/list indexing - simplify to existence constraint
    if '[' in z3_condition and ']' in z3_condition:
        # Complex indexing - simplify
        # For now, just extract comparisons and ignore indexing
        simple_cmp = re.search(r'(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+)', z3_condition)
        if simple_cmp:
            # Keep the comparison structure but note it's simplified
            pass  # Keep as is for now
    
    # =========================================================================
    # APPLY NEGATION
    # =========================================================================
    
    if negated:
        z3_condition = f"Not({z3_condition})"
    
    return z3_condition


def convert_path_to_smt(
    condition_path: List[str],
    func_signature: str = "",
    code: str = "",
    verbose: bool = False,
    input_vars: List[str] = None
) -> PathConstraintSet:
    """
    Convert a sequence of path conditions to SMT constraints.
    
    This function:
    1. Parses each condition string
    2. Filters out loop headers, empty conditions, and internal-only conditions
    3. Converts remaining conditions to Z3 syntax
    4. Extracts and types variables
    
    Args:
        condition_path: List of condition strings from sampled_condition_paths
        func_signature: Function signature for type hints
        code: Source code for additional context
        verbose: If True, print info about filtered conditions
        input_vars: Known input variable names for better filtering
        
    Returns:
        PathConstraintSet containing Z3 constraints
    """
    result = PathConstraintSet()
    result.path_description = " -> ".join(condition_path)
    
    filtered_count = 0
    
    for cond_str in condition_path:
        parsed = parse_path_condition(cond_str, input_vars=input_vars)
        
        # Skip empty/filtered conditions
        if not parsed.condition:
            filtered_count += 1
            if verbose:
                print(f"  Filtered: {cond_str}")
            continue
        
        # Additional check: filter conditions with undeclared loop variables
        # that would cause NameError when used with ForAll/Exists
        if _has_undeclared_quantifier_vars(parsed.condition, input_vars):
            filtered_count += 1
            if verbose:
                print(f"  Filtered (undeclared vars): {cond_str}")
            continue
        
        # Extract variables
        variables = extract_variables_from_condition(parsed.condition)
        result.all_variables.update(variables)
        
        # Convert to Z3
        z3_code = python_condition_to_z3(parsed.condition, parsed.negated)
        
        if z3_code and z3_code != "True":
            constraint = SMTConstraint(
                z3_code=z3_code,
                variables=variables,
                description=cond_str
            )
            result.constraints.append(constraint)
    
    if verbose and filtered_count > 0:
        print(f"  Total filtered: {filtered_count}/{len(condition_path)} conditions")
    
    return result


def _has_undeclared_quantifier_vars(condition: str, input_vars: List[str] = None) -> bool:
    """
    Check if a condition likely needs ForAll/Exists with undeclared variables.
    
    Conditions like "ForAll([i], ...)" or that reference loop indices (i, j, k)
    will cause NameError if those variables aren't declared in the generated code.
    This is a common failure mode in the benchmark.
    
    Args:
        condition: The condition string
        input_vars: Known input variable names
        
    Returns:
        True if the condition likely needs undeclared quantifier variables
    """
    # Common single-letter loop indices that cause problems
    loop_indices = {'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q'}
    
    # Extract variable names
    var_names = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', condition))
    builtins = {'True', 'False', 'None', 'and', 'or', 'not', 'in',
                'len', 'range', 'enumerate', 'str', 'int', 'float', 'bool',
                'isdigit', 'isalpha', 'ord', 'chr', 'abs', 'max', 'min', 'sum'}
    var_names = var_names - builtins
    
    if not var_names:
        return False
    
    # Check for loop indices not in input_vars
    input_vars_set = set(input_vars) if input_vars else set()
    
    for var in var_names:
        if var in loop_indices and var not in input_vars_set:
            # This variable is a loop index and not an input parameter
            # It will likely cause NameError in ForAll/Exists
            return True
    
    return False


def generate_smt_problem_for_path(
    task_data: Dict[str, Any],
    path_index: int,
    paths_data: Dict[str, Any]
) -> str:
    """
    Generate a complete SMT problem description for Agentic-Z3.
    
    Args:
        task_data: Task data from leetcode-py.jsonl
        path_index: Index of the target path in sampled_condition_paths
        paths_data: Path data from tgt_paths.jsonl
        
    Returns:
        Natural language problem description for Agentic-Z3
    """
    func_name = task_data.get('func_name', '')
    description = task_data.get('description', '')
    code = task_data.get('python_solution', '')
    
    condition_path = paths_data.get('sampled_condition_paths', [])[path_index]
    
    # Convert to SMT constraints
    constraint_set = convert_path_to_smt(condition_path, "", code)
    
    # Build problem description for Agentic-Z3
    path_desc = []
    for i, cond in enumerate(condition_path):
        path_desc.append(f"{i+1}. {cond}")
    
    problem = f"""
Generate test input for Python function '{func_name}' that follows this execution path:

Function description:
{description}

Target execution path (constraints to satisfy):
{chr(10).join(path_desc)}

Variables identified:
{', '.join(f"{k}: {v}" for k, v in constraint_set.all_variables.items())}

Find concrete values for the function input parameters that will cause the function 
to execute through the specified path. All path conditions must be satisfied in sequence.

Output the solution as a Python test case.
"""
    return problem


def path_conditions_to_precondition(condition_path: List[str]) -> str:
    """
    Convert path conditions to a precondition string for display/debugging.
    """
    conditions = []
    for cond_str in condition_path:
        parsed = parse_path_condition(cond_str)
        if parsed.condition:
            if parsed.negated:
                conditions.append(f"NOT ({parsed.condition})")
            else:
                conditions.append(f"({parsed.condition})")
    
    return " AND ".join(conditions) if conditions else "True"


# Utility function for testing
def test_conversion():
    """Test the path conversion functionality."""
    test_conditions = [
        "Line 26: (l < r)",
        "Line 28: NOT (summ == 0)",
        "Line 20: (i in range(len(nums) - 2))",
        "Line 44: (isConnected[i][j] == 1)",
        "Line 17: (c.isdigit())",
        "Line 21: (i > 0 and nums[i] == nums[i - 1])",
    ]
    
    print("Testing path condition parsing:")
    print("=" * 60)
    
    for cond in test_conditions:
        parsed = parse_path_condition(cond)
        z3_code = python_condition_to_z3(parsed.condition, parsed.negated)
        variables = extract_variables_from_condition(parsed.condition)
        
        print(f"\nOriginal: {cond}")
        print(f"Parsed: line={parsed.line_number}, negated={parsed.negated}")
        print(f"Condition: {parsed.condition}")
        print(f"Z3 Code: {z3_code}")
        print(f"Variables: {variables}")


if __name__ == "__main__":
    test_conversion()


