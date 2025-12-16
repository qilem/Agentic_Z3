"""
Z3 Executor Module for Agentic-Z3

The core interface for executing Z3 Python code with:
1. Timeout management to prevent infinite loops
2. Unsat core tracking for diagnosis
3. Safe sandboxed execution via subprocess
4. Automatic preprocessing to ensure assert_and_track usage

Key Features:

1. TIMEOUT HANDLING:
   Uses Z3's set_param('timeout', ms) to prevent runaway solvers.
   Also implements subprocess-level timeout as a safety net.

2. UNSAT CORE TRACKING:
   Ensures solver.set(unsat_core=True) is enabled.
   Parses the unsat_core() output to extract constraint names.

3. CRITICAL: preprocess_for_tracking():
   If the LLM generates code using solver.add() instead of
   solver.assert_and_track(), this function auto-converts it.
   This is essential because unsat core diagnosis ONLY works
   with tracked constraints.
   
   Pattern: solver.add(expr) → solver.assert_and_track(expr, "c_auto_N")

4. SANDBOXED EXECUTION:
   Code is executed in a subprocess for isolation.
   Captures stdout/stderr for result parsing.
   Handles crashes gracefully.
"""

from dataclasses import dataclass
from typing import Optional
import subprocess
import tempfile
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings
from agentic_z3.utils.logger import get_logger, LogCategory

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """
    Result of Z3 code execution.
    
    Attributes:
        status: One of 'sat', 'unsat', 'unknown', 'error'
        model: The satisfying model as string (if SAT)
        unsat_core: List of constraint names in conflict (if UNSAT)
        error: Error message (if ERROR)
        stdout: Raw stdout from execution
        stderr: Raw stderr from execution
        timed_out: True if execution hit timeout
    """
    status: str
    model: Optional[str] = None
    unsat_core: list[str] = None
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    
    def __post_init__(self):
        if self.unsat_core is None:
            self.unsat_core = []


class Z3Executor:
    """
    Executor for Z3 Python code with safety features.
    
    The Z3Executor provides:
    1. Safe subprocess-based execution
    2. Automatic timeout management
    3. Unsat core extraction
    4. Critical preprocessing for assert_and_track
    
    Preprocessing (CRITICAL):
    The LLM may generate code using solver.add() which doesn't
    support unsat core tracking. The preprocess_for_tracking()
    function automatically converts these to assert_and_track()
    calls with auto-generated constraint names.
    
    Execution Flow:
    1. Preprocess code (inject unsat_core=True, convert add→assert_and_track)
    2. Write code to temp file
    3. Execute in subprocess with timeout
    4. Parse output for result, model, unsat_core
    5. Return structured ExecutionResult
    
    Attributes:
        default_timeout: Default timeout in milliseconds
        python_executable: Path to Python interpreter
    """
    
    def __init__(
        self,
        default_timeout: Optional[int] = None,
        python_executable: Optional[str] = None
    ):
        """
        Initialize the Z3 executor.
        
        Args:
            default_timeout: Override default Z3 timeout (ms)
            python_executable: Override Python interpreter path
        """
        self.default_timeout = default_timeout or settings.Z3_TIMEOUT
        self.python_executable = python_executable or sys.executable
        
        logger.info(
            f"Z3Executor initialized: timeout={self.default_timeout}ms",
            category=LogCategory.SYSTEM
        )
    
    def run_with_unsat_core_tracking(
        self,
        code: str,
        timeout_ms: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute Z3 code with full unsat core tracking support.
        
        This is the main execution method. It:
        1. Preprocesses code to ensure assert_and_track usage
        2. Injects unsat_core=True if not present
        3. Executes in sandboxed subprocess
        4. Parses result and extracts unsat core if applicable
        
        Args:
            code: Z3 Python code to execute
            timeout_ms: Optional timeout override
            
        Returns:
            ExecutionResult with status, model, unsat_core, etc.
        """
        timeout = timeout_ms or self.default_timeout
        
        logger.debug(
            f"Executing Z3 code ({len(code)} chars) with {timeout}ms timeout",
            category=LogCategory.Z3
        )
        
        # Preprocess code for tracking
        processed_code = self.preprocess_for_tracking(code)
        
        # Inject unsat_core setting if not present
        processed_code = self._inject_unsat_core_setting(processed_code)
        
        # Execute
        return self._execute_in_subprocess(processed_code, timeout)
    
    def check_sat_with_timeout(
        self,
        code: str,
        timeout_ms: Optional[int] = None
    ) -> ExecutionResult:
        """
        Quick check_sat execution for type probing.
        
        Used by the Worker's interactive_probe() to verify types.
        Does NOT require assert_and_track preprocessing since
        we don't need unsat core for probing.
        
        Args:
            code: Minimal Z3 probe code
            timeout_ms: Short timeout for quick check
            
        Returns:
            ExecutionResult with status (no unsat core parsing)
        """
        timeout = timeout_ms or settings.Z3_PROBE_TIMEOUT
        
        logger.debug(
            f"Probe execution ({len(code)} chars) with {timeout}ms timeout",
            category=LogCategory.Z3
        )
        
        # For probes, just inject timeout, don't need tracking
        processed_code = self._inject_timeout_setting(code, timeout)
        
        return self._execute_in_subprocess(processed_code, timeout)
    
    def _fix_string_char_mismatch(self, code: str) -> str:
        """
        Fix Z3 String/Char sort mismatch errors.
        
        In Z3's Python API:
        - s[i] returns a Char (single character)
        - s.at(i) returns a String (length-1 substring)
        - StringVal('x') is a String
        
        Comparing s[i] == StringVal('x') causes Z3Exception: sort mismatch
        
        This method rewrites such patterns to use s.at(i) == StringVal('x')
        which is the correct way to do character-at-position comparisons.
        
        Args:
            code: Z3 Python code that may have string indexing issues
            
        Returns:
            Code with s[i] == StringVal patterns fixed to use s.at(i)
        """
        # Pattern: variable[index] == StringVal('char') or != StringVal('char')
        # Capture: var_name[index_expr] op StringVal('char')
        # Replace with: var_name.at(index_expr) op StringVal('char')
        
        # Match patterns like: s[i] == StringVal('x'), s[0] != StringVal('.'), etc.
        # The pattern handles variable names, index expressions, and both == and !=
        
        pattern = r'(\w+)\[([^\]]+)\]\s*(==|!=)\s*(StringVal\s*\([^)]+\))'
        
        def replace_with_at(match: re.Match) -> str:
            var_name = match.group(1)
            index_expr = match.group(2)
            operator = match.group(3)
            string_val = match.group(4)
            
            logger.debug(
                f"Fixing string/char mismatch: {var_name}[{index_expr}] -> {var_name}.at({index_expr})",
                category=LogCategory.Z3
            )
            
            return f"{var_name}.at({index_expr}) {operator} {string_val}"
        
        fixed_code = re.sub(pattern, replace_with_at, code)
        
        if fixed_code != code:
            # Count how many fixes we made
            fix_count = len(re.findall(pattern, code))
            logger.info(
                f"Auto-fixed {fix_count} string/char mismatch(es) (s[i] -> s.at(i))",
                category=LogCategory.Z3
            )
        
        return fixed_code
    
    def _sanitize_invalid_string_methods(self, code: str) -> str:
        """
        Rewrite invalid Python string methods on Z3 String objects.
        
        LLMs commonly generate code that calls Python string methods on Z3 strings,
        which crashes with AttributeError. This method rewrites the most common
        patterns to valid Z3 equivalents.
        
        Handles:
        - s.replace(a, b) -> Replace(s, a, b)
        - s.strip() -> And(Not(PrefixOf(StringVal(' '), s)), Not(SuffixOf(StringVal(' '), s)))
        - s.at(i).isdigit() / s[i].isdigit() -> And(StrToCode(s.at(i)) >= 48, StrToCode(s.at(i)) <= 57)
        - s.at(i).isalpha() / s[i].isalpha() -> Or(And(...uppercase...), And(...lowercase...))
        
        Args:
            code: Z3 Python code that may have invalid string method calls
            
        Returns:
            Code with invalid patterns rewritten to valid Z3 equivalents
        """
        original_code = code
        fix_count = 0
        
        # 1. Fix .replace(a, b) -> Replace(s, a, b)
        # Pattern: var.replace(arg1, arg2)
        replace_pattern = r'(\w+)\.replace\(([^,]+),\s*([^)]+)\)'
        
        def fix_replace(match: re.Match) -> str:
            var = match.group(1)
            arg1 = match.group(2).strip()
            arg2 = match.group(3).strip()
            return f"Replace({var}, {arg1}, {arg2})"
        
        new_code = re.sub(replace_pattern, fix_replace, code)
        if new_code != code:
            fix_count += len(re.findall(replace_pattern, code))
            code = new_code
        
        # 2. Fix s == s.strip() or comparisons involving .strip()
        # This is tricky - we rewrite "var == var.strip()" to no-whitespace constraint
        # Pattern: var == var.strip() or var.strip() == var
        strip_eq_pattern1 = r'(\w+)\s*==\s*\1\.strip\(\)'
        strip_eq_pattern2 = r'(\w+)\.strip\(\)\s*==\s*\1'
        
        def fix_strip_eq(match: re.Match) -> str:
            var = match.group(1)
            return f"And(Not(PrefixOf(StringVal(\" \"), {var})), Not(SuffixOf(StringVal(\" \"), {var})))"
        
        new_code = re.sub(strip_eq_pattern1, fix_strip_eq, code)
        if new_code != code:
            fix_count += len(re.findall(strip_eq_pattern1, code))
            code = new_code
        
        new_code = re.sub(strip_eq_pattern2, fix_strip_eq, code)
        if new_code != code:
            fix_count += len(re.findall(strip_eq_pattern2, code))
            code = new_code
        
        # 2b. Fix standalone .strip() calls (not in comparison)
        # Pattern: var.strip() when not already handled
        standalone_strip_pattern = r'(\w+)\.strip\(\)'
        # Only replace if it's still there after the comparison fixes
        if '.strip()' in code:
            def fix_standalone_strip(match: re.Match) -> str:
                var = match.group(1)
                # Can't directly replace standalone strip() meaningfully,
                # but we can at least prevent the crash by removing it
                # and logging a warning
                logger.warning(
                    f"Removing unsupported .strip() call on {var} - may affect semantics",
                    category=LogCategory.Z3
                )
                return var
            new_code = re.sub(standalone_strip_pattern, fix_standalone_strip, code)
            if new_code != code:
                fix_count += len(re.findall(standalone_strip_pattern, code))
                code = new_code
        
        # 3. Fix .isdigit() on string expressions
        # Pattern: expr.isdigit() where expr might be s.at(i) or s[i]
        # s.at(i).isdigit() -> And(StrToCode(s.at(i)) >= 48, StrToCode(s.at(i)) <= 57)
        isdigit_at_pattern = r'(\w+)\.at\(([^)]+)\)\.isdigit\(\)'
        
        def fix_isdigit_at(match: re.Match) -> str:
            var = match.group(1)
            idx = match.group(2)
            return f"And(StrToCode({var}.at({idx})) >= 48, StrToCode({var}.at({idx})) <= 57)"
        
        new_code = re.sub(isdigit_at_pattern, fix_isdigit_at, code)
        if new_code != code:
            fix_count += len(re.findall(isdigit_at_pattern, code))
            code = new_code
        
        # s[i].isdigit() -> And(StrToCode(s.at(i)) >= 48, StrToCode(s.at(i)) <= 57)
        isdigit_bracket_pattern = r'(\w+)\[([^\]]+)\]\.isdigit\(\)'
        
        def fix_isdigit_bracket(match: re.Match) -> str:
            var = match.group(1)
            idx = match.group(2)
            return f"And(StrToCode({var}.at({idx})) >= 48, StrToCode({var}.at({idx})) <= 57)"
        
        new_code = re.sub(isdigit_bracket_pattern, fix_isdigit_bracket, code)
        if new_code != code:
            fix_count += len(re.findall(isdigit_bracket_pattern, code))
            code = new_code
        
        # Standalone var.isdigit()
        isdigit_standalone_pattern = r'(\w+)\.isdigit\(\)'
        
        def fix_isdigit_standalone(match: re.Match) -> str:
            var = match.group(1)
            return f"And(StrToCode({var}) >= 48, StrToCode({var}) <= 57)"
        
        new_code = re.sub(isdigit_standalone_pattern, fix_isdigit_standalone, code)
        if new_code != code:
            fix_count += len(re.findall(isdigit_standalone_pattern, code))
            code = new_code
        
        # 4. Fix .isalpha() on string expressions
        # s.at(i).isalpha() -> Or(And(code >= 65, code <= 90), And(code >= 97, code <= 122))
        isalpha_at_pattern = r'(\w+)\.at\(([^)]+)\)\.isalpha\(\)'
        
        def fix_isalpha_at(match: re.Match) -> str:
            var = match.group(1)
            idx = match.group(2)
            return (f"Or(And(StrToCode({var}.at({idx})) >= 65, StrToCode({var}.at({idx})) <= 90), "
                    f"And(StrToCode({var}.at({idx})) >= 97, StrToCode({var}.at({idx})) <= 122))")
        
        new_code = re.sub(isalpha_at_pattern, fix_isalpha_at, code)
        if new_code != code:
            fix_count += len(re.findall(isalpha_at_pattern, code))
            code = new_code
        
        # s[i].isalpha()
        isalpha_bracket_pattern = r'(\w+)\[([^\]]+)\]\.isalpha\(\)'
        
        def fix_isalpha_bracket(match: re.Match) -> str:
            var = match.group(1)
            idx = match.group(2)
            return (f"Or(And(StrToCode({var}.at({idx})) >= 65, StrToCode({var}.at({idx})) <= 90), "
                    f"And(StrToCode({var}.at({idx})) >= 97, StrToCode({var}.at({idx})) <= 122))")
        
        new_code = re.sub(isalpha_bracket_pattern, fix_isalpha_bracket, code)
        if new_code != code:
            fix_count += len(re.findall(isalpha_bracket_pattern, code))
            code = new_code
        
        # Standalone var.isalpha()
        isalpha_standalone_pattern = r'(\w+)\.isalpha\(\)'
        
        def fix_isalpha_standalone(match: re.Match) -> str:
            var = match.group(1)
            return (f"Or(And(StrToCode({var}) >= 65, StrToCode({var}) <= 90), "
                    f"And(StrToCode({var}) >= 97, StrToCode({var}) <= 122))")
        
        new_code = re.sub(isalpha_standalone_pattern, fix_isalpha_standalone, code)
        if new_code != code:
            fix_count += len(re.findall(isalpha_standalone_pattern, code))
            code = new_code
        
        if fix_count > 0:
            logger.info(
                f"Auto-sanitized {fix_count} invalid Python string method(s) on Z3 strings",
                category=LogCategory.Z3
            )
        
        return code
    
    def preprocess_for_tracking(self, code: str) -> str:
        """
        CRITICAL: Convert solver.add() to solver.assert_and_track().
        
        This is the safety net that ensures unsat core diagnosis works
        even if the LLM generates code using the standard solver.add().
        
        The conversion:
            solver.add(x > 0)  
            → solver.assert_and_track(x > 0, "c_auto_1")
        
        Also handles:
        - Detecting existing assert_and_track (skip those)
        - Generating unique constraint names
        - Preserving multi-line add() calls
        - Detecting duplicate constraint names and renaming
        - Fixing String/Char sort mismatches (s[i] -> s.at(i))
        - Sanitizing invalid Python string methods on Z3 strings
        
        Args:
            code: Original Z3 code
            
        Returns:
            Code with add() converted to assert_and_track()
        """
        # First, sanitize invalid Python string methods (isdigit, strip, replace, etc.)
        code = self._sanitize_invalid_string_methods(code)
        
        # Then fix any string/char sort mismatches (s[i] -> s.at(i))
        code = self._fix_string_char_mismatch(code)
        
        # Skip if already using assert_and_track
        if "assert_and_track" in code and "solver.add(" not in code:
            logger.debug("Code already uses assert_and_track", category=LogCategory.Z3)
            return code
        
        # Track existing constraint names to avoid duplicates
        existing_names = set(re.findall(r'assert_and_track\([^,]+,\s*["\']([^"\']+)["\']', code))
        
        # Counter for auto-generated names
        counter = [0]
        
        def generate_unique_name() -> str:
            """Generate a unique constraint name."""
            counter[0] += 1
            name = f"c_auto_{counter[0]}"
            while name in existing_names:
                counter[0] += 1
                name = f"c_auto_{counter[0]}"
            existing_names.add(name)
            return name
        
        def replace_add(match: re.Match) -> str:
            """Replace solver.add() with assert_and_track()."""
            # Get the full matched text
            full_match = match.group(0)
            
            # Extract the solver variable name (might be 's', 'solver', etc.)
            solver_name = match.group(1)
            
            # Extract the expression inside add()
            expr = match.group(2)
            
            # Generate unique constraint name
            name = generate_unique_name()
            
            logger.debug(
                f"Converting: {solver_name}.add({expr[:30]}...) → assert_and_track(..., '{name}')",
                category=LogCategory.Z3
            )
            
            return f"{solver_name}.assert_and_track({expr}, \"{name}\")"
        
        # Pattern to match solver.add(...) calls
        # Handles: solver.add(expr), s.add(expr), my_solver.add(expr)
        # The expression can contain nested parentheses
        pattern = r'(\w+)\.add\(([^)]+(?:\([^)]*\)[^)]*)*)\)'
        
        processed = re.sub(pattern, replace_add, code)
        
        # Log if conversions were made
        if processed != code:
            conversion_count = counter[0]
            logger.info(
                f"Auto-converted {conversion_count} solver.add() calls to assert_and_track()",
                category=LogCategory.Z3
            )
        
        # Handle duplicate constraint names in assert_and_track calls
        processed = self._deduplicate_constraint_names(processed)
        
        # Inject Min/Max helpers if needed
        processed = self._inject_min_max_helpers(processed)
        
        return processed
    
    def _fix_fstring_labels(self, code: str) -> str:
        """
        Replace f-string labels in assert_and_track with unique literal labels.
        
        LLMs sometimes generate code like:
            solver.assert_and_track(x > 0, f"c_bound_{i}")
        
        This causes problems because:
        1. If i is a Z3 variable, it won't work as expected
        2. Loop iterations create duplicate labels (Z3Exception: named assertion defined twice)
        
        This method replaces f-string labels with unique auto-generated literals.
        
        Args:
            code: Z3 Python code that may have f-string labels
            
        Returns:
            Code with f-string labels replaced by unique literals
        """
        # Pattern: assert_and_track(..., f"..." or f'...')
        fstring_label_pattern = r'(assert_and_track\([^,]+,\s*)f(["\'])([^"\']*)\2(\s*\))'
        
        counter = [0]
        
        def replace_fstring(match: re.Match) -> str:
            prefix = match.group(1)
            quote = match.group(2)
            # original_template = match.group(3)  # e.g., "c_bound_{i}"
            suffix = match.group(4)
            counter[0] += 1
            new_label = f"c_fstring_auto_{counter[0]}"
            logger.debug(
                f"Replacing f-string label with '{new_label}'",
                category=LogCategory.Z3
            )
            return f'{prefix}"{new_label}"{suffix}'
        
        new_code = re.sub(fstring_label_pattern, replace_fstring, code)
        
        if new_code != code:
            logger.info(
                f"Replaced {counter[0]} f-string label(s) in assert_and_track with unique literals",
                category=LogCategory.Z3
            )
        
        return new_code
    
    def _inject_min_max_helpers(self, code: str) -> str:
        """
        Inject Min/Max helper functions if the code uses them.
        
        Z3 Python API doesn't have built-in Min/Max functions for arbitrary
        argument counts. LLMs often generate code using Min(a, b, c, ...) which
        causes NameError.
        
        This method injects helper definitions at the start of the code.
        
        Args:
            code: Z3 Python code that may use Min/Max
            
        Returns:
            Code with Min/Max helpers injected if needed
        """
        needs_min = re.search(r'\bMin\s*\(', code) is not None
        needs_max = re.search(r'\bMax\s*\(', code) is not None
        
        if not needs_min and not needs_max:
            return code
        
        # Build helper code
        helpers = []
        
        if needs_min:
            helpers.append('''
# Auto-injected Min helper (Z3 doesn't have built-in Min for arbitrary args)
def Min(*args):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        args = list(args[0])
    if len(args) == 0:
        raise ValueError("Min requires at least one argument")
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = If(arg < result, arg, result)
    return result
''')
            logger.info("Injected Min() helper function", category=LogCategory.Z3)
        
        if needs_max:
            helpers.append('''
# Auto-injected Max helper (Z3 doesn't have built-in Max for arbitrary args)
def Max(*args):
    if len(args) == 1 and hasattr(args[0], '__iter__'):
        args = list(args[0])
    if len(args) == 0:
        raise ValueError("Max requires at least one argument")
    if len(args) == 1:
        return args[0]
    result = args[0]
    for arg in args[1:]:
        result = If(arg > result, arg, result)
    return result
''')
            logger.info("Injected Max() helper function", category=LogCategory.Z3)
        
        # Insert helpers after imports
        helper_code = '\n'.join(helpers)
        
        # Find the best insertion point (after imports, before main code)
        lines = code.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1
            elif stripped and not stripped.startswith('#'):
                # Found first non-import, non-comment line
                break
        
        lines.insert(insert_idx, helper_code)
        return '\n'.join(lines)
    
    def _deduplicate_constraint_names(self, code: str) -> str:
        """
        Ensure all constraint names are unique.
        
        If the LLM used duplicate names in assert_and_track calls,
        append unique suffixes to make them distinct.
        """
        # First, fix f-string labels (common source of duplicates in loops)
        code = self._fix_fstring_labels(code)
        
        # Find all constraint names
        names = re.findall(r'assert_and_track\([^,]+,\s*["\']([^"\']+)["\']', code)
        
        # Check for duplicates
        seen = {}
        duplicates = set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen[name] = seen.get(name, 0) + 1
        
        if not duplicates:
            return code
        
        logger.warning(
            f"Duplicate constraint names detected: {duplicates}",
            category=LogCategory.Z3
        )
        
        # Rename duplicates
        for dup_name in duplicates:
            occurrence = [0]
            
            def rename_duplicate(match: re.Match) -> str:
                occurrence[0] += 1
                if occurrence[0] == 1:
                    return match.group(0)  # Keep first occurrence
                new_name = f"{dup_name}_{occurrence[0]}"
                return match.group(0).replace(f'"{dup_name}"', f'"{new_name}"').replace(f"'{dup_name}'", f"'{new_name}'")
            
            pattern = rf'assert_and_track\([^)]+,\s*["\']({re.escape(dup_name)})["\']'
            code = re.sub(pattern, rename_duplicate, code)
        
        return code
    
    def _inject_unsat_core_setting(self, code: str) -> str:
        """
        Inject solver.set(unsat_core=True) if not present.
        
        Must be called BEFORE adding constraints.
        """
        if "unsat_core=True" in code or "unsat_core = True" in code:
            return code
        
        # Find where solver is created and inject setting after
        # Patterns: Solver(), solver = Solver(), s = Solver()
        patterns = [
            (r'(\w+)\s*=\s*Solver\(\)', r'\g<0>\n\1.set(unsat_core=True)'),
            (r'(solver)\s*=\s*Solver\(\)', r'\g<0>\nsolver.set(unsat_core=True)'),
        ]
        
        for pattern, replacement in patterns:
            if re.search(pattern, code):
                code = re.sub(pattern, replacement, code, count=1)
                logger.debug("Injected unsat_core=True setting", category=LogCategory.Z3)
                return code
        
        # Fallback: Add at the beginning after imports
        import_end = code.rfind('import')
        if import_end != -1:
            newline_pos = code.find('\n', import_end)
            if newline_pos != -1:
                code = code[:newline_pos+1] + "\n# Auto-injected\nsolver.set(unsat_core=True)\n" + code[newline_pos+1:]
        
        return code
    
    def _inject_timeout_setting(self, code: str, timeout_ms: int) -> str:
        """Inject Z3 timeout setting into code."""
        timeout_line = f"\nfrom z3 import set_param\nset_param('timeout', {timeout_ms})\n"
        
        # Add after imports
        if "from z3 import" in code or "import z3" in code:
            # Find last import line
            lines = code.split('\n')
            last_import_idx = 0
            for i, line in enumerate(lines):
                if 'import' in line:
                    last_import_idx = i
            
            lines.insert(last_import_idx + 1, f"set_param('timeout', {timeout_ms})")
            return '\n'.join(lines)
        
        return timeout_line + code
    
    def _execute_in_subprocess(
        self,
        code: str,
        timeout_ms: int
    ) -> ExecutionResult:
        """
        Execute code in an isolated subprocess.
        
        Provides safety through:
        - Process isolation (crashes don't affect main process)
        - Timeout enforcement at subprocess level
        - Clean capture of stdout/stderr
        """
        # Convert timeout to seconds for subprocess
        timeout_sec = (timeout_ms / 1000) + 5  # Extra buffer for startup
        
        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                f.write(code)
                temp_path = f.name
            
            try:
                # Execute in subprocess
                result = subprocess.run(
                    [self.python_executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec
                )
                
                stdout = result.stdout
                stderr = result.stderr
                
                # Parse results
                return self._parse_output(stdout, stderr)
                
            except subprocess.TimeoutExpired:
                logger.warning(
                    f"Subprocess timeout after {timeout_sec}s",
                    category=LogCategory.Z3
                )
                return ExecutionResult(
                    status="unknown",
                    error=f"Execution timed out after {timeout_ms}ms",
                    timed_out=True
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                    
        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}", category=LogCategory.Z3)
            return ExecutionResult(
                status="error",
                error=str(e)
            )
    
    def _parse_output(self, stdout: str, stderr: str) -> ExecutionResult:
        """
        Parse subprocess output to extract results.
        
        Looks for:
        - "sat" / "unsat" / "unknown" in output
        - Model output after "Model:"
        - Unsat core after "Unsat core:" or "unsat_core()"
        - Error messages in stderr
        """
        combined = stdout + "\n" + stderr
        combined_lower = combined.lower()
        
        # Check for errors first
        if stderr and ("error" in stderr.lower() or "traceback" in stderr.lower()):
            return ExecutionResult(
                status="error",
                error=stderr[:500],
                stdout=stdout,
                stderr=stderr
            )
        
        # Determine status
        status = "unknown"
        if "result: sat" in combined_lower or "\nsat" in combined_lower or combined_lower.strip().endswith('sat'):
            status = "sat"
        elif "result: unsat" in combined_lower or "\nunsat" in combined_lower:
            status = "unsat"
        elif "unknown" in combined_lower:
            status = "unknown"
        
        # Extract model if SAT
        model = None
        if status == "sat":
            model_match = re.search(r'Model:\s*(.+?)(?:\n|$)', combined, re.DOTALL)
            if model_match:
                model = model_match.group(1).strip()
            else:
                # Try to find model output in different formats
                model_match = re.search(r'\[([^\]]+)\]', combined)
                if model_match:
                    model = model_match.group(0)
        
        # Extract unsat core if UNSAT
        unsat_core = []
        if status == "unsat":
            # Pattern: Unsat core: [c_1, c_2, ...]
            core_match = re.search(r'[Uu]nsat.?core[:\s]*\[([^\]]*)\]', combined)
            if core_match:
                core_str = core_match.group(1)
                # Parse constraint names
                unsat_core = [
                    name.strip().strip("'\"")
                    for name in core_str.split(',')
                    if name.strip()
                ]
        
        return ExecutionResult(
            status=status,
            model=model,
            unsat_core=unsat_core,
            stdout=stdout,
            stderr=stderr
        )




