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
        
        Args:
            code: Original Z3 code
            
        Returns:
            Code with add() converted to assert_and_track()
        """
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
        
        return processed
    
    def _deduplicate_constraint_names(self, code: str) -> str:
        """
        Ensure all constraint names are unique.
        
        If the LLM used duplicate names in assert_and_track calls,
        append unique suffixes to make them distinct.
        """
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


