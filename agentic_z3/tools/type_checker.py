"""
Type Checker Module for Agentic-Z3

Static analysis of Z3 Python code to:
1. Extract variable declarations and their Z3 types
2. Detect type mismatches (Int vs Real operations)
3. Validate code against blueprint specifications

This module supports the Type-Aware Probing mechanism by providing
static analysis capabilities. While the Worker's interactive_probe()
does dynamic validation, this module provides complementary static
analysis.

Type Issues in Z3:
Z3 is sensitive to types:
- Int('x') + Real('y') may cause unexpected behavior
- Comparing Int to Real requires explicit coercion
- BitVec operations must match bit widths

The TypeChecker identifies these issues BEFORE runtime, enabling
the Worker to fix them before submitting to Z3.
"""

from dataclasses import dataclass, field
from typing import Optional
import ast
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agentic_z3.core.state import PlanBlueprint
from agentic_z3.utils.logger import get_logger, LogCategory

logger = get_logger(__name__)


@dataclass
class TypeInfo:
    """
    Information about a Z3 variable's type.
    
    Attributes:
        name: Variable name
        z3_type: Z3 type (Int, Real, Bool, BitVec, etc.)
        line_number: Where the variable is declared
        bit_width: For BitVec types, the width
    """
    name: str
    z3_type: str
    line_number: int = 0
    bit_width: Optional[int] = None


@dataclass
class TypeError:
    """
    A detected type error or warning.
    
    Attributes:
        message: Description of the error
        line_number: Where the error occurs
        severity: 'error' or 'warning'
        suggestion: How to fix it
    """
    message: str
    line_number: int = 0
    severity: str = "error"
    suggestion: str = ""


@dataclass  
class TypeReport:
    """
    Complete type analysis report for Z3 code.
    
    Attributes:
        variables: Dict mapping variable names to TypeInfo
        errors: List of detected type errors
        warnings: List of type warnings
        is_valid: True if no errors (warnings OK)
    """
    variables: dict[str, TypeInfo] = field(default_factory=dict)
    errors: list[TypeError] = field(default_factory=list)
    warnings: list[TypeError] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def get_type(self, var_name: str) -> Optional[str]:
        """Get the Z3 type of a variable."""
        if var_name in self.variables:
            return self.variables[var_name].z3_type
        return None


class TypeChecker:
    """
    Static type analyzer for Z3 Python code.
    
    The TypeChecker performs static analysis to:
    1. Extract all variable declarations with their Z3 types
    2. Detect type mismatches in operations
    3. Validate against blueprint specifications
    4. Provide suggestions for type fixes
    
    Analysis is done without executing the code, using:
    - AST parsing for structured analysis
    - Regex patterns for Z3-specific constructs
    - Heuristic type inference
    
    This complements the Worker's dynamic type probing by catching
    issues that can be detected statically.
    
    Supported Z3 Types:
    - Int, Ints
    - Real, Reals  
    - Bool, Bools
    - BitVec, BitVecs
    - Array
    - Datatype (partial)
    """
    
    # Patterns for Z3 type declarations
    TYPE_PATTERNS = {
        "Int": r"(\w+)\s*=\s*Int\s*\(['\"](\w+)['\"]\)",
        "Ints": r"(\w+(?:\s*,\s*\w+)*)\s*=\s*Ints\s*\(['\"]([^'\"]+)['\"]\)",
        "Real": r"(\w+)\s*=\s*Real\s*\(['\"](\w+)['\"]\)",
        "Reals": r"(\w+(?:\s*,\s*\w+)*)\s*=\s*Reals\s*\(['\"]([^'\"]+)['\"]\)",
        "Bool": r"(\w+)\s*=\s*Bool\s*\(['\"](\w+)['\"]\)",
        "Bools": r"(\w+(?:\s*,\s*\w+)*)\s*=\s*Bools\s*\(['\"]([^'\"]+)['\"]\)",
        "BitVec": r"(\w+)\s*=\s*BitVec\s*\(['\"](\w+)['\"],\s*(\d+)\)",
        "BitVecs": r"(\w+(?:\s*,\s*\w+)*)\s*=\s*BitVecs\s*\(['\"]([^'\"]+)['\"],\s*(\d+)\)",
        "String": r"(\w+)\s*=\s*String\s*\(['\"](\w+)['\"]\)",
        "Strings": r"(\w+(?:\s*,\s*\w+)*)\s*=\s*Strings\s*\(['\"]([^'\"]+)['\"]\)",
    }
    
    # Operations that require type matching
    BINARY_OPS = {'+', '-', '*', '/', '<', '>', '<=', '>=', '==', '!='}
    
    def __init__(self):
        """Initialize the type checker."""
        logger.debug("TypeChecker initialized", category=LogCategory.SYSTEM)
    
    def analyze_types(self, code: str) -> TypeReport:
        """
        Analyze Z3 code to extract variable types and detect issues.
        
        This is the main analysis method. It:
        1. Parses the code to find variable declarations
        2. Builds a type map of all Z3 variables
        3. Analyzes operations for type compatibility
        4. Returns a comprehensive TypeReport
        
        Args:
            code: Z3 Python code to analyze
            
        Returns:
            TypeReport with variables, errors, and warnings
        """
        report = TypeReport()
        
        # Extract variable declarations
        self._extract_declarations(code, report)
        
        # Analyze operations for type issues
        self._analyze_operations(code, report)
        
        logger.debug(
            f"Type analysis: {len(report.variables)} vars, "
            f"{len(report.errors)} errors, {len(report.warnings)} warnings",
            category=LogCategory.SYSTEM
        )
        
        return report
    
    def validate_against_blueprint(
        self,
        code: str,
        blueprint: PlanBlueprint
    ) -> list[TypeError]:
        """
        Validate code types against blueprint specifications.
        
        Checks that the code declares variables with the types
        specified in the Architect's blueprint.
        
        Args:
            code: Generated Z3 code
            blueprint: Architect's blueprint with variable specs
            
        Returns:
            List of type errors where code doesn't match blueprint
        """
        errors = []
        
        # Get types from code
        report = self.analyze_types(code)
        
        # Check each blueprint variable
        for var_spec in blueprint.variables:
            var_name = var_spec.get("name", "")
            expected_type = var_spec.get("type", "")
            
            if not var_name:
                continue
            
            if var_name not in report.variables:
                errors.append(TypeError(
                    message=f"Blueprint variable '{var_name}' not declared in code",
                    severity="error",
                    suggestion=f"Add: {var_name} = {expected_type}('{var_name}')"
                ))
            else:
                actual_type = report.variables[var_name].z3_type
                if actual_type != expected_type:
                    errors.append(TypeError(
                        message=f"Variable '{var_name}' has type {actual_type}, expected {expected_type}",
                        line_number=report.variables[var_name].line_number,
                        severity="error",
                        suggestion=f"Change to: {var_name} = {expected_type}('{var_name}')"
                    ))
        
        return errors
    
    def _extract_declarations(self, code: str, report: TypeReport) -> None:
        """
        Extract Z3 variable declarations from code.
        
        Populates report.variables with TypeInfo for each found variable.
        """
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check each type pattern
            for z3_type, pattern in self.TYPE_PATTERNS.items():
                for match in re.finditer(pattern, line):
                    if z3_type in ('Int', 'Real', 'Bool', 'String'):
                        # Single variable: x = Int('x'), s = String('s')
                        var_name = match.group(1)
                        report.variables[var_name] = TypeInfo(
                            name=var_name,
                            z3_type=z3_type,
                            line_number=line_num
                        )
                    
                    elif z3_type in ('Ints', 'Reals', 'Bools', 'Strings'):
                        # Multiple variables: x, y = Ints('x y')
                        var_names = [v.strip() for v in match.group(1).split(',')]
                        base_type = z3_type[:-1]  # Ints → Int, Strings → String
                        for var_name in var_names:
                            report.variables[var_name] = TypeInfo(
                                name=var_name,
                                z3_type=base_type,
                                line_number=line_num
                            )
                    
                    elif z3_type == 'BitVec':
                        # BitVec with width: x = BitVec('x', 32)
                        var_name = match.group(1)
                        bit_width = int(match.group(3))
                        report.variables[var_name] = TypeInfo(
                            name=var_name,
                            z3_type=z3_type,
                            line_number=line_num,
                            bit_width=bit_width
                        )
                    
                    elif z3_type == 'BitVecs':
                        # Multiple BitVecs: x, y = BitVecs('x y', 32)
                        var_names = [v.strip() for v in match.group(1).split(',')]
                        bit_width = int(match.group(3))
                        for var_name in var_names:
                            report.variables[var_name] = TypeInfo(
                                name=var_name,
                                z3_type='BitVec',
                                line_number=line_num,
                                bit_width=bit_width
                            )
    
    def _analyze_operations(self, code: str, report: TypeReport) -> None:
        """
        Analyze operations for type compatibility.
        
        Looks for common type issues:
        - Int + Real mixing
        - BitVec width mismatches
        - Boolean in arithmetic
        """
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip comments and import lines
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('import') or stripped.startswith('from'):
                continue
            
            # Check for Int/Real mixing
            if self._check_int_real_mixing(line, report):
                report.warnings.append(TypeError(
                    message="Potential Int/Real mixing detected",
                    line_number=line_num,
                    severity="warning",
                    suggestion="Use ToReal() or ToInt() for explicit conversion"
                ))
            
            # Check BitVec width consistency
            bv_issue = self._check_bitvec_widths(line, report)
            if bv_issue:
                report.errors.append(TypeError(
                    message=bv_issue,
                    line_number=line_num,
                    severity="error",
                    suggestion="Ensure all BitVec operands have the same width"
                ))
    
    def _check_int_real_mixing(
        self, 
        line: str, 
        report: TypeReport
    ) -> bool:
        """
        Check if a line mixes Int and Real types in operations.
        
        Returns True if potential mixing is detected.
        """
        # Get Int and Real variables
        int_vars = {name for name, info in report.variables.items() 
                    if info.z3_type == 'Int'}
        real_vars = {name for name, info in report.variables.items() 
                     if info.z3_type == 'Real'}
        
        if not int_vars or not real_vars:
            return False
        
        # Check if both types appear in an expression
        for op in self.BINARY_OPS:
            # Look for patterns like "int_var + real_var"
            pattern = rf'(\w+)\s*{re.escape(op)}\s*(\w+)'
            for match in re.finditer(pattern, line):
                left, right = match.group(1), match.group(2)
                if (left in int_vars and right in real_vars) or \
                   (left in real_vars and right in int_vars):
                    return True
        
        return False
    
    def _check_bitvec_widths(
        self, 
        line: str, 
        report: TypeReport
    ) -> Optional[str]:
        """
        Check if BitVec operations use consistent widths.
        
        Returns error message if width mismatch found, None otherwise.
        """
        # Get BitVec variables with their widths
        bv_vars = {
            name: info.bit_width 
            for name, info in report.variables.items()
            if info.z3_type == 'BitVec' and info.bit_width
        }
        
        if len(bv_vars) < 2:
            return None
        
        # Check operations between BitVec variables
        for op in self.BINARY_OPS:
            pattern = rf'(\w+)\s*{re.escape(op)}\s*(\w+)'
            for match in re.finditer(pattern, line):
                left, right = match.group(1), match.group(2)
                if left in bv_vars and right in bv_vars:
                    if bv_vars[left] != bv_vars[right]:
                        return (f"BitVec width mismatch: {left} ({bv_vars[left]} bits) "
                               f"and {right} ({bv_vars[right]} bits)")
        
        return None
    
    def suggest_type_for_value(self, value: str) -> str:
        """
        Suggest a Z3 type based on a value literal.
        
        Useful for helping the Architect choose appropriate types.
        
        Args:
            value: A value literal (e.g., "42", "3.14", "True")
            
        Returns:
            Suggested Z3 type name
        """
        value = value.strip()
        
        if value.lower() in ('true', 'false'):
            return 'Bool'
        
        try:
            int(value)
            return 'Int'
        except ValueError:
            pass
        
        try:
            float(value)
            return 'Real'
        except ValueError:
            pass
        
        # Default to Int for unknown
        return 'Int'




