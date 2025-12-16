"""
State Management Module for Agentic-Z3

Defines the SMTState class which serves as the "shared memory" between all agents.
This is the central data structure that tracks:
- The problem being solved
- The Architect's blueprint
- The Worker's generated code
- Execution results and history
- Retry counts for TTRL mechanism

The state is immutable-by-convention: agents receive the state and return
new/modified values, but the Engine is responsible for state updates.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class ExecutionStatus(Enum):
    """
    Possible states of Z3 execution.
    
    PENDING: Initial state, no execution attempted yet
    SAT: Satisfiable - a model exists
    UNSAT: Unsatisfiable - no model can satisfy all constraints
    UNKNOWN: Solver couldn't determine (usually timeout)
    ERROR: Code execution failed (syntax error, type error, etc.)
    """
    PENDING = "pending"
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class PlanBlueprint:
    """
    Structured output from the Architect agent.
    
    This represents the hierarchical decomposition of the problem:
    - Variables with their Z3 types
    - Constraint groups with dependencies
    - Solving strategy hints
    
    Attributes:
        problem_analysis: Brief analysis of the problem structure
        variables: List of variable definitions with types
        constraint_groups: Logical groupings of constraints
        solving_strategy: Recommended approach for the Worker
        raw_json: Original JSON from the Architect (for debugging)
    """
    problem_analysis: str = ""
    variables: list[dict] = field(default_factory=list)
    constraint_groups: list[dict] = field(default_factory=list)
    solving_strategy: str = ""
    raw_json: Optional[dict] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "PlanBlueprint":
        """Create a PlanBlueprint from Architect's JSON output."""
        return cls(
            problem_analysis=data.get("problem_analysis", ""),
            variables=data.get("variables", []),
            constraint_groups=data.get("constraint_groups", []),
            solving_strategy=data.get("solving_strategy", ""),
            raw_json=data
        )
    
    def get_variable_names(self) -> list[str]:
        """Extract just the variable names from the blueprint."""
        return [v.get("name", "") for v in self.variables]
    
    def get_constraint_group_names(self) -> list[str]:
        """Extract constraint group names for unsat core mapping."""
        return [g.get("name", "") for g in self.constraint_groups]
    
    def get_constraints_by_group(self, group_name: str) -> list[str]:
        """Get constraints belonging to a specific group."""
        for group in self.constraint_groups:
            if group.get("name") == group_name:
                return group.get("constraints", [])
        return []


@dataclass
class ProbeResult:
    """
    Result of the Worker's type-checking probe.
    
    Before generating full code, the Worker creates a minimal script
    to verify variable types work correctly with Z3. This catches
    type mismatches early, before investing in full code generation.
    
    Attributes:
        success: True if probe executed without type errors
        type_report: Mapping of variable names to verified types
        errors: List of type errors encountered
        probe_code: The probe script that was executed
    """
    success: bool = False
    type_report: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    probe_code: str = ""


@dataclass
class DiagnosisReport:
    """
    Coach's analysis of a failed execution.
    
    Maps Z3's unsat core back to the Architect's constraint groups
    and provides actionable recommendations for fixing the conflict.
    
    Attributes:
        conflicting_groups: Names of constraint groups in conflict
        conflict_explanation: Natural language explanation of the conflict
        suggested_fixes: Specific recommendations for resolution
        severity: Impact level (high/medium/low)
        raw_json: Original JSON from the Coach
    """
    conflicting_groups: list[str] = field(default_factory=list)
    conflict_explanation: str = ""
    suggested_fixes: list[str] = field(default_factory=list)
    severity: str = "medium"
    raw_json: Optional[dict] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "DiagnosisReport":
        """Create a DiagnosisReport from Coach's JSON output."""
        return cls(
            conflicting_groups=data.get("conflicting_groups", []),
            conflict_explanation=data.get("conflict_explanation", ""),
            suggested_fixes=data.get("suggested_fixes", []),
            severity=data.get("severity", "medium"),
            raw_json=data
        )


@dataclass
class SMTState:
    """
    Central state object shared between all agents.
    
    This is the "shared memory" that tracks the entire solving process.
    The Engine is responsible for creating, updating, and passing this
    state to agents. Agents read the state and return new values.
    
    Lifecycle:
    1. Created with problem_description
    2. Architect populates plan_blueprint
    3. Worker populates probe_results, then current_code
    4. Engine populates execution_status, model, unsat_core_dump
    5. On failure, Coach populates diagnosis, Worker may retry
    6. failure_summaries accumulate for soft reset context
    
    Attributes:
        problem_description: Original user input (NEVER modified - ground truth)
        plan_blueprint: Architect's structured decomposition
        current_code: Worker's generated Z3 Python script
        execution_status: Current state of Z3 execution
        model: Satisfying assignment if SAT (as string)
        unsat_core_dump: Conflicting constraint names if UNSAT
        retry_count: Number of TTRL iterations for current problem
        failure_summaries: Compressed history of failed attempts
        probe_results: Type verification from interactive probing
        diagnosis: Coach's analysis of most recent failure
        planning_iterations: Number of times Architect has re-planned
    """
    # Core problem (immutable ground truth)
    problem_description: str
    
    # Architect output
    plan_blueprint: Optional[PlanBlueprint] = None
    
    # Worker output
    current_code: str = ""
    probe_results: Optional[ProbeResult] = None
    
    # Execution results
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    model: Optional[str] = None
    unsat_core_dump: list[str] = field(default_factory=list)
    error_message: str = ""
    
    # TTRL tracking
    retry_count: int = 0
    failure_summaries: list[str] = field(default_factory=list)
    
    # Coach output
    diagnosis: Optional[DiagnosisReport] = None
    
    # Planning tracking
    planning_iterations: int = 0
    
    # Last attempt tracking (preserved across resets for final status reporting)
    # These are populated before reset_for_retry() so we can report meaningful
    # final status when retries exhaust instead of returning PENDING.
    last_execution_status: Optional[ExecutionStatus] = None
    last_error_message: str = ""
    last_code: str = ""
    last_unsat_core: list[str] = field(default_factory=list)
    last_model: Optional[str] = None
    
    def preserve_last_attempt(self) -> None:
        """
        Preserve current attempt results before reset.
        
        This enables meaningful final status reporting when retries exhaust.
        Called by the Engine before reset_for_retry() to capture the last
        ERROR/UNKNOWN state so we don't lose it.
        """
        self.last_execution_status = self.execution_status
        self.last_error_message = self.error_message
        self.last_code = self.current_code
        self.last_unsat_core = list(self.unsat_core_dump)
        self.last_model = self.model
    
    def restore_last_attempt(self) -> None:
        """
        Restore the last attempt results after retries exhaust.
        
        Called by the Engine when max retries reached to ensure we return
        a meaningful status (ERROR/UNKNOWN) instead of PENDING.
        """
        if self.last_execution_status is not None:
            self.execution_status = self.last_execution_status
            self.error_message = self.last_error_message
            self.current_code = self.last_code
            self.unsat_core_dump = self.last_unsat_core
            self.model = self.last_model
    
    def reset_for_retry(self) -> None:
        """
        Reset execution-related fields for a new TTRL iteration.
        
        Keeps: problem_description, plan_blueprint, failure_summaries, last_* fields
        Resets: current_code, execution_status, model, etc.
        Increments: retry_count
        
        Note: Call preserve_last_attempt() BEFORE this method if you want
        to preserve the current results for final status reporting.
        """
        self.current_code = ""
        self.probe_results = None
        self.execution_status = ExecutionStatus.PENDING
        self.model = None
        self.unsat_core_dump = []
        self.error_message = ""
        self.diagnosis = None
        self.retry_count += 1
    
    def reset_for_replanning(self) -> None:
        """
        Reset for a new planning iteration after UNSAT diagnosis.
        
        Keeps: problem_description, failure_summaries
        Resets: Everything else including last_* tracking fields
        Increments: planning_iterations
        """
        self.plan_blueprint = None
        self.current_code = ""
        self.probe_results = None
        self.execution_status = ExecutionStatus.PENDING
        self.model = None
        self.unsat_core_dump = []
        self.error_message = ""
        self.diagnosis = None
        self.retry_count = 0
        self.planning_iterations += 1
        # Clear last attempt tracking since we're starting fresh
        self.last_execution_status = None
        self.last_error_message = ""
        self.last_code = ""
        self.last_unsat_core = []
        self.last_model = None
    
    def add_failure_summary(self, summary: str) -> None:
        """
        Record a compressed summary of a failed attempt.
        
        These summaries are preserved across soft resets and used
        to prevent the Worker from repeating failed strategies.
        
        Args:
            summary: Brief description of what was tried and why it failed
        """
        self.failure_summaries.append(summary)
    
    def get_failure_context(self, max_summaries: int = 5) -> str:
        """
        Get recent failure summaries for soft reset context.
        
        Returns the most recent N failure summaries, formatted
        as a numbered list for injection into the Worker's prompt.
        
        Args:
            max_summaries: Maximum number of summaries to include
            
        Returns:
            Formatted string of failure summaries
        """
        recent = self.failure_summaries[-max_summaries:]
        if not recent:
            return "No previous failures recorded."
        
        lines = [f"{i+1}. {summary}" for i, summary in enumerate(recent)]
        return "\n".join(lines)
    
    def should_trigger_soft_reset(self, threshold: int) -> bool:
        """
        Check if soft reset should be triggered based on retry count.
        
        Soft reset is triggered when the Worker has failed `threshold`
        consecutive times with UNKNOWN (timeout) status.
        
        Args:
            threshold: Number of consecutive failures before soft reset
            
        Returns:
            True if soft reset should be triggered
        """
        return self.retry_count > 0 and self.retry_count % threshold == 0
    
    def to_summary_dict(self) -> dict:
        """
        Create a summary dictionary for logging/debugging.
        
        Returns a condensed view of the current state without
        the full code or verbose fields.
        """
        summary = {
            "status": self.execution_status.value,
            "retry_count": self.retry_count,
            "planning_iterations": self.planning_iterations,
            "has_blueprint": self.plan_blueprint is not None,
            "has_code": bool(self.current_code),
            "has_model": self.model is not None,
            "unsat_core_size": len(self.unsat_core_dump),
            "failure_count": len(self.failure_summaries),
        }
        # Include last attempt info if available
        if self.last_execution_status is not None:
            summary["last_status"] = self.last_execution_status.value
            summary["has_last_error"] = bool(self.last_error_message)
        return summary




