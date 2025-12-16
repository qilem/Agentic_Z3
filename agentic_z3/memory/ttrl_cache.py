"""
TTRL Cache Module for Agentic-Z3

Ephemeral cache for Test-Time Reinforcement Learning that:
1. Tracks attempts during a solving session
2. Detects when soft reset should be triggered
3. Provides compressed failure summaries for soft reset context

This cache is NOT persistent - it resets between problems.
Its purpose is to support the TTRL mechanism where we:
- Track consecutive failures
- Compress failed attempt history
- Trigger soft reset when stuck in local optima

The key insight from BFS-Prover-V2:
When Z3 keeps returning UNKNOWN (timeout), we're likely stuck in a
local optimum where the LLM generates similar failing strategies.
The soft reset mechanism uses this cache to:
1. Detect the stuck condition (consecutive UNKNOWNs)
2. Summarize what was tried (failure context)
3. Enable fundamentally different exploration
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings
from agentic_z3.utils.logger import get_logger, LogCategory

logger = get_logger(__name__)


@dataclass
class AttemptRecord:
    """
    Record of a single solving attempt.
    
    Captures enough information to:
    1. Detect patterns in failures
    2. Summarize what was tried
    3. Avoid repeating the same mistakes
    
    Attributes:
        code_hash: Hash of the generated code (to detect duplicates)
        code_snippet: First N characters of code for summary
        result: Outcome (sat, unsat, unknown, error)
        error: Error message if any
        timestamp: When the attempt was made
        strategy_hint: Optional hint about the strategy used
    """
    code_hash: str
    code_snippet: str
    result: str
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_hint: str = ""
    
    def to_summary(self) -> str:
        """Create a one-line summary of this attempt."""
        if self.error:
            return f"Attempt ({self.result}): {self.error[:100]}"
        elif self.strategy_hint:
            return f"Attempt ({self.result}): {self.strategy_hint}"
        else:
            return f"Attempt ({self.result}): {self.code_snippet[:50]}..."


class TTRLCache:
    """
    Ephemeral cache for tracking TTRL (Test-Time RL) attempts.
    
    The TTRL cache serves several purposes:
    
    1. ATTEMPT TRACKING:
       Records each code generation and execution attempt,
       including the result and any errors. This enables
       analysis of failure patterns.
    
    2. SOFT RESET DETECTION:
       Tracks consecutive failures (especially UNKNOWN/timeout)
       to detect when the system is stuck in a local optimum.
       When threshold is reached, signals for soft reset.
    
    3. FAILURE SUMMARIZATION:
       Compresses the history of failed attempts into a
       summary that can be injected into the LLM prompt
       after soft reset, telling it what NOT to try again.
    
    4. DUPLICATE DETECTION:
       Uses code hashing to detect if the LLM is generating
       identical or very similar code, which indicates
       it's stuck and needs soft reset.
    
    This cache is EPHEMERAL - it clears between problems.
    It's not for learning across problems (that's SkillLibrary).
    
    Attributes:
        attempts: List of attempt records
        consecutive_unknowns: Count for soft reset trigger
        consecutive_errors: Count for error pattern detection
        _code_hashes: Set for duplicate detection
    """
    
    def __init__(self):
        """Initialize an empty TTRL cache."""
        self.attempts: list[AttemptRecord] = []
        self.consecutive_unknowns: int = 0
        self.consecutive_errors: int = 0
        self._code_hashes: set[str] = set()
        self._soft_reset_count: int = 0
        
        logger.debug("TTRL cache initialized", category=LogCategory.SYSTEM)
    
    def record_attempt(
        self,
        code: str,
        result: str,
        error: str = "",
        strategy_hint: str = ""
    ) -> None:
        """
        Record a solving attempt.
        
        This should be called after each Z3 execution, regardless
        of the result. It tracks:
        - The code that was tried (hashed + snippet)
        - The result (sat, unsat, unknown, error)
        - Any error message
        - Optional strategy hint from the Worker
        
        Args:
            code: The Z3 Python code that was executed
            result: The execution result (sat/unsat/unknown/error)
            error: Error message if result was error
            strategy_hint: Optional description of strategy used
        """
        # Hash the code for duplicate detection
        code_hash = self._hash_code(code)
        
        # Check for duplicate
        is_duplicate = code_hash in self._code_hashes
        if is_duplicate:
            logger.warning(
                "Duplicate code detected - LLM may be stuck",
                category=LogCategory.AGENT
            )
        self._code_hashes.add(code_hash)
        
        # Create record
        record = AttemptRecord(
            code_hash=code_hash,
            code_snippet=code[:200],
            result=result.lower(),
            error=error,
            strategy_hint=strategy_hint
        )
        self.attempts.append(record)
        
        # Update consecutive counters
        if result.lower() == "unknown":
            self.consecutive_unknowns += 1
            self.consecutive_errors = 0
        elif result.lower() == "error":
            self.consecutive_errors += 1
            self.consecutive_unknowns = 0
        else:
            # SAT or UNSAT resets counters
            self.consecutive_unknowns = 0
            self.consecutive_errors = 0
        
        logger.debug(
            f"Recorded attempt: {result} (consecutive unknowns: {self.consecutive_unknowns})",
            category=LogCategory.SYSTEM
        )
    
    def should_trigger_soft_reset(self) -> bool:
        """
        Check if soft reset should be triggered.
        
        Soft reset is triggered when:
        1. Consecutive UNKNOWN (timeout) results >= threshold - 1
           (we use threshold - 1 so soft reset fires before retries exhaust)
        2. OR duplicate code detected (even 1 duplicate suggests LLM is stuck)
        
        NOTE: We do NOT trigger soft reset for consecutive ERRORs (except in extreme cases)
        because runtime errors are often fixable via error-driven repair - passing the
        error message back to the LLM. Soft reset would clear that error context.
        Only trigger for errors if we have many consecutive (4+) without progress,
        suggesting the LLM truly cannot fix the error.
        
        Returns:
            True if soft reset should be triggered
        """
        threshold = settings.SOFT_RESET_THRESHOLD
        # Use threshold - 1 so we trigger soft reset before we run out of retries
        # e.g., if threshold=3 and max_retries=3, we want to reset after 2 failures
        effective_threshold = max(2, threshold - 1)
        
        # Check consecutive timeouts (UNKNOWN) - these indicate structural issues
        # that need fundamentally different approaches
        if self.consecutive_unknowns >= effective_threshold:
            logger.info(
                f"Soft reset trigger: {self.consecutive_unknowns} consecutive timeouts",
                category=LogCategory.SYSTEM
            )
            return True
        
        # For errors, we're much more conservative because error-driven repair
        # (passing error back to LLM) often works. Only reset after many errors.
        # Use a higher threshold: at least 4 consecutive errors AND the error
        # message is the same (stuck in a loop).
        error_hard_threshold = max(4, threshold + 1)
        if self.consecutive_errors >= error_hard_threshold:
            # Check if we're generating the same error repeatedly
            recent_errors = [a.error for a in self.attempts[-self.consecutive_errors:] 
                           if a.result == "error" and a.error]
            if len(recent_errors) >= 2:
                # Check if errors are similar (first 100 chars match)
                first_error = recent_errors[0][:100] if recent_errors else ""
                all_similar = all(e[:100] == first_error for e in recent_errors)
                if all_similar:
                    logger.info(
                        f"Soft reset trigger: {self.consecutive_errors} consecutive identical errors",
                        category=LogCategory.SYSTEM
                    )
                    return True
        
        # Check for duplicates - even one duplicate suggests LLM is stuck
        # and needs a reset to explore different strategies
        total_attempts = len(self.attempts)
        unique_hashes = len(self._code_hashes)
        if total_attempts >= 2 and unique_hashes < total_attempts:
            logger.info(
                f"Soft reset trigger: duplicate code detected ({unique_hashes} unique/{total_attempts} total)",
                category=LogCategory.SYSTEM
            )
            return True
        
        return False
    
    def get_failure_summary(self, max_attempts: int = 5, include_errors: bool = True) -> str:
        """
        Get a compressed summary of failed attempts.
        
        This summary is injected into the LLM prompt after soft reset
        to tell it what NOT to try again. The summary includes:
        - What strategies were tried
        - What errors occurred (full error messages if include_errors=True)
        - Patterns in the failures
        
        Args:
            max_attempts: Maximum number of recent attempts to include
            include_errors: If True, include full error messages (truncated) for debugging
            
        Returns:
            Formatted string summary of failures
        """
        if not self.attempts:
            return "No previous attempts recorded."
        
        # Get recent failed attempts (not SAT)
        failed = [a for a in self.attempts if a.result != "sat"]
        recent = failed[-max_attempts:]
        
        if not recent:
            return "No failures recorded."
        
        # Build summary
        lines = [f"Summary of {len(recent)} recent failed attempts:"]
        
        for i, attempt in enumerate(recent, 1):
            lines.append(f"{i}. {attempt.to_summary()}")
            # Include the actual error message for error-driven repair
            if include_errors and attempt.error and attempt.result == "error":
                # Truncate long errors but include enough to be useful
                error_snippet = attempt.error[:500].strip()
                if error_snippet:
                    lines.append(f"   Error: {error_snippet}")
        
        # Add pattern analysis
        unknown_count = sum(1 for a in recent if a.result == "unknown")
        error_count = sum(1 for a in recent if a.result == "error")
        
        if unknown_count > 0:
            lines.append(f"\nPattern: {unknown_count} timeouts - try simpler constraints or different triggers")
        if error_count > 0:
            lines.append(f"\nPattern: {error_count} errors - check type compatibility and syntax")
            # Extract common error patterns
            error_msgs = [a.error for a in recent if a.result == "error" and a.error]
            if error_msgs:
                # Check for common Z3 issues
                if any("sort mismatch" in e.lower() for e in error_msgs):
                    lines.append("   Common issue: Z3 sort mismatch - check String/Char types, Int/Real mixing")
                if any("unknown sort" in e.lower() for e in error_msgs):
                    lines.append("   Common issue: Unknown sort - ensure proper Z3 imports and type declarations")
        
        return "\n".join(lines)
    
    def on_soft_reset(self) -> None:
        """
        Called when soft reset is performed.
        
        Resets the consecutive counters but keeps the attempt
        history for continued failure tracking.
        """
        self._soft_reset_count += 1
        self.consecutive_unknowns = 0
        self.consecutive_errors = 0
        # Keep _code_hashes to still detect duplicates
        
        logger.info(
            f"Soft reset #{self._soft_reset_count} recorded",
            category=LogCategory.SYSTEM
        )
    
    def clear_for_new_problem(self) -> None:
        """
        Clear the cache for a new problem.
        
        This should be called when starting to solve a new problem.
        Resets all tracking state.
        """
        self.attempts.clear()
        self.consecutive_unknowns = 0
        self.consecutive_errors = 0
        self._code_hashes.clear()
        self._soft_reset_count = 0
        
        logger.debug("TTRL cache cleared for new problem", category=LogCategory.SYSTEM)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the current solving session.
        
        Useful for debugging and analysis.
        """
        total = len(self.attempts)
        by_result = {}
        for attempt in self.attempts:
            by_result[attempt.result] = by_result.get(attempt.result, 0) + 1
        
        return {
            "total_attempts": total,
            "by_result": by_result,
            "unique_codes": len(self._code_hashes),
            "soft_resets": self._soft_reset_count,
            "consecutive_unknowns": self.consecutive_unknowns,
            "consecutive_errors": self.consecutive_errors
        }
    
    def _hash_code(self, code: str) -> str:
        """
        Hash code for duplicate detection.
        
        Normalizes whitespace before hashing to catch
        semantically identical code with different formatting.
        """
        import hashlib
        
        # Normalize: remove extra whitespace, lowercase
        normalized = ' '.join(code.split()).lower()
        
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]




