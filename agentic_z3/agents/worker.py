"""
Worker Agent Module for Agentic-Z3

The Worker is the code generator and strategist that:
1. Performs interactive type probing before full code generation
2. Generates Z3 Python code from blueprints with tracked constraints
3. Implements TTRL (Test-Time RL) with Soft Reset for exploration diversity

Key Innovations:

1. Type-Aware Interactive Probing:
   Before generating full Z3 code, the Worker creates a minimal "probe"
   script that just declares variables and runs trivial constraints.
   This catches type mismatches early (Int vs Real, etc.) before
   investing in complex code generation.

2. Soft Reset Mechanism (from BFS-Prover-V2):
   When Z3 returns UNKNOWN (timeout) multiple times, instead of just
   retrying, the Worker performs a "soft reset":
   - CLEARS the LLM conversation history (forgets failed attempts)
   - BUT RETAINS: original problem + compressed failure summaries
   - BOOSTS temperature (0.2 → 0.7) to force exploration diversity
   
   This prevents the LLM from getting stuck in local optima where
   it keeps generating similar failed strategies.

Critical Implementation Detail - assert_and_track:
   All constraints MUST use solver.assert_and_track(constraint, "name")
   NOT solver.add(constraint). This enables unsat core tracking.
   The Worker's prompts enforce this, and z3_executor has a fallback
   preprocessor that auto-converts add() calls if the LLM fails.
"""

from typing import Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings
from agentic_z3.agents.base_agent import BaseAgent
from agentic_z3.core.state import SMTState, PlanBlueprint, ProbeResult
from agentic_z3.memory.skill_library import SkillLibrary, SkillTemplate
from agentic_z3.memory.ttrl_cache import TTRLCache
from agentic_z3.tools.z3_executor import Z3Executor
from agentic_z3.utils.logger import get_logger, LogCategory
from agentic_z3.utils.prompter import PromptManager

logger = get_logger(__name__)


class Worker(BaseAgent):
    """
    Code generation agent with type probing and TTRL soft reset.
    
    The Worker's responsibilities:
    1. Generate minimal probe scripts for type verification
    2. Generate full Z3 code with named constraints for tracking
    3. Manage exploration diversity through soft reset mechanism
    
    Type-Aware Probing:
    Before generating complex constraint code, the Worker first creates
    a minimal script that just declares variables with their blueprint
    types and adds trivial constraints. This catches errors like:
    - Incorrect type declarations (Int when should be Real)
    - Type coercion issues (mixing Int and Real in operations)
    
    TTRL Soft Reset:
    When the solver times out repeatedly, traditional retry just generates
    similar code. Soft reset fundamentally changes the exploration:
    - Conversation history is CLEARED (forgets the failed path)
    - Failure summaries are PRESERVED (knows what NOT to do)
    - Temperature is BOOSTED (forces different generation)
    
    Attributes:
        z3_executor: Interface for running Z3 code
        ttrl_cache: Cache for tracking failed attempts
        skill_library: Library for retrieving similar skills
        prompt_manager: Template manager for prompts
    """
    
    def __init__(
        self,
        z3_executor: Z3Executor,
        ttrl_cache: TTRLCache,
        skill_library: SkillLibrary,
        **kwargs
    ):
        """
        Initialize the Worker with required dependencies.
        
        Args:
            z3_executor: For running probe and full Z3 scripts
            ttrl_cache: For tracking attempts and triggering soft reset
            skill_library: For retrieving helpful templates
        """
        self.z3_executor = z3_executor
        self.ttrl_cache = ttrl_cache
        self.skill_library = skill_library
        self.prompt_manager = PromptManager()
        
        # Track if we're in post-soft-reset mode
        self._post_soft_reset = False
        
        super().__init__(**kwargs)
    
    @property
    def system_prompt(self) -> str:
        """Return the Worker's system prompt emphasizing assert_and_track."""
        return self.prompt_manager.get_worker_system_prompt()
    
    def _build_probe_code(self, variables: list[dict]) -> str:
        """
        Deterministically build probe code from blueprint variables.
        
        This replaces the LLM-based probe generation with a fast, deterministic
        approach that constructs valid Z3 code directly from the variable specs.
        
        Args:
            variables: List of variable definitions from blueprint
                       Each has 'name' and 'type' keys
        
        Returns:
            Executable Z3 probe script
        """
        lines = [
            "from z3 import *",
            "",
            "# Variable declarations",
        ]
        
        # Type mapping for Z3 declarations
        type_map = {
            "Int": ("Int", "0"),
            "Real": ("Real", "0.0"),
            "Bool": ("Bool", "True"),
            "String": ("String", '""'),
            # BitVec needs special handling for width
        }
        
        for var in variables:
            name = var.get("name", "x")
            var_type = var.get("type", "Int")
            
            if var_type.startswith("BitVec"):
                # Extract bit width: BitVec(32) or BitVec32
                import re
                width_match = re.search(r'(\d+)', var_type)
                width = width_match.group(1) if width_match else "32"
                lines.append(f"{name} = BitVec('{name}', {width})")
            elif var_type in type_map:
                z3_type, _ = type_map[var_type]
                lines.append(f"{name} = {z3_type}('{name}')")
            else:
                # Default to Int for unknown types
                lines.append(f"{name} = Int('{name}')")
        
        lines.extend([
            "",
            "# Solver setup",
            "solver = Solver()",
            "",
            "# Trivial constraints to verify types",
        ])
        
        # Add trivial constraints per type
        for var in variables:
            name = var.get("name", "x")
            var_type = var.get("type", "Int")
            
            if var_type == "Int":
                lines.append(f"solver.add({name} >= 0)")
            elif var_type == "Real":
                lines.append(f"solver.add({name} >= 0.0)")
            elif var_type == "Bool":
                lines.append(f"solver.add(Or({name}, Not({name})))")  # Always true
            elif var_type == "String":
                lines.append(f"solver.add(Length({name}) >= 0)")  # Always true
            elif var_type.startswith("BitVec"):
                lines.append(f"solver.add({name} >= 0)")  # Unsigned comparison
            else:
                lines.append(f"solver.add({name} == {name})")  # Reflexive, always true
        
        lines.extend([
            "",
            "# Check satisfiability",
            "result = solver.check()",
            "print(f'Result: {result}')",
        ])
        
        return "\n".join(lines)
    
    def interactive_probe(
        self, 
        blueprint: Optional[PlanBlueprint]
    ) -> ProbeResult:
        """
        Perform type-aware interactive probing before full code generation.
        
        This is the first phase of the Worker's process. Instead of
        immediately generating full constraint code, we first:
        1. Generate a minimal script with just variable declarations (DETERMINISTIC)
        2. Add trivial constraints to verify type compatibility
        3. Execute against Z3 with a short timeout
        4. Catch type errors BEFORE investing in complex generation
        
        NOTE: This method now uses DETERMINISTIC probe generation instead of
        calling the LLM. This saves an LLM round-trip per retry and is faster.
        The probe code is constructed directly from the blueprint variables.
        
        Args:
            blueprint: The Architect's plan with variable types
            
        Returns:
            ProbeResult with success status and any type errors
        """
        if blueprint is None or not blueprint.variables:
            return ProbeResult(
                success=False,
                errors=["No blueprint or variables provided"]
            )
        
        logger.info(
            f"Probing types for {len(blueprint.variables)} variables",
            category=LogCategory.AGENT
        )
        
        try:
            # Generate probe code DETERMINISTICALLY (no LLM call!)
            probe_code = self._build_probe_code(blueprint.variables)
            
            logger.debug(
                f"Probe code (deterministic):\n{probe_code}",
                category=LogCategory.AGENT
            )
            
            # Execute probe with short timeout
            result = self.z3_executor.check_sat_with_timeout(
                probe_code,
                timeout_ms=settings.Z3_PROBE_TIMEOUT
            )
            
            if result.status == "error":
                # Type error or syntax error in probe
                logger.warning(
                    f"Probe failed: {result.error}",
                    category=LogCategory.Z3
                )
                return ProbeResult(
                    success=False,
                    errors=[result.error or "Unknown probe error"],
                    probe_code=probe_code
                )
            
            # Build type report from blueprint
            type_report = {
                var["name"]: var["type"] 
                for var in blueprint.variables
            }
            
            logger.info(
                "Type probe successful",
                category=LogCategory.AGENT
            )
            
            return ProbeResult(
                success=True,
                type_report=type_report,
                probe_code=probe_code
            )
            
        except Exception as e:
            logger.error(
                f"Probe generation failed: {e}",
                category=LogCategory.AGENT
            )
            return ProbeResult(
                success=False,
                errors=[str(e)]
            )
    
    def generate_code(
        self,
        blueprint: Optional[PlanBlueprint],
        probe_result: ProbeResult,
        skills: list[SkillTemplate],
        failure_context: Optional[str] = None
    ) -> str:
        """
        Generate full Z3 Python code from blueprint.
        
        This is the main code generation phase. The Worker:
        1. Uses the verified types from probe_result
        2. Incorporates relevant skills from the library
        3. Generates all constraints with assert_and_track for tracking
        4. Uses constraint naming that matches blueprint groups
        5. Learns from previous failures if failure_context is provided
        
        CRITICAL: All constraints must use:
            solver.assert_and_track(constraint, "c_groupname_N")
        NOT:
            solver.add(constraint)
        
        The naming convention enables the Coach to map unsat cores
        back to blueprint constraint groups for meaningful diagnosis.
        
        Args:
            blueprint: The Architect's structured plan
            probe_result: Verified types from probing phase
            skills: Retrieved skill templates from library
            failure_context: Optional error info from previous attempt (stderr/traceback)
                to help the model avoid repeating the same mistake
            
        Returns:
            Complete executable Z3 Python script
        """
        if blueprint is None:
            return self._generate_fallback_code()
        
        logger.info(
            f"Generating code: {len(blueprint.variables)} vars, "
            f"{len(blueprint.constraint_groups)} groups, "
            f"{len(skills)} skills",
            category=LogCategory.AGENT
        )
        
        # Build the generation prompt
        prompt_parts = [
            "Generate Z3 Python code for this blueprint:",
            "",
            f"Variables: {blueprint.variables}",
            "",
            f"Constraint Groups: {blueprint.constraint_groups}",
            "",
            f"Strategy Hint: {blueprint.solving_strategy}",
        ]
        
        # CRITICAL: Add failure context if this is a retry
        # This enables error-driven repair - the model can learn from the traceback
        if failure_context:
            prompt_parts.append("")
            prompt_parts.append("=" * 60)
            prompt_parts.append("PREVIOUS ATTEMPT FAILED - FIX THE ERROR BELOW:")
            prompt_parts.append("=" * 60)
            prompt_parts.append(failure_context[:1500])  # Truncate very long tracebacks
            prompt_parts.append("")
            prompt_parts.append("You MUST fix this error. Generate corrected code that avoids this issue.")
            prompt_parts.append("=" * 60)
        
        # Add skill templates if available
        if skills:
            prompt_parts.append("")
            prompt_parts.append("Helpful skill templates from similar problems:")
            for i, skill in enumerate(skills[:3]):
                prompt_parts.append(f"\nSkill {i+1}: {skill.description}")
                prompt_parts.append(f"Pattern: {skill.skeleton_code[:200]}...")
        
        # Add verified types
        if probe_result.type_report:
            prompt_parts.append("")
            prompt_parts.append(f"Verified Types: {probe_result.type_report}")
        
        # Reminder about assert_and_track
        prompt_parts.append("")
        prompt_parts.append("REMINDER: Use solver.assert_and_track(constraint, 'c_groupname_N') for ALL constraints!")
        
        user_message = "\n".join(prompt_parts)
        
        # Use elevated temperature if post-soft-reset
        temp = settings.LLM_TEMPERATURE_SOFT_RESET if self._post_soft_reset else self.temperature
        
        try:
            response = self._call_llm(
                user_message,
                temperature=temp
            )
            
            code = self._extract_code(response)
            
            # Validate assert_and_track usage
            if "solver.add(" in code and "assert_and_track" not in code:
                logger.warning(
                    "Generated code uses solver.add() - will be auto-converted",
                    category=LogCategory.AGENT
                )
            
            # Reset post-soft-reset flag after successful generation
            self._post_soft_reset = False
            
            return code
            
        except Exception as e:
            logger.error(
                f"Code generation failed: {e}",
                category=LogCategory.AGENT
            )
            return self._generate_fallback_code()
    
    def perform_soft_reset(self, state: SMTState) -> None:
        """
        Trigger a soft reset to force exploration diversity.
        
        This is the key TTRL mechanism from BFS-Prover-V2. When the
        solver times out repeatedly, we don't just retry - we fundamentally
        change the exploration strategy.
        
        SOFT RESET PROCESS:
        1. Get compressed failure summary BEFORE clearing (what NOT to do)
        2. Clear conversation history (forget the failed path)
        3. Reconstruct minimal context:
           - System prompt (role definition)
           - Original problem (ground truth)
           - Failure summary (avoid these approaches)
           - Explicit instruction to try different strategy
        4. Boost temperature (0.2 → 0.7) for generation diversity
        
        CRITICAL: We do NOT just clear messages[]. We carefully reconstruct
        context so the LLM:
        - Knows the problem (re-injected from state.problem_description)
        - Knows what failed (failure summaries preserved)
        - Is forced to explore differently (temperature boost)
        
        Args:
            state: Current SMTState with problem and failure history
        """
        logger.info(
            "Performing SOFT RESET - clearing history, preserving context",
            category=LogCategory.AGENT
        )
        
        # Step 1: Get failure summary BEFORE clearing
        failure_summary = self.ttrl_cache.get_failure_summary()
        
        # Also get failure summaries from state
        state_failures = state.get_failure_context(max_summaries=5)
        combined_failures = f"{failure_summary}\n\nFrom state history:\n{state_failures}"
        
        # Step 2: Clear conversation history
        self._reset_conversation()
        
        # Step 3: Reconstruct minimal context
        # Note: System prompt is already restored by _reset_conversation()
        
        # Re-inject problem (CRITICAL - this is the ground truth)
        self.messages.append({
            "role": "user",
            "content": f"Problem to solve: {state.problem_description}"
        })
        
        # Inject failure context (what NOT to do)
        self.messages.append({
            "role": "user", 
            "content": f"""PREVIOUS APPROACHES FAILED. Review and avoid:

{combined_failures}

You MUST try a FUNDAMENTALLY DIFFERENT approach.
Do not repeat any strategy that resembles the failed ones above."""
        })
        
        # Step 4: Boost temperature
        self.temperature = settings.LLM_TEMPERATURE_SOFT_RESET
        self._post_soft_reset = True
        
        logger.info(
            f"Soft reset complete: temperature boosted to {self.temperature}",
            category=LogCategory.AGENT
        )
    
    def generate_alternative_strategy(
        self,
        blueprint: PlanBlueprint,
        failed_approaches: list[str]
    ) -> str:
        """
        Generate an alternative solving strategy after failures.
        
        Used during TTRL when we need fundamentally different approaches.
        Examples of strategy alternatives:
        - Different quantifier triggers
        - Auxiliary variables
        - Problem decomposition
        - Different constraint encodings
        
        Args:
            blueprint: Current blueprint
            failed_approaches: List of approaches that failed
            
        Returns:
            New Z3 code with alternative strategy
        """
        prompt = self.prompt_manager.get_worker_soft_reset_prompt(
            problem=blueprint.problem_analysis,
            blueprint=blueprint.raw_json or {},
            failure_summary="\n".join(failed_approaches)
        )
        
        try:
            response = self._call_llm(
                prompt,
                temperature=settings.LLM_TEMPERATURE_SOFT_RESET
            )
            return self._extract_code(response)
        except Exception as e:
            logger.error(f"Alternative strategy generation failed: {e}",
                        category=LogCategory.AGENT)
            return self._generate_fallback_code()
    
    def _generate_fallback_code(self) -> str:
        """
        Generate minimal fallback code when generation fails.
        
        Returns a simple template that at least runs without error.
        """
        return '''from z3 import *

# Fallback template - generation failed
x = Int('x')
solver = Solver()
solver.set(unsat_core=True)
solver.assert_and_track(x >= 0, "c_fallback_1")
solver.assert_and_track(x <= 100, "c_fallback_2")

result = solver.check()
print(f"Result: {result}")
if result == sat:
    print(f"Model: {solver.model()}")
elif result == unsat:
    print(f"Unsat core: {solver.unsat_core()}")
'''
    
    def reset_temperature(self) -> None:
        """
        Reset temperature to base level after successful solve.
        
        Called when we want to return to deterministic generation
        after a soft reset period.
        """
        self.temperature = self.base_temperature
        self._post_soft_reset = False
        logger.debug(
            f"Temperature reset to {self.temperature}",
            category=LogCategory.AGENT
        )




