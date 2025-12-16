"""
Engine Module - The Orchestrator for Agentic-Z3

Implements the main state machine that coordinates all agents:
1. Architect plans the problem structure
2. Worker generates and executes Z3 code
3. Coach diagnoses failures and crystallizes successes

The Engine manages the TTRL (Test-Time Reinforcement Learning) loop:
- Retries with increasing exploration on UNKNOWN (timeout)
- Triggers soft reset after threshold consecutive failures
- Routes UNSAT results to Coach for diagnosis and re-planning

State Machine Transitions:
    INIT → Planning → Probing → Coding → Solving
    Solving → SAT → Crystallize → DONE
    Solving → UNSAT → Diagnose → Planning (loop)
    Solving → UNKNOWN → SoftReset? → Coding (loop)
    Any → ERROR → Diagnose → Retry
"""

from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings
from agentic_z3.core.state import (
    SMTState, 
    ExecutionStatus, 
    PlanBlueprint,
    ProbeResult,
    DiagnosisReport
)
from agentic_z3.agents.architect import Architect
from agentic_z3.agents.worker import Worker
from agentic_z3.agents.coach import Coach
from agentic_z3.memory.skill_library import SkillLibrary
from agentic_z3.memory.ttrl_cache import TTRLCache
from agentic_z3.tools.z3_executor import Z3Executor
from agentic_z3.utils.logger import get_logger, LogCategory

logger = get_logger(__name__)


class Engine:
    """
    Main orchestrator for the Agentic-Z3 solving pipeline.
    
    The Engine implements a state machine that coordinates three agents:
    - Architect: Decomposes problems into structured blueprints
    - Worker: Generates type-safe Z3 code with TTRL soft reset
    - Coach: Diagnoses failures and extracts successful patterns
    
    The solving loop implements:
    1. Hierarchical Planning: Problems are decomposed before coding
    2. Type-Aware Probing: Variables are type-checked before full generation
    3. TTRL with Soft Reset: Failed attempts trigger exploration diversity
    4. Skill Crystallization: Successful patterns are saved for reuse
    
    Attributes:
        z3_timeout: Milliseconds before Z3 solver times out
        max_retries: Maximum TTRL iterations before giving up
        soft_reset_threshold: Consecutive failures before soft reset
        architect: The planning agent
        worker: The code generation agent
        coach: The diagnosis and crystallization agent
        skill_library: ChromaDB-backed skill storage
        ttrl_cache: Ephemeral cache for soft reset context
        z3_executor: Z3 interface with unsat core tracking
    """
    
    def __init__(
        self,
        z3_timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        soft_reset_threshold: Optional[int] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_curriculum_warmup: Optional[bool] = None,
        history_mode: Optional[str] = None,
        enable_skill_crystallization: Optional[bool] = None,
        enable_skill_library: Optional[bool] = None
    ):
        """
        Initialize the Engine with configuration.
        
        Args:
            z3_timeout: Override Z3 solver timeout (ms)
            max_retries: Override maximum TTRL retries
            soft_reset_threshold: Override soft reset threshold
            model: Override LLM model for all agents
            temperature: Override LLM temperature for all agents
            max_tokens: Override LLM max tokens for all agents
            enable_curriculum_warmup: Override curriculum warmup (False to disable for benchmark)
            history_mode: Override agent history mode ('stateful', 'trimmed', 'stateless')
            enable_skill_crystallization: Override skill crystallization (False to disable for benchmark)
            enable_skill_library: Override skill library usage (False to disable for benchmark)
        """
        # Configuration with defaults from settings
        self.z3_timeout = z3_timeout or settings.Z3_TIMEOUT
        self.max_retries = max_retries or settings.MAX_TTRL_RETRIES
        self.soft_reset_threshold = soft_reset_threshold or settings.SOFT_RESET_THRESHOLD
        self.max_planning_iterations = settings.MAX_PLANNING_ITERATIONS
        
        # Curriculum warmup override (None = use settings default)
        self.enable_curriculum_warmup = enable_curriculum_warmup
        
        # Benchmark optimization overrides
        self.enable_skill_crystallization = (
            enable_skill_crystallization 
            if enable_skill_crystallization is not None 
            else settings.ENABLE_SKILL_CRYSTALLIZATION
        )
        self.enable_skill_library = (
            enable_skill_library
            if enable_skill_library is not None
            else settings.ENABLE_SKILL_LIBRARY
        )
        
        # LLM configuration for agents
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history_mode = history_mode
        
        # Initialize components
        self.skill_library = SkillLibrary() if self.enable_skill_library else None
        self.ttrl_cache = TTRLCache()
        self.z3_executor = Z3Executor(default_timeout=self.z3_timeout)
        
        # Agent kwargs (only include non-None values)
        agent_kwargs = {}
        if model is not None:
            agent_kwargs['model'] = model
        if temperature is not None:
            agent_kwargs['temperature'] = temperature
        if max_tokens is not None:
            agent_kwargs['max_tokens'] = max_tokens
        if history_mode is not None:
            agent_kwargs['history_mode'] = history_mode
        
        # Initialize agents with LLM config
        self.architect = Architect(**agent_kwargs)
        self.worker = Worker(
            z3_executor=self.z3_executor,
            ttrl_cache=self.ttrl_cache,
            skill_library=self.skill_library,
            **agent_kwargs
        )
        self.coach = Coach(
            skill_library=self.skill_library,
            z3_executor=self.z3_executor,
            **agent_kwargs
        )
        
        logger.info(
            f"Engine initialized: timeout={self.z3_timeout}ms, "
            f"max_retries={self.max_retries}, soft_reset_threshold={self.soft_reset_threshold}, "
            f"model={model or 'default'}, temperature={temperature or 'default'}, "
            f"history_mode={history_mode or 'default'}, skill_crystallization={self.enable_skill_crystallization}, "
            f"skill_library={self.enable_skill_library}",
            category=LogCategory.SYSTEM
        )
    
    def _reset_agents_for_new_problem(self) -> None:
        """
        Reset agent conversation histories at the start of a new problem.
        
        In stateful mode, this prevents cross-problem history bleed.
        In stateless/trimmed modes, this is a no-op but ensures clean state.
        
        Note: Only resets if history_mode is 'stateful' to allow multi-turn
        conversations within a single problem. For benchmark runs with
        stateless mode, conversations are already independent per call.
        """
        if self.history_mode == "stateful":
            logger.debug(
                "Resetting agent conversations for new problem (stateful mode)",
                category=LogCategory.SYSTEM
            )
            self.architect._reset_conversation()
            self.worker._reset_conversation()
            self.coach._reset_conversation()
    
    def create_blueprint(self, problem: str) -> Optional[PlanBlueprint]:
        """
        Create a blueprint for a problem WITHOUT solving it.
        
        This is useful for caching blueprints across multiple related problems
        (e.g., different paths of the same function in a benchmark).
        
        Args:
            problem: Natural language description of the problem
            
        Returns:
            PlanBlueprint or None if planning fails
        """
        logger.info(f"Creating blueprint for: {problem[:100]}...", 
                    category=LogCategory.SYSTEM)
        return self.architect.create_blueprint(problem)
    
    def solve_with_blueprint(
        self, 
        problem: str, 
        blueprint: PlanBlueprint
    ) -> SMTState:
        """
        Solve a problem using a pre-existing blueprint.
        
        This skips the Architect planning phase, which is useful when:
        - The blueprint was cached from a previous call
        - Multiple related problems share the same structure (e.g., different paths)
        - You want to test different TTRL strategies on the same blueprint
        
        Args:
            problem: Natural language description of the problem
            blueprint: Pre-existing blueprint to use (skips Architect)
            
        Returns:
            Final SMTState with execution results
        """
        logger.info(f"Starting solve with cached blueprint for: {problem[:100]}...", 
                    category=LogCategory.SYSTEM)
        
        # Initialize state with the pre-existing blueprint
        state = SMTState(problem_description=problem)
        state.plan_blueprint = blueprint
        self.ttrl_cache.clear_for_new_problem()
        
        # Reset agents for new problem (even with cached blueprint)
        self._reset_agents_for_new_problem()
        
        # Skip directly to TTRL loop (no planning phase)
        state = self._ttrl_loop(state)
        
        # Handle results
        if state.execution_status == ExecutionStatus.SAT:
            self._crystallize_success(state)
        elif state.execution_status == ExecutionStatus.UNSAT:
            # Diagnose but don't re-plan since we're using a cached blueprint
            diagnosis = self._diagnose_unsat(state)
            state.diagnosis = diagnosis
        
        return state
    
    def solve(self, problem: str) -> SMTState:
        """
        Main entry point: solve an SMT problem.
        
        Implements the full state machine:
        1. Planning: Architect creates blueprint
        2. TTRL Loop: Worker generates code, Engine executes
        3. Result Handling: SAT→crystallize, UNSAT→diagnose, UNKNOWN→retry
        
        Args:
            problem: Natural language description of the problem
            
        Returns:
            Final SMTState with execution results
        """
        logger.info(f"Starting solve for problem: {problem[:100]}...", 
                    category=LogCategory.SYSTEM)
        
        # Initialize state
        state = SMTState(problem_description=problem)
        self.ttrl_cache.clear_for_new_problem()
        
        # Reset agent conversations if in stateful mode (to prevent cross-problem history bleed)
        # In stateless/trimmed modes, this is a no-op but ensures clean state per problem
        self._reset_agents_for_new_problem()
        
        # Main planning loop (re-plan on UNSAT diagnosis)
        while state.planning_iterations < self.max_planning_iterations:
            # Phase 1: Planning
            state = self._planning_phase(state)
            if state.plan_blueprint is None:
                logger.error("Planning failed - no blueprint generated", 
                            category=LogCategory.SYSTEM)
                state.execution_status = ExecutionStatus.ERROR
                state.error_message = "Failed to generate plan blueprint"
                return state
            
            # Phase 2: TTRL Loop (Code → Execute → Diagnose)
            state = self._ttrl_loop(state)
            
            # Check exit conditions
            if state.execution_status == ExecutionStatus.SAT:
                # Success! Crystallize and return
                self._crystallize_success(state)
                return state
            
            elif state.execution_status == ExecutionStatus.UNSAT:
                # Diagnose and potentially re-plan
                diagnosis = self._diagnose_unsat(state)
                state.diagnosis = diagnosis
                
                if diagnosis and diagnosis.severity == "high":
                    # Major conflict - need to re-plan
                    logger.info("High severity conflict - re-planning required",
                                category=LogCategory.SYSTEM)
                    state.add_failure_summary(
                        f"UNSAT: {diagnosis.conflict_explanation}"
                    )
                    state.reset_for_replanning()
                    continue
                else:
                    # UNSAT is the correct answer (no solution exists)
                    logger.info("UNSAT confirmed - problem has no solution",
                                category=LogCategory.SYSTEM)
                    return state
            
            else:
                # UNKNOWN or ERROR after max retries - give up
                logger.warning(
                    f"Giving up after {state.retry_count} retries",
                    category=LogCategory.SYSTEM
                )
                return state
        
        logger.warning(
            f"Max planning iterations ({self.max_planning_iterations}) reached",
            category=LogCategory.SYSTEM
        )
        return state
    
    def _planning_phase(self, state: SMTState) -> SMTState:
        """
        Execute the planning phase with the Architect.
        
        If this is a re-planning iteration (after UNSAT diagnosis),
        the Architect receives the diagnosis to inform refinement.
        
        Args:
            state: Current state with problem description
            
        Returns:
            State updated with plan_blueprint
        """
        logger.info(
            f"Planning phase (iteration {state.planning_iterations + 1})",
            category=LogCategory.SYSTEM
        )
        
        # Check if we should generate warmup curriculum
        # Use instance override if set, otherwise fall back to settings
        warmup_enabled = (
            self.enable_curriculum_warmup 
            if self.enable_curriculum_warmup is not None 
            else settings.ENABLE_CURRICULUM_WARMUP
        )
        if (warmup_enabled and 
            state.planning_iterations == 0 and
            self._is_complex_problem(state.problem_description)):
            self._run_curriculum_warmup(state.problem_description)
        
        # Generate or refine blueprint
        if state.diagnosis is not None:
            # Re-planning after failure
            logger.info("Refining blueprint based on diagnosis", 
                        category=LogCategory.AGENT)
            blueprint = self.architect.refine_blueprint(
                state.plan_blueprint,
                state.diagnosis
            )
        else:
            # Initial planning
            blueprint = self.architect.create_blueprint(state.problem_description)
        
        state.plan_blueprint = blueprint
        
        if blueprint:
            logger.info(
                f"Blueprint created: {len(blueprint.variables)} variables, "
                f"{len(blueprint.constraint_groups)} constraint groups",
                category=LogCategory.AGENT
            )
        
        return state
    
    def _ttrl_loop(self, state: SMTState) -> SMTState:
        """
        Execute the TTRL (Test-Time Reinforcement Learning) loop.
        
        This loop:
        1. Worker probes types (interactive probing)
        2. Worker generates full Z3 code
        3. Z3 executor runs the code
        4. On UNKNOWN/ERROR, retry with possible soft reset
        
        The loop exits when:
        - SAT: Solution found
        - UNSAT: No solution exists (may need re-planning)
        - Max retries reached
        
        Args:
            state: State with blueprint ready
            
        Returns:
            State updated with execution results
        """
        logger.info("Entering TTRL loop", category=LogCategory.SYSTEM)
        
        # Track if we need soft reset for next iteration
        # (checked after recording attempt, applied at start of next iteration)
        _pending_soft_reset = False
        
        # Cache probe result across retries (probe only depends on blueprint,
        # which doesn't change during retries). This saves LLM calls.
        _cached_probe_result: Optional[ProbeResult] = None
        
        # Cache skill retrieval ONCE per path (not per retry)
        # Skills depend on problem_description which doesn't change during retries
        _cached_skills = []
        if self.enable_skill_library and self.skill_library is not None:
            _cached_skills = self.skill_library.retrieve(
                state.problem_description,
                top_k=settings.SKILL_RETRIEVAL_TOP_K
            )
            logger.debug(
                f"Retrieved {len(_cached_skills)} skills (cached for retries)",
                category=LogCategory.SYSTEM
            )
        else:
            logger.debug(
                "Skill library disabled, skipping retrieval",
                category=LogCategory.SYSTEM
            )
        
        while state.retry_count < self.max_retries:
            logger.info(
                f"TTRL iteration {state.retry_count + 1}/{self.max_retries}",
                category=LogCategory.SYSTEM
            )
            
            # Apply pending soft reset from previous iteration's failure analysis
            if _pending_soft_reset:
                logger.info(
                    "Applying soft reset from previous iteration",
                    category=LogCategory.SYSTEM
                )
                self.worker.perform_soft_reset(state)
                self.ttrl_cache.on_soft_reset()
                _pending_soft_reset = False
            
            # Step 1: Interactive type probing (CACHED across retries)
            # The probe only depends on the blueprint, which doesn't change during retries
            if _cached_probe_result is None:
                probe_result = self.worker.interactive_probe(state.plan_blueprint)
                if probe_result.success:
                    _cached_probe_result = probe_result
                    logger.debug(
                        "Probe result cached for subsequent retries",
                        category=LogCategory.SYSTEM
                    )
            else:
                probe_result = _cached_probe_result
                logger.debug(
                    "Using cached probe result",
                    category=LogCategory.SYSTEM
                )
            
            state.probe_results = probe_result
            
            if not probe_result.success:
                logger.warning(
                    f"Type probe failed: {probe_result.errors}",
                    category=LogCategory.Z3
                )
                state.add_failure_summary(
                    f"Type probe failed: {', '.join(probe_result.errors)}"
                )
                # Record probe failure as an attempt for stuck detection
                self.ttrl_cache.record_attempt(
                    code="# probe failed",
                    result="error",
                    error=", ".join(probe_result.errors)
                )
                # Check if we should soft reset on next iteration
                if self.ttrl_cache.should_trigger_soft_reset():
                    _pending_soft_reset = True
                state.reset_for_retry()
                continue
            
            # Step 2: Generate full Z3 code
            # Use cached skills (retrieved once before the retry loop)
            skills = _cached_skills
            
            # Build failure context for error-driven repair
            # This helps the LLM avoid repeating the same mistake
            failure_context = None
            if state.last_error_message:
                # Include the last error/traceback so the model can learn from it
                failure_context = f"Error from previous attempt:\n{state.last_error_message}"
                # Also include recent failure summaries if available
                recent_failures = state.get_failure_context(max_summaries=3)
                if recent_failures and "No previous failures" not in recent_failures:
                    failure_context += f"\n\nRecent failure history:\n{recent_failures}"
            
            code = self.worker.generate_code(
                state.plan_blueprint,
                probe_result,
                skills,
                failure_context=failure_context
            )
            state.current_code = code
            
            logger.debug(f"Generated code:\n{code[:500]}...", 
                        category=LogCategory.AGENT)
            
            # Step 3: Execute with Z3
            result = self.z3_executor.run_with_unsat_core_tracking(code)
            
            # Record in TTRL cache (prefer full stderr for better error-driven repair)
            self.ttrl_cache.record_attempt(
                code=code,
                result=result.status,
                error=result.stderr or result.error or ""
            )
            
            # Update state with results
            state.execution_status = ExecutionStatus[result.status.upper()]
            state.model = result.model
            state.unsat_core_dump = result.unsat_core
            # Prefer full raw stderr (traceback) when available.
            # `result.error` may be a truncated summary depending on executor parsing.
            state.error_message = result.stderr or result.error or ""
            
            logger.info(
                f"Z3 result: {result.status}",
                category=LogCategory.Z3
            )

            # If execution errored, surface the full traceback/stderr in logs.
            # This is the most actionable debugging signal when you see "Z3 result: error".
            if result.status == "error":
                if result.stderr:
                    logger.error(
                        "Z3 execution failed. Full stderr/traceback:\n"
                        + result.stderr,
                        category=LogCategory.Z3,
                    )
                elif result.error:
                    logger.error(
                        "Z3 execution failed. Error:\n" + result.error,
                        category=LogCategory.Z3,
                    )
            
            # Check exit conditions
            if state.execution_status in (ExecutionStatus.SAT, ExecutionStatus.UNSAT):
                return state
            
            # UNKNOWN or ERROR - prepare for retry
            if state.execution_status == ExecutionStatus.UNKNOWN:
                state.add_failure_summary(
                    f"Timeout (UNKNOWN) - solver couldn't determine in {self.z3_timeout}ms"
                )
            elif state.execution_status == ExecutionStatus.ERROR:
                state.add_failure_summary(
                    f"Execution error: {state.error_message[:100]}"
                )
            
            # Check if we should trigger soft reset for next iteration
            # This uses TTRLCache's real "stuck" signals: consecutive errors/unknowns,
            # duplicate code hashes - more reliable than just retry_count % threshold
            if self.ttrl_cache.should_trigger_soft_reset():
                logger.info(
                    "Soft reset will be triggered next iteration (TTRLCache detected stuck: "
                    f"consecutive_unknowns={self.ttrl_cache.consecutive_unknowns}, "
                    f"consecutive_errors={self.ttrl_cache.consecutive_errors})",
                    category=LogCategory.SYSTEM
                )
                _pending_soft_reset = True
            
            # Preserve current attempt results before resetting
            # so we can restore meaningful final status if retries exhaust
            state.preserve_last_attempt()
            state.reset_for_retry()
        
        logger.warning(
            f"TTRL loop exhausted ({self.max_retries} retries)",
            category=LogCategory.SYSTEM
        )
        
        # Restore the last attempt results so we return ERROR/UNKNOWN
        # instead of PENDING when retries are exhausted
        state.restore_last_attempt()
        
        return state
    
    def _diagnose_unsat(self, state: SMTState) -> Optional[DiagnosisReport]:
        """
        Have the Coach diagnose an UNSAT result.
        
        Maps the unsat core back to blueprint constraint groups
        and determines if re-planning is needed.
        
        Args:
            state: State with UNSAT result and unsat_core_dump
            
        Returns:
            DiagnosisReport with conflict analysis
        """
        if not state.unsat_core_dump:
            logger.info(
                "No unsat core available - UNSAT is definitive",
                category=LogCategory.Z3
            )
            return None
        
        logger.info(
            f"Diagnosing UNSAT with core: {state.unsat_core_dump}",
            category=LogCategory.AGENT
        )
        
        return self.coach.analyze_unsat_core(
            state.unsat_core_dump,
            state.plan_blueprint,
            state.current_code
        )
    
    def _crystallize_success(self, state: SMTState) -> None:
        """
        Have the Coach extract a reusable skill from success.
        
        Converts the successful solution into a parameterized template
        and stores it in the skill library for future reuse.
        
        Args:
            state: State with SAT result and successful code
        """
        # Skip crystallization if disabled (e.g., for benchmark speed)
        if not self.enable_skill_crystallization:
            logger.debug(
                "Skill crystallization disabled, skipping",
                category=LogCategory.SYSTEM
            )
            return
        
        # Skip if skill library is disabled
        if not self.enable_skill_library or self.skill_library is None:
            logger.debug(
                "Skill library disabled, skipping crystallization",
                category=LogCategory.SYSTEM
            )
            return
        
        logger.info(
            "Crystallizing successful solution into skill template",
            category=LogCategory.AGENT
        )
        
        try:
            # Pass use_llm=False if crystallization is disabled to use heuristic only
            # This saves LLM calls while still creating basic templates
            template = self.coach.crystallize_skill(
                state.current_code,
                state.plan_blueprint,
                use_llm=self.enable_skill_crystallization
            )
            
            if template:
                self.skill_library.store(
                    template,
                    metadata={
                        "problem_type": self._categorize_problem(state.problem_description),
                        "variable_count": len(state.plan_blueprint.variables),
                        "constraint_count": sum(
                            len(g.get("constraints", []))
                            for g in state.plan_blueprint.constraint_groups
                        )
                    }
                )
                logger.info(
                    f"Skill '{template.template_name}' saved to library",
                    category=LogCategory.SYSTEM
                )
        except Exception as e:
            logger.warning(
                f"Failed to crystallize skill: {e}",
                category=LogCategory.SYSTEM
            )
    
    def _run_curriculum_warmup(self, problem: str) -> None:
        """
        Generate and solve warmup problems for curriculum learning.
        
        When a complex problem is detected, the Architect generates
        simplified variants. Solving these warms up the skill library
        with basic templates before tackling the full problem.
        
        Args:
            problem: The complex problem description
        """
        logger.info(
            "Running curriculum warmup for complex problem",
            category=LogCategory.SYSTEM
        )
        
        try:
            curriculum = self.architect.generate_warmup_curriculum(problem)
            
            for i, warmup_problem in enumerate(curriculum[:3]):  # Max 3 warmups
                logger.info(
                    f"Warmup {i+1}/3: {warmup_problem[:50]}...",
                    category=LogCategory.SYSTEM
                )
                
                # Solve warmup with reduced retry budget
                warmup_state = SMTState(problem_description=warmup_problem)
                warmup_state.plan_blueprint = self.architect.create_blueprint(warmup_problem)
                
                if warmup_state.plan_blueprint:
                    # Quick solve attempt
                    probe = self.worker.interactive_probe(warmup_state.plan_blueprint)
                    if probe.success:
                        code = self.worker.generate_code(
                            warmup_state.plan_blueprint,
                            probe,
                            []  # No skill retrieval for warmup
                        )
                        result = self.z3_executor.run_with_unsat_core_tracking(code)
                        
                        if result.status == "sat":
                            warmup_state.current_code = code
                            self._crystallize_success(warmup_state)
                            
        except Exception as e:
            logger.warning(
                f"Curriculum warmup failed: {e}",
                category=LogCategory.SYSTEM
            )
    
    def _is_complex_problem(self, problem: str) -> bool:
        """
        Heuristically determine if a problem is complex.
        
        Complex problems benefit from curriculum warmup.
        Indicators: length, quantifiers, multiple constraints, etc.
        
        Args:
            problem: Problem description
            
        Returns:
            True if problem appears complex
        """
        # Simple heuristics - can be refined
        complexity_indicators = [
            len(problem) > 500,
            "forall" in problem.lower(),
            "exists" in problem.lower(),
            "optimize" in problem.lower(),
            "schedule" in problem.lower(),
            problem.count(" and ") > 3,
            problem.count(" or ") > 3,
        ]
        return sum(complexity_indicators) >= 2
    
    def _categorize_problem(self, problem: str) -> str:
        """
        Categorize a problem for skill library tagging.
        
        Args:
            problem: Problem description
            
        Returns:
            Category string (e.g., "arithmetic", "scheduling")
        """
        problem_lower = problem.lower()
        
        if any(w in problem_lower for w in ["schedule", "task", "resource"]):
            return "scheduling"
        elif any(w in problem_lower for w in ["route", "path", "graph"]):
            return "graph"
        elif any(w in problem_lower for w in ["optimize", "maximize", "minimize"]):
            return "optimization"
        elif any(w in problem_lower for w in ["prove", "theorem", "forall"]):
            return "theorem_proving"
        else:
            return "arithmetic"




