"""
Coach Agent Module for Agentic-Z3

The Coach is the diagnostician and librarian that:
1. Analyzes UNSAT results by mapping unsat cores to blueprint groups
2. Provides actionable diagnosis for the Architect to refine plans
3. Crystallizes successful solutions into reusable skill templates
4. Manages the evolutionary skill library through curriculum learning

Key Innovations:

1. Structured Diagnosis (from Deep Research):
   Instead of just forwarding Z3's unsat core, the Coach:
   - Maps constraint names (c_boundary_1) back to blueprint groups
   - Identifies which logical constraint groups are in conflict
   - Provides natural language explanation of WHY they conflict
   - Suggests specific fixes for the Architect to implement

2. Skill Crystallization (from LEGO-Prover):
   Successful solutions are not just cached - they're generalized:
   - Specific numbers are replaced with placeholders (100 → {{PARAM_A}})
   - Constraint patterns are extracted as reusable templates
   - Templates are tagged with problem type for retrieval
   - Success rates are tracked for template prioritization

The Coach enables the system to learn from both failures (diagnosis)
and successes (crystallization), building an evolutionary skill set.
"""

from typing import Optional
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agentic_z3.agents.base_agent import BaseAgent
from agentic_z3.core.state import PlanBlueprint, DiagnosisReport
from agentic_z3.memory.skill_library import SkillLibrary, SkillTemplate
from agentic_z3.tools.z3_executor import Z3Executor
from agentic_z3.utils.logger import get_logger, LogCategory
from agentic_z3.utils.prompter import PromptManager

logger = get_logger(__name__)


class Coach(BaseAgent):
    """
    Diagnosis and skill management agent.
    
    The Coach has two main responsibilities:
    
    1. DIAGNOSIS: When Z3 returns UNSAT with an unsat core
       - Map constraint names back to blueprint constraint groups
       - Analyze why those groups are in logical conflict
       - Generate actionable recommendations for the Architect
       - Assess severity to determine if re-planning is needed
    
    2. CRYSTALLIZATION: When Z3 returns SAT
       - Extract the successful solution pattern
       - Parameterize it by replacing literals with placeholders
       - Store as a reusable skill template in the library
       - Track metadata for future retrieval relevance
    
    The Coach bridges the gap between Z3's technical output and
    the Architect's high-level planning, enabling effective iteration.
    
    Attributes:
        skill_library: ChromaDB-backed storage for skill templates
        z3_executor: For additional validation if needed
        prompt_manager: Template manager for consistent prompts
    """
    
    def __init__(
        self,
        skill_library: SkillLibrary,
        z3_executor: Z3Executor,
        **kwargs
    ):
        """
        Initialize the Coach with required dependencies.
        
        Args:
            skill_library: For storing crystallized skills
            z3_executor: For validation (optional use)
        """
        self.skill_library = skill_library
        self.z3_executor = z3_executor
        self.prompt_manager = PromptManager()
        super().__init__(**kwargs)
    
    @property
    def system_prompt(self) -> str:
        """Return the Coach's system prompt for diagnosis/crystallization."""
        return self.prompt_manager.get_coach_system_prompt()
    
    def analyze_unsat_core(
        self,
        core_list: list[str],
        blueprint: Optional[PlanBlueprint],
        code: str
    ) -> Optional[DiagnosisReport]:
        """
        Analyze Z3's unsat core and map it to blueprint constraint groups.
        
        This is the core diagnostic function. Given an unsat core like:
            ["c_boundary_1", "c_boundary_2", "c_logic_1"]
        
        The Coach:
        1. Parses constraint names to extract group prefixes
        2. Maps to blueprint constraint groups (boundary_conditions, logic)
        3. Analyzes WHY these groups conflict logically
        4. Generates natural language explanation for humans
        5. Suggests specific fixes (relax bounds, modify logic, etc.)
        6. Assesses severity (high = re-planning needed, low = UNSAT is correct)
        
        Args:
            core_list: Constraint names from Z3's unsat_core()
            blueprint: The Architect's plan for context
            code: The generated code for reference
            
        Returns:
            DiagnosisReport with conflict analysis and recommendations
        """
        if not core_list:
            logger.info(
                "Empty unsat core - problem is definitively unsatisfiable",
                category=LogCategory.AGENT
            )
            return DiagnosisReport(
                conflicting_groups=[],
                conflict_explanation="Problem is unsatisfiable with no core (trivially false)",
                severity="low"  # This is the correct answer
            )
        
        logger.info(
            f"Analyzing unsat core: {core_list}",
            category=LogCategory.AGENT
        )
        
        # Step 1: Map constraint names to groups
        group_mapping = self._map_constraints_to_groups(core_list, blueprint)
        conflicting_groups = list(set(group_mapping.values()))
        
        logger.debug(
            f"Constraint to group mapping: {group_mapping}",
            category=LogCategory.AGENT
        )
        
        # Step 2: Use LLM for deep analysis
        diagnosis_prompt = self.prompt_manager.get_coach_diagnose_prompt(
            unsat_core=core_list,
            blueprint=blueprint.raw_json if blueprint else {},
            code=code
        )
        
        try:
            response = self._call_llm(
                diagnosis_prompt,
                json_mode=True
            )
            
            diagnosis_data = self._extract_json(response)
            
            if diagnosis_data:
                # Merge LLM analysis with our mapping
                diagnosis_data["conflicting_groups"] = conflicting_groups
                return DiagnosisReport.from_dict(diagnosis_data)
            
        except Exception as e:
            logger.warning(
                f"LLM diagnosis failed, using heuristic: {e}",
                category=LogCategory.AGENT
            )
        
        # Fallback: Generate heuristic diagnosis
        return self._generate_heuristic_diagnosis(
            core_list, conflicting_groups, blueprint
        )
    
    def crystallize_skill(
        self,
        code: str,
        blueprint: Optional[PlanBlueprint],
        use_llm: bool = True
    ) -> Optional[SkillTemplate]:
        """
        Extract a reusable skill template from successful code.
        
        Skill crystallization converts a specific solution into a
        generalized template that can help solve similar problems.
        
        The process:
        1. Parse the code to identify literal values (numbers, strings)
        2. Replace literals with parameterized placeholders
        3. Extract the constraint pattern structure
        4. Generate description and applicability tags (optionally via LLM)
        5. Create a SkillTemplate for the library
        
        Example transformation:
            Original: solver.assert_and_track(x <= 100, "c_bound")
            Template: solver.assert_and_track(x <= {{UPPER_BOUND}}, "c_bound")
        
        Args:
            code: Successful Z3 Python code
            blueprint: Blueprint for context and tagging
            use_llm: If False, skip LLM call and use heuristic only (faster for benchmarks)
            
        Returns:
            SkillTemplate ready for storage, or None on failure
        """
        logger.info(
            "Crystallizing successful solution into skill template",
            category=LogCategory.AGENT
        )
        
        # Step 1: Parameterize the code
        parameterized_code, parameters = self._parameterize_code(code)
        
        # Step 2: Use LLM for rich description (if enabled)
        if use_llm:
            crystallize_prompt = self.prompt_manager.get_coach_crystallize_prompt(
                code=code,
                blueprint=blueprint.raw_json if blueprint else {},
                problem_type=self._infer_problem_type(blueprint)
            )
            
            try:
                response = self._call_llm(
                    crystallize_prompt,
                    json_mode=True,
                    add_to_history=False  # Don't pollute history with crystallization
                )
                
                template_data = self._extract_json(response)
                
                if template_data:
                    # Create skill template
                    return SkillTemplate(
                        template_name=template_data.get("template_name", "unnamed_skill"),
                        description=template_data.get("description", ""),
                        parameters=parameters or template_data.get("parameters", []),
                        skeleton_code=parameterized_code,
                        applicable_patterns=template_data.get("applicable_patterns", [])
                    )
                    
            except Exception as e:
                logger.warning(
                    f"LLM crystallization failed, using heuristic: {e}",
                    category=LogCategory.AGENT
                )
        
        # Fallback: Create basic template (heuristic, no LLM)
        return SkillTemplate(
            template_name=f"skill_{hash(code) % 10000}",
            description="Auto-extracted skill template",
            parameters=parameters,
            skeleton_code=parameterized_code,
            applicable_patterns=[]
        )
    
    def diagnose(
        self,
        error_message: str,
        code: str,
        blueprint: Optional[PlanBlueprint]
    ) -> str:
        """
        Provide general diagnosis for any error, not just UNSAT.
        
        Handles:
        - Syntax errors in generated code
        - Type errors during execution
        - Runtime exceptions
        - Timeout analysis
        
        Args:
            error_message: The error from execution
            code: The code that produced the error
            blueprint: Blueprint for context
            
        Returns:
            Natural language diagnosis with suggestions
        """
        logger.info(
            f"Diagnosing error: {error_message[:100]}...",
            category=LogCategory.AGENT
        )
        
        prompt = f"""Diagnose this Z3 execution error:

Error: {error_message}

Code:
{code[:1000]}

Blueprint context: {blueprint.raw_json if blueprint else 'N/A'}

Provide:
1. Root cause analysis
2. Specific fix suggestions
3. Whether this is a code bug or problem specification issue"""
        
        try:
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            return f"Diagnosis failed: {e}. Original error: {error_message}"
    
    def _map_constraints_to_groups(
        self,
        core_list: list[str],
        blueprint: Optional[PlanBlueprint]
    ) -> dict[str, str]:
        """
        Map constraint names from unsat core to blueprint groups.
        
        Naming convention expected: c_groupname_N or c_groupname
        Examples:
            c_boundary_1 → boundary
            c_logic_conditions_2 → logic_conditions
            c_auto_5 → auto (fallback for auto-converted constraints)
        
        Args:
            core_list: Constraint names from Z3
            blueprint: Blueprint with group definitions
            
        Returns:
            Dict mapping constraint name → group name
        """
        mapping = {}
        
        # Get known group names from blueprint
        known_groups = set()
        if blueprint:
            known_groups = set(blueprint.get_constraint_group_names())
        
        for constraint in core_list:
            # Parse constraint name
            # Expected format: c_groupname_N or c_groupname
            parts = constraint.split('_')
            
            if len(parts) >= 2 and parts[0] == 'c':
                # Extract group name (everything between c_ and _N)
                if parts[-1].isdigit():
                    group_name = '_'.join(parts[1:-1])
                else:
                    group_name = '_'.join(parts[1:])
                
                # Try to match to known groups
                if group_name in known_groups:
                    mapping[constraint] = group_name
                else:
                    # Fuzzy match
                    for known in known_groups:
                        if group_name in known or known in group_name:
                            mapping[constraint] = known
                            break
                    else:
                        mapping[constraint] = group_name or "unknown"
            else:
                mapping[constraint] = "unknown"
        
        return mapping
    
    def _generate_heuristic_diagnosis(
        self,
        core_list: list[str],
        conflicting_groups: list[str],
        blueprint: Optional[PlanBlueprint]
    ) -> DiagnosisReport:
        """
        Generate a heuristic diagnosis when LLM fails.
        
        Uses pattern matching on group names to suggest common fixes.
        """
        explanation_parts = []
        suggestions = []
        
        # Common conflict patterns
        if any("bound" in g for g in conflicting_groups):
            explanation_parts.append("Boundary constraints may be too restrictive")
            suggestions.append("Relax upper/lower bounds")
        
        if any("logic" in g for g in conflicting_groups):
            explanation_parts.append("Logical constraints may be contradictory")
            suggestions.append("Review logical AND/OR structure")
        
        if len(conflicting_groups) > 2:
            explanation_parts.append("Multiple constraint groups in conflict")
            suggestions.append("Consider breaking into smaller sub-problems")
        
        if not explanation_parts:
            explanation_parts.append(f"Constraints {core_list} are mutually unsatisfiable")
            suggestions.append("Review constraint definitions")
        
        # Determine severity
        if len(core_list) <= 2:
            severity = "low"  # Probably correct UNSAT
        elif len(conflicting_groups) > 2:
            severity = "high"  # Complex conflict, needs re-planning
        else:
            severity = "medium"
        
        return DiagnosisReport(
            conflicting_groups=conflicting_groups,
            conflict_explanation=" | ".join(explanation_parts),
            suggested_fixes=suggestions,
            severity=severity
        )
    
    def _parameterize_code(self, code: str) -> tuple[str, list[str]]:
        """
        Replace literal values with parameters for skill generalization.
        
        Transforms specific values into reusable placeholders:
            100 → {{PARAM_1}}
            "hello" → {{PARAM_2}}
        
        Args:
            code: Original Z3 code
            
        Returns:
            Tuple of (parameterized code, list of parameter names)
        """
        parameterized = code
        parameters = []
        param_counter = 0
        
        # Replace numeric literals (but not in variable names or indices)
        def replace_number(match):
            nonlocal param_counter
            num = match.group(0)
            # Skip small numbers (likely indices) and very specific numbers
            if int(float(num)) < 10 or '.' in num:
                return num
            param_counter += 1
            param_name = f"PARAM_{param_counter}"
            parameters.append(param_name)
            return "{{" + param_name + "}}"
        
        # Match numbers not preceded by underscore or letter
        parameterized = re.sub(
            r'(?<![_\w])(\d+)(?![_\w])',
            replace_number,
            parameterized
        )
        
        return parameterized, parameters
    
    def _infer_problem_type(self, blueprint: Optional[PlanBlueprint]) -> str:
        """Infer problem type from blueprint for tagging."""
        if not blueprint:
            return "unknown"
        
        analysis = (blueprint.problem_analysis or "").lower()
        groups = " ".join(blueprint.get_constraint_group_names()).lower()
        
        if "schedul" in analysis or "schedul" in groups:
            return "scheduling"
        elif "optim" in analysis:
            return "optimization"
        elif "graph" in analysis or "path" in groups:
            return "graph"
        elif "arith" in analysis or "bound" in groups:
            return "arithmetic"
        else:
            return "general"




