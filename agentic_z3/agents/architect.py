"""
Architect Agent Module for Agentic-Z3

The Architect is the strategic planner that:
1. Decomposes complex problems into structured blueprints
2. Identifies variables with precise Z3 types
3. Groups constraints logically for unsat core diagnosis
4. Generates curriculum warmup problems for complex inputs

Key Innovation - Hierarchical Planning:
Instead of generating monolithic Z3 code, the Architect creates a
blueprint that the Worker can implement incrementally. This enables:
- Type-safe code generation (types are pre-declared)
- Meaningful unsat core analysis (constraints are grouped)
- Iterative refinement on failure (groups can be modified)

The Architect NEVER generates code - only structured plans.
"""

from typing import Optional
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agentic_z3.agents.base_agent import BaseAgent
from agentic_z3.core.state import PlanBlueprint, DiagnosisReport
from agentic_z3.utils.logger import get_logger, LogCategory
from agentic_z3.utils.prompter import PromptManager

logger = get_logger(__name__)


class Architect(BaseAgent):
    """
    Strategic planning agent that decomposes problems into blueprints.
    
    The Architect's role is to:
    1. Analyze the problem structure and identify variables
    2. Determine precise Z3 types for each variable (Int, Real, Bool, BitVec)
    3. Group constraints logically with named categories
    4. Provide solving strategy hints for the Worker
    
    The blueprint output enables:
    - Type-Aware Probing: Worker can verify types before full generation
    - Meaningful Diagnosis: Coach can map unsat cores to constraint groups
    - Iterative Refinement: Failed groups can be modified without full rewrite
    
    Curriculum Generation:
    For complex problems, the Architect can generate simplified "warmup"
    versions that build the skill library before tackling the full problem.
    
    Attributes:
        prompt_manager: Template manager for consistent prompts
    """
    
    def __init__(self, **kwargs):
        """Initialize the Architect with prompt templates."""
        self.prompt_manager = PromptManager()
        super().__init__(**kwargs)
    
    @property
    def system_prompt(self) -> str:
        """Return the Architect's system prompt defining its planner role."""
        return self.prompt_manager.get_architect_system_prompt()
    
    def create_blueprint(
        self, 
        problem: str,
        feedback: Optional[str] = None
    ) -> Optional[PlanBlueprint]:
        """
        Create a structured blueprint from a problem description.
        
        The blueprint is a hierarchical decomposition that:
        1. Identifies all variables and their Z3 types
        2. Groups constraints into logical categories
        3. Documents dependencies between constraint groups
        4. Provides strategy hints for the Worker
        
        Args:
            problem: Natural language problem description
            feedback: Optional feedback from previous failed attempt
            
        Returns:
            PlanBlueprint with structured decomposition, or None on failure
            
        Example Output:
            {
                "problem_analysis": "Simple linear constraint problem",
                "variables": [
                    {"name": "x", "type": "Int", "description": "First integer"},
                    {"name": "y", "type": "Int", "description": "Second integer"}
                ],
                "constraint_groups": [
                    {
                        "name": "boundary_conditions",
                        "description": "Variable bounds",
                        "constraints": ["x >= 0", "y >= 0"],
                        "depends_on": []
                    },
                    {
                        "name": "sum_requirement",
                        "description": "Sum constraint",
                        "constraints": ["x + y == 10"],
                        "depends_on": ["boundary_conditions"]
                    }
                ],
                "solving_strategy": "Direct solving should work"
            }
        """
        logger.info(
            f"Creating blueprint for: {problem[:100]}...",
            category=LogCategory.AGENT
        )
        
        # Build the prompt
        user_message = f"Create a blueprint for this problem:\n\n{problem}"
        
        if feedback:
            user_message += f"\n\nPrevious attempt feedback:\n{feedback}"
        
        try:
            # Request JSON-formatted response
            response = self._call_llm(
                user_message,
                json_mode=True
            )
            
            # Parse the response
            blueprint_data = self._extract_json(response)
            
            if blueprint_data is None:
                logger.error(
                    "Failed to parse blueprint JSON",
                    category=LogCategory.AGENT
                )
                return None
            
            # Validate required fields
            if not self._validate_blueprint(blueprint_data):
                logger.error(
                    "Blueprint missing required fields",
                    category=LogCategory.AGENT
                )
                return None
            
            blueprint = PlanBlueprint.from_dict(blueprint_data)
            
            logger.info(
                f"Blueprint created: {len(blueprint.variables)} vars, "
                f"{len(blueprint.constraint_groups)} groups",
                category=LogCategory.AGENT
            )
            
            return blueprint
            
        except Exception as e:
            logger.error(
                f"Blueprint creation failed: {e}",
                category=LogCategory.AGENT
            )
            return None
    
    def refine_blueprint(
        self,
        original_blueprint: Optional[PlanBlueprint],
        diagnosis: DiagnosisReport
    ) -> Optional[PlanBlueprint]:
        """
        Refine a blueprint based on Coach's diagnosis of failure.
        
        When the Worker generates code that results in UNSAT with conflicts
        identified by the Coach, the Architect refines the blueprint to
        resolve those conflicts.
        
        The refinement process:
        1. Receives the original blueprint and diagnosis
        2. Identifies which constraint groups need modification
        3. Adjusts constraints or groupings to resolve conflicts
        4. Returns a new blueprint for the Worker to implement
        
        Args:
            original_blueprint: The blueprint that led to failure
            diagnosis: Coach's analysis of the conflict
            
        Returns:
            Refined PlanBlueprint, or None on failure
        """
        if original_blueprint is None:
            logger.warning(
                "Cannot refine None blueprint - creating new",
                category=LogCategory.AGENT
            )
            return None
        
        logger.info(
            f"Refining blueprint based on diagnosis: {diagnosis.conflict_explanation[:100]}",
            category=LogCategory.AGENT
        )
        
        # Build refinement prompt
        user_message = self.prompt_manager.get_architect_refine_prompt(
            blueprint=original_blueprint.raw_json or {},
            diagnosis=diagnosis.conflict_explanation + "\n" + 
                      "Suggested fixes: " + ", ".join(diagnosis.suggested_fixes)
        )
        
        try:
            response = self._call_llm(
                user_message,
                json_mode=True
            )
            
            blueprint_data = self._extract_json(response)
            
            if blueprint_data is None or not self._validate_blueprint(blueprint_data):
                logger.error(
                    "Failed to parse refined blueprint",
                    category=LogCategory.AGENT
                )
                return None
            
            refined = PlanBlueprint.from_dict(blueprint_data)
            
            logger.info(
                f"Blueprint refined: {len(refined.variables)} vars, "
                f"{len(refined.constraint_groups)} groups",
                category=LogCategory.AGENT
            )
            
            return refined
            
        except Exception as e:
            logger.error(
                f"Blueprint refinement failed: {e}",
                category=LogCategory.AGENT
            )
            return None
    
    def generate_warmup_curriculum(self, problem: str) -> list[str]:
        """
        Generate simplified warmup problems for curriculum learning.
        
        When a problem is complex, solving simplified versions first
        builds the skill library with basic templates. These templates
        then assist the Worker when tackling the full problem.
        
        The curriculum progression:
        1. Simplest: Basic variable setup with trivial constraints
        2. Medium: Core logic with reduced scale
        3. Advanced: Full structure with relaxed constraints
        
        Inspired by SolSearch's curriculum-guided approach.
        
        Args:
            problem: The complex problem description
            
        Returns:
            List of 3 simplified problem descriptions, easiest first
        """
        logger.info(
            "Generating warmup curriculum",
            category=LogCategory.AGENT
        )
        
        user_message = self.prompt_manager.get_architect_curriculum_prompt(problem)
        
        try:
            response = self._call_llm(
                user_message,
                json_mode=True,
                add_to_history=False  # Don't pollute main conversation
            )
            
            curriculum_data = self._extract_json(response)
            
            if curriculum_data is None:
                # Try to extract as array directly
                if isinstance(curriculum_data, list):
                    return curriculum_data[:3]
                logger.warning(
                    "Failed to parse curriculum - using fallback",
                    category=LogCategory.AGENT
                )
                return self._generate_fallback_curriculum(problem)
            
            # Handle different response formats
            if isinstance(curriculum_data, list):
                problems = curriculum_data
            elif isinstance(curriculum_data, dict):
                problems = curriculum_data.get("problems", [])
                if not problems:
                    problems = curriculum_data.get("warmup_problems", [])
                if not problems:
                    # Try to extract values
                    problems = list(curriculum_data.values())[:3]
            else:
                problems = []
            
            # Ensure we have string problems
            result = []
            for p in problems[:3]:
                if isinstance(p, str):
                    result.append(p)
                elif isinstance(p, dict) and "description" in p:
                    result.append(p["description"])
                elif isinstance(p, dict) and "problem" in p:
                    result.append(p["problem"])
            
            if len(result) < 3:
                logger.warning(
                    f"Only got {len(result)} curriculum problems",
                    category=LogCategory.AGENT
                )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Curriculum generation failed: {e}",
                category=LogCategory.AGENT
            )
            return self._generate_fallback_curriculum(problem)
    
    def _validate_blueprint(self, data: dict) -> bool:
        """
        Validate that a blueprint has all required fields.
        
        Required structure:
        - variables: list of {name, type}
        - constraint_groups: list of {name, constraints}
        
        Args:
            data: Parsed JSON blueprint
            
        Returns:
            True if blueprint is valid
        """
        if not isinstance(data, dict):
            return False
        
        # Check variables
        variables = data.get("variables", [])
        if not isinstance(variables, list):
            return False
        for var in variables:
            if not isinstance(var, dict):
                return False
            if "name" not in var or "type" not in var:
                return False
        
        # Check constraint groups
        groups = data.get("constraint_groups", [])
        if not isinstance(groups, list):
            return False
        for group in groups:
            if not isinstance(group, dict):
                return False
            if "name" not in group:
                return False
        
        return True
    
    def _generate_fallback_curriculum(self, problem: str) -> list[str]:
        """
        Generate a basic fallback curriculum when LLM fails.
        
        Creates generic simplifications based on common patterns.
        
        Args:
            problem: Original problem
            
        Returns:
            List of 3 simplified problems
        """
        return [
            f"Simplified version 1: Consider a basic version where {problem[:50]}... with only 2 variables.",
            f"Simplified version 2: {problem[:50]}... with relaxed constraints (bounds doubled).",
            f"Simplified version 3: {problem[:50]}... with one constraint group removed.",
        ]
    
    def estimate_complexity(self, problem: str) -> dict:
        """
        Estimate the complexity of a problem for deciding on warmup.
        
        Analyzes the problem text to estimate:
        - Variable count
        - Constraint count
        - Presence of quantifiers
        - Optimization vs satisfiability
        
        Args:
            problem: Problem description
            
        Returns:
            Dict with complexity metrics
        """
        problem_lower = problem.lower()
        
        return {
            "estimated_variables": problem.count("variable") + 
                                   problem.count(" x") + 
                                   problem.count(" y") + 
                                   problem.count(" z"),
            "has_quantifiers": "forall" in problem_lower or "exists" in problem_lower,
            "is_optimization": any(w in problem_lower 
                                   for w in ["optimize", "maximize", "minimize"]),
            "constraint_indicators": problem.count(" and ") + 
                                     problem.count(" or ") + 
                                     problem.count(","),
            "text_length": len(problem),
        }


