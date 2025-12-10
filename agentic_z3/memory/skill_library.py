"""
Skill Library Module for Agentic-Z3

ChromaDB-backed vector database for storing and retrieving reusable
SMT templates (skills) that can accelerate solving similar problems.

Key Features:
1. Semantic Search: Templates are embedded and retrieved by similarity
2. Parameterization: Templates have placeholders for problem-specific values
3. Success Tracking: Templates track their success rate for prioritization
4. Persistence: Skills survive across sessions for cumulative learning

This implements the "Skill Crystallization" concept from LEGO-Prover,
enabling the system to learn from successful solutions and apply
similar patterns to new problems.

Curriculum Learning:
As warmup problems are solved, their crystallized skills build a
foundation that accelerates solving harder problems. This creates
a virtuous cycle of learning.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import json
import hashlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings
from agentic_z3.utils.logger import get_logger, LogCategory

logger = get_logger(__name__)

# Lazy import chromadb to handle missing dependency gracefully
chromadb = None


def _get_chromadb():
    """Lazy load chromadb to handle missing dependency."""
    global chromadb
    if chromadb is None:
        try:
            import chromadb as _chromadb
            chromadb = _chromadb
        except ImportError:
            logger.warning(
                "chromadb not installed - skill library will use in-memory fallback",
                category=LogCategory.SYSTEM
            )
    return chromadb


@dataclass
class SkillTemplate:
    """
    A reusable SMT solving pattern extracted from successful solutions.
    
    Skills are parameterized templates that capture the structure of
    a solution without specific values. For example:
    
    Original code:
        solver.assert_and_track(x <= 100, "c_bound")
        solver.assert_and_track(x + y == 50, "c_sum")
    
    Skill template:
        solver.assert_and_track(x <= {{UPPER_BOUND}}, "c_bound")
        solver.assert_and_track(x + y == {{TARGET_SUM}}, "c_sum")
    
    Attributes:
        template_name: Unique identifier for the skill
        description: Natural language description of what this skill does
        parameters: List of placeholder names (UPPER_BOUND, TARGET_SUM, etc.)
        skeleton_code: The parameterized Z3 code template
        applicable_patterns: Problem types this skill applies to
        success_count: Number of times this skill contributed to success
        use_count: Total times this skill was retrieved
        metadata: Additional tracking data
    """
    template_name: str
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    skeleton_code: str = ""
    applicable_patterns: list[str] = field(default_factory=list)
    success_count: int = 0
    use_count: int = 0
    metadata: dict = field(default_factory=dict)
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of this skill."""
        if self.use_count == 0:
            return 0.5  # Neutral prior for unused skills
        return self.success_count / self.use_count
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "template_name": self.template_name,
            "description": self.description,
            "parameters": self.parameters,
            "skeleton_code": self.skeleton_code,
            "applicable_patterns": self.applicable_patterns,
            "success_count": self.success_count,
            "use_count": self.use_count,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SkillTemplate":
        """Create from dictionary."""
        return cls(
            template_name=data.get("template_name", ""),
            description=data.get("description", ""),
            parameters=data.get("parameters", []),
            skeleton_code=data.get("skeleton_code", ""),
            applicable_patterns=data.get("applicable_patterns", []),
            success_count=data.get("success_count", 0),
            use_count=data.get("use_count", 0),
            metadata=data.get("metadata", {})
        )
    
    def instantiate(self, param_values: dict[str, Any]) -> str:
        """
        Instantiate the template with specific parameter values.
        
        Args:
            param_values: Mapping of parameter names to values
            
        Returns:
            Code with placeholders replaced by values
        """
        code = self.skeleton_code
        for param, value in param_values.items():
            placeholder = "{{" + param + "}}"
            code = code.replace(placeholder, str(value))
        return code


class SkillLibrary:
    """
    ChromaDB-backed storage for SMT skill templates.
    
    The skill library enables curriculum learning by:
    1. Storing successful solution patterns
    2. Retrieving relevant skills for new problems
    3. Tracking success rates for prioritization
    4. Persisting across sessions for cumulative learning
    
    Vector Search:
    Skills are embedded using ChromaDB's default embedding model.
    Retrieval finds semantically similar skills based on:
    - Problem description
    - Template description
    - Applicable patterns
    
    If ChromaDB is not available, falls back to simple in-memory storage
    with keyword-based retrieval.
    
    Attributes:
        persist_path: Directory for ChromaDB persistence
        collection: ChromaDB collection for skills
        _fallback_storage: In-memory fallback when ChromaDB unavailable
    """
    
    COLLECTION_NAME = "agentic_z3_skills"
    
    def __init__(self, persist_path: Optional[str] = None):
        """
        Initialize the skill library.
        
        Args:
            persist_path: Override default persistence path
        """
        self.persist_path = persist_path or settings.CHROMA_PERSIST_PATH
        self._fallback_storage: dict[str, SkillTemplate] = {}
        self._client = None
        self._collection = None
        
        self._initialize_storage()
    
    def _initialize_storage(self) -> None:
        """Initialize ChromaDB or fallback storage."""
        chroma = _get_chromadb()
        
        if chroma is not None:
            try:
                # Ensure directory exists
                os.makedirs(self.persist_path, exist_ok=True)
                
                # Initialize persistent client
                self._client = chroma.PersistentClient(path=self.persist_path)
                
                # Get or create collection
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"description": "Agentic-Z3 SMT skill templates"}
                )
                
                logger.info(
                    f"Skill library initialized with ChromaDB at {self.persist_path}",
                    category=LogCategory.SYSTEM
                )
                return
                
            except Exception as e:
                logger.warning(
                    f"ChromaDB initialization failed: {e}, using fallback",
                    category=LogCategory.SYSTEM
                )
        
        # Fallback to in-memory storage
        logger.info(
            "Using in-memory skill storage (no persistence)",
            category=LogCategory.SYSTEM
        )
    
    def store(
        self, 
        template: SkillTemplate,
        metadata: Optional[dict] = None
    ) -> str:
        """
        Store a skill template in the library.
        
        If a template with the same name exists, it will be updated.
        
        Args:
            template: The skill template to store
            metadata: Additional metadata for search
            
        Returns:
            The ID of the stored skill
        """
        # Generate stable ID from template name
        skill_id = self._generate_id(template.template_name)
        
        # Merge metadata
        if metadata:
            template.metadata.update(metadata)
        
        # Create search text for embedding
        search_text = self._create_search_text(template)
        
        if self._collection is not None:
            try:
                # Store in ChromaDB
                self._collection.upsert(
                    ids=[skill_id],
                    documents=[search_text],
                    metadatas=[{
                        "template_json": json.dumps(template.to_dict()),
                        "template_name": template.template_name,
                        "success_rate": template.get_success_rate()
                    }]
                )
                
                logger.info(
                    f"Stored skill '{template.template_name}' (id={skill_id})",
                    category=LogCategory.SYSTEM
                )
                return skill_id
                
            except Exception as e:
                logger.warning(
                    f"ChromaDB store failed: {e}, using fallback",
                    category=LogCategory.SYSTEM
                )
        
        # Fallback storage
        self._fallback_storage[skill_id] = template
        return skill_id
    
    def retrieve(
        self, 
        query: str,
        top_k: int = 3
    ) -> list[SkillTemplate]:
        """
        Retrieve similar skill templates for a query.
        
        Uses semantic search to find skills that may be relevant
        to the given problem description.
        
        Args:
            query: Problem description or search query
            top_k: Maximum number of skills to return
            
        Returns:
            List of relevant SkillTemplates, sorted by relevance
        """
        if self._collection is not None:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, self._collection.count() or 1)
                )
                
                templates = []
                if results and results.get("metadatas"):
                    for metadata in results["metadatas"][0]:
                        template_json = metadata.get("template_json", "{}")
                        template = SkillTemplate.from_dict(json.loads(template_json))
                        templates.append(template)
                
                logger.debug(
                    f"Retrieved {len(templates)} skills for query",
                    category=LogCategory.SYSTEM
                )
                return templates
                
            except Exception as e:
                logger.warning(
                    f"ChromaDB retrieve failed: {e}, using fallback",
                    category=LogCategory.SYSTEM
                )
        
        # Fallback: Simple keyword matching
        return self._fallback_retrieve(query, top_k)
    
    def update_success_rate(self, skill_id: str, success: bool) -> None:
        """
        Update the success tracking for a skill.
        
        Called after a skill was retrieved and used. Updates the
        success/use counts to track effectiveness.
        
        Args:
            skill_id: The skill's ID
            success: Whether the skill contributed to success
        """
        if self._collection is not None:
            try:
                # Get current skill
                result = self._collection.get(ids=[skill_id])
                
                if result and result.get("metadatas"):
                    metadata = result["metadatas"][0]
                    template = SkillTemplate.from_dict(
                        json.loads(metadata.get("template_json", "{}"))
                    )
                    
                    # Update counts
                    template.use_count += 1
                    if success:
                        template.success_count += 1
                    
                    # Re-store with updated counts
                    self.store(template)
                    return
                    
            except Exception as e:
                logger.warning(
                    f"Failed to update skill success rate: {e}",
                    category=LogCategory.SYSTEM
                )
        
        # Fallback update
        if skill_id in self._fallback_storage:
            self._fallback_storage[skill_id].use_count += 1
            if success:
                self._fallback_storage[skill_id].success_count += 1
    
    def get_all_skills(self) -> list[SkillTemplate]:
        """
        Retrieve all stored skills.
        
        Useful for debugging and skill library analysis.
        """
        if self._collection is not None:
            try:
                results = self._collection.get()
                
                templates = []
                if results and results.get("metadatas"):
                    for metadata in results["metadatas"]:
                        template_json = metadata.get("template_json", "{}")
                        template = SkillTemplate.from_dict(json.loads(template_json))
                        templates.append(template)
                
                return templates
                
            except Exception as e:
                logger.warning(
                    f"Failed to get all skills: {e}",
                    category=LogCategory.SYSTEM
                )
        
        return list(self._fallback_storage.values())
    
    def clear(self) -> None:
        """Clear all skills from the library."""
        if self._collection is not None:
            try:
                # Delete and recreate collection
                self._client.delete_collection(self.COLLECTION_NAME)
                self._collection = self._client.create_collection(
                    name=self.COLLECTION_NAME
                )
            except Exception as e:
                logger.warning(f"Failed to clear ChromaDB: {e}")
        
        self._fallback_storage.clear()
        logger.info("Skill library cleared", category=LogCategory.SYSTEM)
    
    def _generate_id(self, name: str) -> str:
        """Generate a stable ID from template name."""
        return hashlib.sha256(name.encode()).hexdigest()[:16]
    
    def _create_search_text(self, template: SkillTemplate) -> str:
        """Create searchable text for embedding."""
        parts = [
            template.template_name,
            template.description,
            " ".join(template.applicable_patterns),
            " ".join(template.parameters)
        ]
        return " | ".join(filter(None, parts))
    
    def _fallback_retrieve(
        self, 
        query: str,
        top_k: int
    ) -> list[SkillTemplate]:
        """
        Simple keyword-based retrieval when ChromaDB unavailable.
        
        Scores templates by keyword overlap with query.
        """
        query_words = set(query.lower().split())
        
        scored = []
        for template in self._fallback_storage.values():
            search_text = self._create_search_text(template).lower()
            template_words = set(search_text.split())
            
            overlap = len(query_words & template_words)
            if overlap > 0:
                # Boost by success rate
                score = overlap * (0.5 + template.get_success_rate())
                scored.append((score, template))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [t for _, t in scored[:top_k]]


