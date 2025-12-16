"""
Agentic-Z3 Configuration Module

Centralized configuration for the autonomous SMT solving framework.
Uses Pydantic Settings for environment variable loading with sensible defaults.

Environment variables can be set directly or via a .env file in the project root.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """
    Global configuration settings for Agentic-Z3.
    
    All settings can be overridden via environment variables with the same name
    (case-insensitive). For example, set MAX_TTRL_RETRIES=10 in environment.
    """
    
    # ==========================================================================
    # TTRL (Test-Time Reinforcement Learning) Settings
    # ==========================================================================
    
    MAX_TTRL_RETRIES: int = Field(
        default=5,
        description="Maximum number of retry iterations in the TTRL loop before giving up"
    )
    
    SOFT_RESET_THRESHOLD: int = Field(
        default=3,
        description="Number of consecutive UNKNOWN/timeout results before triggering soft reset. "
                    "Soft reset clears LLM conversation history but retains failure summaries "
                    "to force exploration of new solution strategies."
    )
    
    # ==========================================================================
    # Z3 Solver Settings
    # ==========================================================================
    
    Z3_TIMEOUT: int = Field(
        default=5000,
        description="Z3 solver timeout in milliseconds. Prevents infinite loops on hard problems."
    )
    
    Z3_PROBE_TIMEOUT: int = Field(
        default=1000,
        description="Shorter timeout for type probing phase (ms). Probes should be fast."
    )
    
    # ==========================================================================
    # LLM Settings
    # ==========================================================================
    
    OPENAI_API_KEY: Optional[str] = Field(
        default='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',
        description="OpenAI API key. Required for LLM-based agents. Set via OPENAI_API_KEY env var or .env file."
    )
    
    LLM_MODEL: str = Field(
        default="gpt-5.2",
        description="Default LLM model for all agents"
    )
    
    LLM_TEMPERATURE: float = Field(
        default=0.2,
        description="Default temperature for LLM generation. Low for deterministic output."
    )
    
    LLM_TEMPERATURE_SOFT_RESET: float = Field(
        default=0.7,
        description="Elevated temperature after soft reset to force exploration diversity"
    )
    
    LLM_MAX_TOKENS: int = Field(
        default=4096,
        description="Maximum tokens for LLM response"
    )
    
    # ==========================================================================
    # Agent Conversation History Settings
    # ==========================================================================
    
    AGENT_HISTORY_MODE: str = Field(
        default="stateful",
        description="Agent conversation history policy: 'stateful' (full history), "
                    "'trimmed' (keep last N messages), 'stateless' (no history). "
                    "Use 'stateless' for benchmark runs to prevent context overflow."
    )
    
    AGENT_MAX_HISTORY_MESSAGES: int = Field(
        default=40,
        description="Maximum messages to keep in 'trimmed' history mode (system prompt + last N turns)"
    )
    
    AGENT_MAX_HISTORY_CHARS: int = Field(
        default=200000,
        description="Maximum total characters in conversation history (safety guard for all modes)"
    )
    
    # ==========================================================================
    # Memory / Skill Library Settings
    # ==========================================================================
    
    CHROMA_PERSIST_PATH: str = Field(
        default="./.agentic_z3_data/chroma_db",
        description="Path to persist ChromaDB skill library"
    )
    
    SKILL_RETRIEVAL_TOP_K: int = Field(
        default=3,
        description="Number of similar skills to retrieve from the library"
    )
    
    ENABLE_SKILL_LIBRARY: bool = Field(
        default=True,
        description="If True, use skill library for retrieval and storage. "
                    "Disable for benchmark runs to reduce overhead."
    )
    
    ENABLE_SKILL_CRYSTALLIZATION: bool = Field(
        default=True,
        description="If True, crystallize successful solutions into skill templates. "
                    "Disable for benchmark runs to save LLM calls."
    )
    
    # ==========================================================================
    # Logging Settings
    # ==========================================================================
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Global log level: DEBUG, INFO, WARNING, ERROR"
    )
    
    LOG_JSON_MODE: bool = Field(
        default=False,
        description="If True, output logs in JSON format for structured aggregation"
    )
    
    LOG_SHOW_AGENT_THOUGHTS: bool = Field(
        default=True,
        description="If True, display verbose agent reasoning in logs"
    )
    
    # ==========================================================================
    # Engine Settings
    # ==========================================================================
    
    MAX_PLANNING_ITERATIONS: int = Field(
        default=3,
        description="Maximum times to re-plan after UNSAT diagnosis before giving up"
    )
    
    ENABLE_CURRICULUM_WARMUP: bool = Field(
        default=True,
        description="If True, generate warmup problems for complex inputs"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance - import this in other modules
settings = Settings()




