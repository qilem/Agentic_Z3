"""
Structured Logging Module for Agentic-Z3

Provides differentiated logging for different system components:
- AGENT: LLM agent thoughts and reasoning (verbose, for debugging)
- Z3: Solver output and execution results
- SYSTEM: High-level status messages and orchestration

Features:
- Color-coded terminal output for quick visual parsing
- JSON mode for structured log aggregation in production
- Category-based filtering to focus on specific components
"""

import logging
import json
import sys
from enum import Enum
from typing import Optional, Any
from datetime import datetime


class LogCategory(Enum):
    """
    Log categories for filtering and visual differentiation.
    
    Each category maps to:
    - A prefix tag for terminal output
    - A color code for visual distinction
    - Independent verbosity control
    """
    AGENT = "AGENT"      # LLM thoughts, prompts, responses
    Z3 = "Z3"            # Solver execution, results, errors
    SYSTEM = "SYSTEM"    # Engine state, orchestration, lifecycle


# ANSI color codes for terminal output
COLORS = {
    LogCategory.AGENT: "\033[94m",   # Blue - for AI reasoning
    LogCategory.Z3: "\033[93m",      # Yellow - for solver output
    LogCategory.SYSTEM: "\033[92m",  # Green - for system status
    "RESET": "\033[0m",
    "ERROR": "\033[91m",             # Red - for errors
    "WARNING": "\033[95m",           # Magenta - for warnings
}


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that supports both human-readable and JSON output.
    
    In terminal mode (default):
        [SYSTEM] 2024-01-15 10:30:45 | INFO | Engine initialized
        [AGENT] 2024-01-15 10:30:46 | DEBUG | Architect reasoning: ...
    
    In JSON mode (for log aggregation):
        {"timestamp": "...", "category": "SYSTEM", "level": "INFO", "message": "..."}
    """
    
    def __init__(self, json_mode: bool = False, use_colors: bool = True):
        super().__init__()
        self.json_mode = json_mode
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        # Extract category from record (default to SYSTEM)
        category = getattr(record, 'category', LogCategory.SYSTEM)
        if isinstance(category, str):
            category = LogCategory[category]
        
        timestamp = datetime.now().isoformat(timespec='seconds')
        
        if self.json_mode:
            log_entry = {
                "timestamp": timestamp,
                "category": category.value,
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            # Include extra fields if present
            if hasattr(record, 'extra_data'):
                log_entry["data"] = record.extra_data
            return json.dumps(log_entry)
        
        # Human-readable format
        level = record.levelname
        message = record.getMessage()
        
        if self.use_colors:
            # Apply category color
            cat_color = COLORS.get(category, "")
            reset = COLORS["RESET"]
            
            # Apply level-specific coloring for errors/warnings
            if level == "ERROR":
                level_color = COLORS["ERROR"]
            elif level == "WARNING":
                level_color = COLORS["WARNING"]
            else:
                level_color = ""
            
            return (
                f"{cat_color}[{category.value}]{reset} "
                f"{timestamp} | {level_color}{level:7}{reset} | {message}"
            )
        
        return f"[{category.value}] {timestamp} | {level:7} | {message}"


class CategoryAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically injects category into log records.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Starting engine", category=LogCategory.SYSTEM)
        logger.debug("Agent thinking...", category=LogCategory.AGENT)
    """
    
    def process(self, msg: str, kwargs: dict) -> tuple:
        # Extract category from kwargs or use default
        category = kwargs.pop('category', LogCategory.SYSTEM)
        extra = kwargs.get('extra', {})
        extra['category'] = category
        kwargs['extra'] = extra
        return msg, kwargs


# Module-level logger cache
_loggers: dict[str, CategoryAdapter] = {}

# Global configuration (set by config module)
_json_mode: bool = False
_show_agent_thoughts: bool = True
_log_level: str = "INFO"


def configure_logging(
    json_mode: bool = False,
    show_agent_thoughts: bool = True,
    log_level: str = "INFO"
) -> None:
    """
    Configure global logging settings.
    
    Should be called once at application startup, typically from config module.
    
    Args:
        json_mode: If True, output logs in JSON format for aggregation
        show_agent_thoughts: If False, suppress verbose AGENT category logs
        log_level: Global minimum log level (DEBUG, INFO, WARNING, ERROR)
    """
    global _json_mode, _show_agent_thoughts, _log_level
    _json_mode = json_mode
    _show_agent_thoughts = show_agent_thoughts
    _log_level = log_level
    
    # Reconfigure existing loggers
    for logger_adapter in _loggers.values():
        _configure_logger(logger_adapter.logger)


def _configure_logger(logger: logging.Logger) -> None:
    """Apply current global configuration to a logger."""
    logger.setLevel(getattr(logging, _log_level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add configured handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter(json_mode=_json_mode))
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False


def get_logger(name: str) -> CategoryAdapter:
    """
    Get a configured logger instance for the given module name.
    
    Returns a CategoryAdapter that supports the 'category' keyword argument
    for all log methods.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        CategoryAdapter wrapping a configured Logger
        
    Example:
        logger = get_logger(__name__)
        logger.info("Engine started", category=LogCategory.SYSTEM)
        logger.debug("Blueprint generated", category=LogCategory.AGENT, 
                     extra={'extra_data': {'variables': 3}})
    """
    if name not in _loggers:
        logger = logging.getLogger(f"agentic_z3.{name}")
        _configure_logger(logger)
        _loggers[name] = CategoryAdapter(logger, {})
    
    return _loggers[name]


class AgentThoughtFilter(logging.Filter):
    """
    Filter that can suppress AGENT category logs when not needed.
    
    Agent thoughts are verbose and useful for debugging but may clutter
    production logs. This filter allows selectively hiding them.
    """
    
    def __init__(self, show_thoughts: bool = True):
        super().__init__()
        self.show_thoughts = show_thoughts
    
    def filter(self, record: logging.LogRecord) -> bool:
        if not self.show_thoughts:
            category = getattr(record, 'category', None)
            if category == LogCategory.AGENT:
                return False
        return True









