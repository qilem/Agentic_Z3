"""
Base Agent Module for Agentic-Z3

Provides the foundation for all agent classes:
- LLM client initialization and API calls
- Conversation history management
- Soft reset capability (clear history while preserving context)
- JSON parsing and validation

All three agents (Architect, Worker, Coach) inherit from BaseAgent
and customize their behavior through system prompts and output parsing.
"""

import json
import re
from typing import Optional, Any
from abc import ABC, abstractmethod
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings
from agentic_z3.utils.logger import get_logger, LogCategory

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Agentic-Z3 system.
    
    Provides common functionality:
    - OpenAI client management
    - Conversation history tracking
    - LLM API calls with retry and error handling
    - JSON extraction from LLM responses
    - Soft reset mechanism for TTRL
    
    Subclasses must implement:
    - system_prompt property: Define agent's role and output format
    - Specific methods for their domain (plan, generate, diagnose, etc.)
    
    The conversation history is managed as a list of messages that can be:
    - Extended with new user/assistant turns
    - Cleared for soft reset (while preserving key context)
    - Inspected for debugging
    
    Attributes:
        client: OpenAI API client
        model: LLM model name (e.g., "gpt-4o")
        temperature: Current sampling temperature
        base_temperature: Default temperature (restored after soft reset cooldown)
        max_tokens: Maximum response tokens
        messages: Conversation history
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            model: Override default LLM model
            temperature: Override default temperature
            max_tokens: Override default max tokens
        """
        # Initialize OpenAI client
        if not settings.OPENAI_API_KEY:
            logger.warning(
                "OPENAI_API_KEY not set - agent will not function",
                category=LogCategory.SYSTEM
            )
        
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = model or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.base_temperature = self.temperature
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        
        # Conversation history starts with system prompt
        self.messages: list[dict[str, str]] = []
        self._initialize_conversation()
        
        logger.debug(
            f"{self.__class__.__name__} initialized with model={self.model}",
            category=LogCategory.AGENT
        )
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Return the system prompt defining this agent's role.
        
        Subclasses must implement this to define their specific
        persona, responsibilities, and output format.
        """
        pass
    
    def _initialize_conversation(self) -> None:
        """Set up the conversation with the system prompt."""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def _call_llm(
        self,
        user_message: str,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        add_to_history: bool = True
    ) -> str:
        """
        Make an LLM API call and return the response.
        
        Manages conversation history and handles API errors gracefully.
        
        Args:
            user_message: The user message to send
            temperature: Override temperature for this call
            json_mode: If True, request JSON-formatted response
            add_to_history: If True, add user message and response to history
            
        Returns:
            The assistant's response text
            
        Raises:
            Exception: If API call fails after retries
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Prepare messages for this call
        call_messages = self.messages.copy()
        call_messages.append({"role": "user", "content": user_message})
        
        logger.debug(
            f"LLM call: {len(call_messages)} messages, temp={temp}, json={json_mode}",
            category=LogCategory.AGENT
        )
        
        try:
            # Build API call parameters
            params = {
                "model": self.model,
                "messages": call_messages,
                "temperature": temp,
                "max_tokens": self.max_tokens,
            }
            
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**params)
            
            assistant_message = response.choices[0].message.content
            
            # Update history if requested
            if add_to_history:
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": assistant_message})
            
            logger.debug(
                f"LLM response: {assistant_message[:200]}...",
                category=LogCategory.AGENT
            )
            
            return assistant_message
            
        except Exception as e:
            logger.error(
                f"LLM API call failed: {e}",
                category=LogCategory.AGENT
            )
            raise
    
    def _reset_conversation(self) -> None:
        """
        Clear conversation history, keeping only the system prompt.
        
        This is the low-level reset - subclasses should typically use
        perform_soft_reset() which preserves essential context.
        """
        self._initialize_conversation()
        logger.debug(
            f"{self.__class__.__name__} conversation reset",
            category=LogCategory.AGENT
        )
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from LLM response text.
        
        Handles cases where JSON is embedded in markdown code blocks
        or surrounded by other text.
        
        Args:
            text: The LLM response text
            
        Returns:
            Parsed JSON as a dict, or None if extraction fails
        """
        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract from markdown code block
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',       # ``` ... ```
            r'\{[\s\S]*\}',                   # Raw JSON object
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        logger.warning(
            f"Failed to extract JSON from response: {text[:100]}...",
            category=LogCategory.AGENT
        )
        return None
    
    def _extract_code(self, text: str, language: str = "python") -> str:
        """
        Extract code from LLM response text.
        
        Handles markdown code blocks and returns clean code.
        
        Args:
            text: The LLM response text
            language: Expected language (for code block matching)
            
        Returns:
            Extracted code, or original text if no code block found
        """
        # Try language-specific code block
        pattern = rf'```{language}\s*([\s\S]*?)\s*```'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        pattern = r'```\s*([\s\S]*?)\s*```'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        
        # Return original text (might be raw code)
        return text.strip()
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation for debugging.
        
        Returns:
            String summarizing message count and last few exchanges
        """
        num_messages = len(self.messages)
        if num_messages <= 1:
            return f"{self.__class__.__name__}: No conversation yet"
        
        last_messages = self.messages[-3:]
        summary_lines = [f"{self.__class__.__name__}: {num_messages} messages"]
        for msg in last_messages:
            role = msg["role"]
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            summary_lines.append(f"  [{role}] {content}")
        
        return "\n".join(summary_lines)
    
    def add_context_message(self, content: str, role: str = "user") -> None:
        """
        Add a context message to the conversation without triggering LLM call.
        
        Useful for injecting context (e.g., failure summaries, retrieved skills)
        before making the next LLM call.
        
        Args:
            content: The message content
            role: Message role (usually "user" or "system")
        """
        self.messages.append({"role": role, "content": content})


