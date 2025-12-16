"""
Base Agent Module for Agentic-Z3

Provides the foundation for all agent classes:
- LLM client initialization and API calls
- Conversation history management
- Soft reset capability (clear history while preserving context)
- JSON parsing and validation
- Rate limiting to avoid API throttling

All three agents (Architect, Worker, Coach) inherit from BaseAgent
and customize their behavior through system prompts and output parsing.
"""

import json
import re
import time
import threading
from typing import Optional, Any
from abc import ABC, abstractmethod
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import settings
from agentic_z3.utils.logger import get_logger, LogCategory


# Simple rate limiter for agent LLM calls
class AgentRateLimiter:
    """Thread-safe rate limiter for agent LLM calls."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, requests_per_minute: int = 50, tokens_per_minute: int = 25000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_times = []
        self.lock = threading.Lock()
        self.max_retries = 5
        self.base_delay = 2.0
    
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def wait_if_needed(self):
        """Wait if we're approaching rate limits."""
        with self.lock:
            now = time.time()
            # Remove old requests (older than 60 seconds)
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            # Wait if at request limit
            if len(self.request_times) >= self.rpm_limit:
                oldest = self.request_times[0]
                wait_time = 60 - (now - oldest) + 1
                if wait_time > 0:
                    logger.debug(f"Rate limit: waiting {wait_time:.1f}s", category=LogCategory.AGENT)
                    time.sleep(wait_time)
            
            self.request_times.append(time.time())
    
    def call_with_retry(self, func, *args, **kwargs):
        """Call a function with exponential backoff on rate limit errors."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self.wait_if_needed()
                return func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a rate limit error
                if "429" in str(e) or "rate limit" in error_str:
                    # Parse wait time from error message if available
                    wait_time = self._parse_retry_after(str(e))
                    if wait_time is None:
                        wait_time = min(self.base_delay * (2 ** attempt), 60.0)
                    
                    logger.warning(
                        f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{self.max_retries}",
                        category=LogCategory.AGENT
                    )
                    time.sleep(wait_time)
                    continue
                
                # For other errors, don't retry
                raise
        
        raise last_error
    
    def _parse_retry_after(self, error_message: str) -> Optional[float]:
        """Parse retry-after time from OpenAI error message."""
        match = re.search(r'try again in (\d+\.?\d*)s', error_message)
        if match:
            return float(match.group(1)) + 1.0
        return None

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
        model: LLM model name (e.g., "gpt-5.2")
        temperature: Current sampling temperature
        base_temperature: Default temperature (restored after soft reset cooldown)
        max_tokens: Maximum response tokens
        messages: Conversation history
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        history_mode: Optional[str] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            model: Override default LLM model
            temperature: Override default temperature
            max_tokens: Override default max tokens
            history_mode: Override default history mode ('stateful', 'trimmed', 'stateless')
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
        
        # History management settings
        self.history_mode = history_mode or settings.AGENT_HISTORY_MODE
        self.max_history_messages = settings.AGENT_MAX_HISTORY_MESSAGES
        self.max_history_chars = settings.AGENT_MAX_HISTORY_CHARS
        
        # Conversation history starts with system prompt
        self.messages: list[dict[str, str]] = []
        self._initialize_conversation()
        
        logger.debug(
            f"{self.__class__.__name__} initialized with model={self.model}, history_mode={self.history_mode}",
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
        Uses rate limiting with exponential backoff for 429 errors.
        Respects history_mode setting for context management.
        
        Args:
            user_message: The user message to send
            temperature: Override temperature for this call
            json_mode: If True, request JSON-formatted response
            add_to_history: If True, add user message and response to history
                           (ignored in stateless mode)
            
        Returns:
            The assistant's response text
            
        Raises:
            Exception: If API call fails after retries
        """
        temp = temperature if temperature is not None else self.temperature
        
        # In stateless mode, only send system prompt + current message
        if self.history_mode == "stateless":
            call_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
        else:
            # In stateful/trimmed modes, use full history
            # Apply trimming before adding new message if in trimmed mode
            if self.history_mode == "trimmed":
                self._trim_history()
            
            # Enforce char limit as safety guard
            self._enforce_char_limit()
            
            # Prepare messages for this call
            call_messages = self.messages.copy()
            call_messages.append({"role": "user", "content": user_message})
        
        logger.debug(
            f"LLM call: {len(call_messages)} messages, mode={self.history_mode}, temp={temp}, json={json_mode}",
            category=LogCategory.AGENT
        )
        
        # Build API call parameters
        params = {
            "model": self.model,
            "messages": call_messages,
            "temperature": temp,
            "max_completion_tokens": self.max_tokens,
        }
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        
        # Use rate limiter for the API call
        rate_limiter = AgentRateLimiter.get_instance()
        
        def _make_api_call():
            return self.client.chat.completions.create(**params)
        
        try:
            response = rate_limiter.call_with_retry(_make_api_call)
            
            assistant_message = response.choices[0].message.content
            
            # Update history if requested and not in stateless mode
            if add_to_history and self.history_mode != "stateless":
                self.messages.append({"role": "user", "content": user_message})
                self.messages.append({"role": "assistant", "content": assistant_message})
            
            logger.debug(
                f"LLM response: {assistant_message[:200]}...",
                category=LogCategory.AGENT
            )
            
            return assistant_message
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for context_length_exceeded error
            if "context_length_exceeded" in error_str or "input tokens exceed" in error_str:
                logger.warning(
                    f"Context length exceeded, attempting recovery with stateless call",
                    category=LogCategory.AGENT
                )
                
                # Retry with stateless call (system prompt + current message only)
                try:
                    stateless_messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                    params["messages"] = stateless_messages
                    
                    response = rate_limiter.call_with_retry(_make_api_call)
                    assistant_message = response.choices[0].message.content
                    
                    # Don't add to history after emergency recovery
                    logger.info(
                        f"Context overflow recovery successful",
                        category=LogCategory.AGENT
                    )
                    
                    return assistant_message
                    
                except Exception as retry_error:
                    logger.error(
                        f"Context overflow recovery failed: {retry_error}",
                        category=LogCategory.AGENT
                    )
                    raise retry_error
            
            # For other errors, re-raise
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
    
    def _get_history_char_count(self) -> int:
        """Get total character count in conversation history."""
        return sum(len(msg.get("content", "")) for msg in self.messages)
    
    def _trim_history(self) -> None:
        """
        Trim conversation history to keep only recent messages.
        
        Keeps system prompt + last N messages according to max_history_messages.
        Used in 'trimmed' history mode.
        """
        if len(self.messages) <= self.max_history_messages + 1:  # +1 for system prompt
            return
        
        # Keep system prompt (first message) + last N messages
        system_prompt = self.messages[0]
        recent_messages = self.messages[-(self.max_history_messages):]
        self.messages = [system_prompt] + recent_messages
        
        logger.debug(
            f"Trimmed history to {len(self.messages)} messages",
            category=LogCategory.AGENT
        )
    
    def _enforce_char_limit(self) -> None:
        """
        Emergency trim if history exceeds character limit.
        
        Progressively drops older messages (keeping system prompt) until
        under the char limit. This is a safety guard for all modes.
        """
        char_count = self._get_history_char_count()
        if char_count <= self.max_history_chars:
            return
        
        logger.warning(
            f"History exceeds {self.max_history_chars} chars ({char_count}), emergency trimming",
            category=LogCategory.AGENT
        )
        
        # Keep system prompt and progressively drop oldest messages
        system_prompt = self.messages[0]
        remaining = self.messages[1:]
        
        while remaining and self._get_history_char_count() > self.max_history_chars:
            remaining = remaining[1:]  # Drop oldest non-system message
            self.messages = [system_prompt] + remaining
        
        logger.debug(
            f"Emergency trim complete: {len(self.messages)} messages, {self._get_history_char_count()} chars",
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




