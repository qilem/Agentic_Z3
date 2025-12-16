"""
Rate Limiter for OpenAI API Calls

Provides thread-safe rate limiting with:
- Token-per-minute (TPM) tracking
- Request-per-minute (RPM) tracking  
- Exponential backoff retry
- Global singleton for cross-runner coordination
"""

import time
import threading
from typing import Optional, Callable, Any
from dataclasses import dataclass
from functools import wraps


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    # OpenAI limits (default for gpt-5.2)
    tokens_per_minute: int = 25000  # Leave some buffer from 30k limit
    requests_per_minute: int = 500
    
    # Retry settings
    max_retries: int = 5
    base_delay: float = 2.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay
    
    # Concurrency settings
    recommended_max_workers: int = 2  # Recommended for rate-limited APIs


class TokenBucket:
    """Thread-safe token bucket rate limiter."""
    
    def __init__(self, tokens_per_minute: int):
        self.capacity = tokens_per_minute
        self.tokens = tokens_per_minute
        self.refill_rate = tokens_per_minute / 60.0  # tokens per second
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int, timeout: float = 120.0) -> bool:
        """
        Try to consume tokens, blocking if necessary.
        
        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens consumed, False if timeout
        """
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            with self.lock:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
            
            # Wait outside the lock
            sleep_time = min(wait_time, deadline - time.time(), 5.0)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now


class RateLimiter:
    """
    Global rate limiter for OpenAI API calls.
    
    Usage:
        limiter = RateLimiter.get_instance()
        
        # Before making an API call
        limiter.wait_if_needed(estimated_tokens=1000)
        
        # Or use the decorator
        @limiter.with_retry
        def call_api():
            ...
    """
    
    _instance: Optional['RateLimiter'] = None
    _lock = threading.Lock()
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.token_bucket = TokenBucket(self.config.tokens_per_minute)
        self.request_times = []  # Timestamps of recent requests
        self.request_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, config: Optional[RateLimitConfig] = None) -> 'RateLimiter':
        """Get or create the global rate limiter instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the global instance (useful for testing)."""
        with cls._lock:
            cls._instance = None
    
    def wait_if_needed(self, estimated_tokens: int = 1000) -> None:
        """
        Wait if necessary to stay within rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for the request
        """
        # Wait for token budget
        self.token_bucket.consume(estimated_tokens)
        
        # Also check request rate
        with self.request_lock:
            now = time.time()
            # Remove old requests (older than 60 seconds)
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            # Wait if at request limit
            if len(self.request_times) >= self.config.requests_per_minute:
                oldest = self.request_times[0]
                wait_time = 60 - (now - oldest)
                if wait_time > 0:
                    time.sleep(wait_time)
            
            self.request_times.append(time.time())
    
    def with_retry(self, func: Callable) -> Callable:
        """
        Decorator that adds retry logic with exponential backoff.
        
        Handles 429 rate limit errors by waiting and retrying.
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_error = None
            
            for attempt in range(self.config.max_retries):
                try:
                    # Wait for rate limit before attempting
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
                            # Exponential backoff
                            wait_time = min(
                                self.config.base_delay * (2 ** attempt),
                                self.config.max_delay
                            )
                        
                        print(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{self.config.max_retries}")
                        time.sleep(wait_time)
                        continue
                    
                    # For other errors, don't retry
                    raise
            
            # All retries exhausted
            raise last_error
        
        return wrapper
    
    def _parse_retry_after(self, error_message: str) -> Optional[float]:
        """Parse retry-after time from OpenAI error message."""
        import re
        # Look for patterns like "Please try again in 13.962s"
        match = re.search(r'try again in (\d+\.?\d*)s', error_message)
        if match:
            return float(match.group(1)) + 1.0  # Add buffer
        return None


def rate_limited_api_call(func: Callable) -> Callable:
    """
    Convenience decorator for rate-limited API calls.
    
    Usage:
        @rate_limited_api_call
        def call_openai(prompt):
            return client.chat.completions.create(...)
    """
    limiter = RateLimiter.get_instance()
    return limiter.with_retry(func)


# Recommended settings for different API tiers
TIER_CONFIGS = {
    'free': RateLimitConfig(
        tokens_per_minute=10000,
        requests_per_minute=200,
        recommended_max_workers=1
    ),
    'tier1': RateLimitConfig(
        tokens_per_minute=25000,
        requests_per_minute=500,
        recommended_max_workers=2
    ),
    'tier2': RateLimitConfig(
        tokens_per_minute=80000,
        requests_per_minute=5000,
        recommended_max_workers=4
    ),
    'tier3': RateLimitConfig(
        tokens_per_minute=500000,
        requests_per_minute=5000,
        recommended_max_workers=8
    ),
}
