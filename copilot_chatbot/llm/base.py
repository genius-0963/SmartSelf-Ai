"""
SmartShelf AI - LLM Base Class

Base class for LLM client implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMClientBase(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config):
        """Initialize LLM client with configuration."""
        self.config = config
        self.model_name = config.model
    
    @abstractmethod
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]] = None,
        conversation_context: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generate response to query with context."""
        pass
