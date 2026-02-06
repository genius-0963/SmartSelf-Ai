"""
SmartShelf AI - Claude LLM Client

Anthropic Claude API client implementation for AI copilot.
Provides alternative to OpenAI with different reasoning capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("anthropic package not installed. Claude client will not be available.")

from .base import LLMClientBase

logger = logging.getLogger(__name__)


class ClaudeClient(LLMClientBase):
    """
    Anthropic Claude API client for AI copilot conversations.
    
    Features:
    - Strong reasoning and analytical capabilities
    - Context-aware responses
    - Conversation history management
    - Error handling and retries
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Claude client."""
        super().__init__(config)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        
        self.api_key = config.get('api_key')
        if not self.api_key:
            raise ValueError("Claude API key is required in config")
        
        self.max_tokens = config.get('max_tokens', 4000)
        self.temperature = config.get('temperature', 0.7)
        self.model_name = config.get('model', 'claude-3-sonnet-20240229')
        
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        logger.info(f"Initialized Claude client with model: {self.model_name}")
    
    async def generate_response(
        self,
        query: str,
        context: List[Dict[str, Any]] = None,
        conversation_context: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate response using Claude API.
        
        Args:
            query: User query/question
            context: Retrieved context documents (from RAG)
            conversation_context: Previous conversation messages
            
        Returns:
            Dictionary with response, metadata, and usage info
        """
        try:
            # Build the conversation messages
            messages = self._build_messages(query, context, conversation_context)
            
            # Create system prompt
            system_prompt = self._create_system_prompt(context)
            
            # Call Claude API
            response = await self._call_claude_api(messages, system_prompt)
            
            # Parse and return response
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"Error generating Claude response: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error: {str(e)}",
                'model_used': self.model_name,
                'tokens_used': 0,
                'cost': 0.0,
                'response_time': 0,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _build_messages(
        self,
        query: str,
        context: List[Dict[str, Any]] = None,
        conversation_context: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Build message list for Claude API."""
        messages = []
        
        # Add conversation history (excluding system messages)
        if conversation_context:
            for msg in conversation_context[-10:]:  # Keep last 10 messages
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
        
        # Add current query
        messages.append({
            'role': 'user',
            'content': self._format_query_with_context(query, context)
        })
        
        return messages
    
    def _create_system_prompt(self, context: List[Dict[str, Any]] = None) -> str:
        """Create system prompt for Claude."""
        base_prompt = """You are SmartShelf AI, an intelligent retail decision copilot. 

Your role is to help small retailers make data-driven business decisions using:
- Demand forecasting insights
- Pricing optimization recommendations  
- Inventory intelligence alerts
- Market analysis and trends

Guidelines:
- Provide actionable, specific recommendations
- Use data and context when available
- Be conversational but professional
- Ask clarifying questions when needed
- Focus on practical business impact
- Explain your reasoning briefly

If you don't have sufficient context or data, acknowledge this and provide general best practices."""
        
        if context:
            base_prompt += f"\n\nContext available: {len(context)} relevant documents retrieved from the retail knowledge base."
        
        return base_prompt
    
    def _format_query_with_context(self, query: str, context: List[Dict[str, Any]] = None) -> str:
        """Format user query with retrieved context."""
        if not context:
            return query
        
        context_text = "\n\n".join([
            f"Context {i+1}:\n{doc.get('content', doc.get('text', ''))}"
            for i, doc in enumerate(context[:5])  # Limit to top 5 context docs
        ])
        
        return f"""Relevant Context:
{context_text}

User Question: {query}"""
    
    async def _call_claude_api(self, messages: List[Dict[str, str]], system_prompt: str) -> Any:
        """Make async call to Claude API."""
        start_time = datetime.utcnow()
        
        try:
            # Run the blocking API call in thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=messages
                )
            )
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Claude API response time: {response_time:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Claude API call failed: {str(e)}")
            raise
    
    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse Claude API response."""
        try:
            content = response.content[0].text if response.content else ""
            
            # Extract usage information
            usage = response.usage if hasattr(response, 'usage') else None
            
            return {
                'response': content,
                'model_used': self.model_name,
                'tokens_used': {
                    'input': usage.input_tokens if usage else 0,
                    'output': usage.output_tokens if usage else 0,
                    'total': (usage.input_tokens + usage.output_tokens) if usage else 0
                },
                'cost': self._calculate_cost(usage),
                'response_time': getattr(response, 'response_time', 0),
                'finish_reason': response.stop_reason if hasattr(response, 'stop_reason') else None,
                'timestamp': datetime.utcnow().isoformat(),
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error parsing Claude response: {str(e)}")
            return {
                'response': "Error parsing response from Claude.",
                'model_used': self.model_name,
                'tokens_used': 0,
                'cost': 0.0,
                'response_time': 0,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _calculate_cost(self, usage: Any) -> float:
        """Calculate cost based on token usage."""
        if not usage:
            return 0.0
        
        # Claude pricing (approximate, check current rates)
        pricing = {
            'claude-3-sonnet-20240229': {
                'input': 0.003 / 1000,  # $3 per 1M input tokens
                'output': 0.015 / 1000  # $15 per 1M output tokens
            },
            'claude-3-haiku-20240307': {
                'input': 0.00025 / 1000,  # $0.25 per 1M input tokens
                'output': 0.00125 / 1000  # $1.25 per 1M output tokens
            },
            'claude-3-opus-20240229': {
                'input': 0.015 / 1000,   # $15 per 1M input tokens
                'output': 0.075 / 1000   # $75 per 1M output tokens
            }
        }
        
        model_pricing = pricing.get(self.model_name, pricing['claude-3-sonnet-20240229'])
        
        input_cost = usage.input_tokens * model_pricing['input']
        output_cost = usage.output_tokens * model_pricing['output']
        
        return input_cost + output_cost
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Claude model."""
        return {
            'name': self.model_name,
            'provider': 'anthropic',
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'capabilities': [
                'reasoning',
                'analysis',
                'conversation',
                'context_understanding'
            ],
            'cost_per_token': {
                'input': 0.003 / 1000,
                'output': 0.015 / 1000
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check if Claude client is healthy."""
        try:
            if not ANTHROPIC_AVAILABLE:
                return {'healthy': False, 'error': 'anthropic package not installed'}
            
            if not self.api_key:
                return {'healthy': False, 'error': 'API key not configured'}
            
            return {
                'healthy': True,
                'model': self.model_name,
                'provider': 'anthropic',
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
