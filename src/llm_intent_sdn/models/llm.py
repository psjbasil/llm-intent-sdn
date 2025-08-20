"""LLM-related data models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class LLMMessage(BaseModel):
    """LLM message model."""
    
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "role": "user",
                "content": "Route traffic from h1 to h2 through the fastest path"
            }
        }


class LLMRequest(BaseModel):
    """LLM request model."""
    
    messages: List[LLMMessage] = Field(..., description="List of messages")
    model: str = Field(default="deepseek/deepseek-chat", description="LLM model name")
    temperature: float = Field(default=0.1, description="Temperature for response generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens in response")
    system_prompt: Optional[str] = Field(None, description="System prompt override")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Route traffic from h1 to h2 through the fastest path"
                    }
                ],
                "model": "deepseek/deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 1000
            }
        }


class LLMResponse(BaseModel):
    """LLM response model."""
    
    content: str = Field(..., description="Generated response content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Token usage statistics")
    finish_reason: str = Field(default="stop", description="Reason for completion")
    response_time_ms: int = Field(..., description="Response time in milliseconds")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "content": "To route traffic from h1 to h2 through the fastest path, I will install flow rules...",
                "model": "deepseek/deepseek-chat",
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 200,
                    "total_tokens": 350
                },
                "finish_reason": "stop",
                "response_time_ms": 1500
            }
        }


class IntentAnalysis(BaseModel):
    """Intent analysis result from LLM."""
    
    intent_type: str = Field(..., description="Detected intent type")
    confidence: float = Field(..., description="Confidence score (0-1)")
    extracted_entities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted entities and parameters"
    )
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Suggested network actions"
    )
    reasoning: str = Field(..., description="LLM's reasoning process")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "intent_type": "routing",
                "confidence": 0.95,
                "extracted_entities": {
                    "source": "h1",
                    "destination": "h2",
                    "optimization": "fastest_path"
                },
                "suggested_actions": [
                    "calculate_shortest_path",
                    "install_flow_rules",
                    "verify_connectivity"
                ],
                "reasoning": "The user wants to establish optimal routing between two hosts..."
            }
        } 