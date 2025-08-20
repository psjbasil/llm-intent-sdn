"""Intent-related data models."""

from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Types of network intents."""
    
    ROUTING = "routing"
    QOS = "qos"
    SECURITY = "security"
    LOAD_BALANCING = "load_balancing"
    ANOMALY_DETECTION = "anomaly_detection"
    TRAFFIC_ENGINEERING = "traffic_engineering"


class IntentStatus(str, Enum):
    """Status of intent processing."""
    
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IntentRequest(BaseModel):
    """Request model for network intent."""
    
    intent_text: str = Field(
        ..., 
        description="Natural language description of the network intent",
        min_length=1,
        max_length=1000
    )
    intent_type: Optional[IntentType] = Field(
        None,
        description="Type of the intent (optional, will be auto-detected)"
    )
    priority: int = Field(
        default=5,
        description="Priority level (1-10, 10 being highest)",
        ge=1,
        le=10
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context or parameters for the intent"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="ID of the user making the request"
    )
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "intent_text": "Route all traffic from host h1 to h2 through the fastest path",
                "intent_type": "routing",
                "priority": 7,
                "context": {"source": "h1", "destination": "h2"},
                "user_id": "admin"
            }
        }


class IntentResponse(BaseModel):
    """Response model for processed intent."""
    
    intent_id: str = Field(..., description="Unique identifier for the intent")
    status: IntentStatus = Field(..., description="Current status of the intent")
    intent_text: str = Field(..., description="Original intent text")
    intent_type: IntentType = Field(..., description="Detected or specified intent type")
    
    # LLM Analysis
    llm_interpretation: str = Field(
        ..., 
        description="LLM's interpretation of the intent"
    )
    extracted_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters extracted from the intent"
    )
    
    # Network Actions
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="List of suggested network actions"
    )
    applied_actions: List[str] = Field(
        default_factory=list,
        description="List of actions that were successfully applied"
    )
    failed_actions: List[str] = Field(
        default_factory=list,
        description="List of actions that failed to apply"
    )
    
    # Flow Rules
    flow_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Flow rules that were installed or modified"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: Optional[int] = Field(
        default=None,
        description="Time taken to process the intent in milliseconds"
    )
    confidence_score: float = Field(
        default=0.0,
        description="Confidence score for the intent analysis (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Error Information
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if the intent processing failed"
    )
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed error information"
    )
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "intent_id": "intent_12345",
                "status": "completed",
                "intent_text": "Route all traffic from host h1 to h2 through the fastest path",
                "intent_type": "routing",
                "llm_interpretation": "User wants to establish optimal routing between h1 and h2",
                "extracted_parameters": {
                    "source": "h1",
                    "destination": "h2", 
                    "optimization": "fastest_path"
                },
                "suggested_actions": ["install_flow_rule", "update_routing_table"],
                "applied_actions": ["install_flow_rule"],
                "failed_actions": [],
                "flow_rules": [
                    {
                        "dpid": 1,
                        "table_id": 0,
                        "priority": 100,
                        "match": {"in_port": 1},
                        "actions": [{"type": "OUTPUT", "port": 2}]
                    }
                ],
                "processing_time_ms": 1500
            }
        } 