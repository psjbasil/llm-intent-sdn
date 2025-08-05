"""Intent processing API routes."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from loguru import logger

from ...models.intent import IntentRequest, IntentResponse
from ...services.intent_processor import IntentProcessor


router = APIRouter()
intent_processor = IntentProcessor()


@router.post("/process", response_model=IntentResponse)
async def process_intent(
    request: IntentRequest,
    background_tasks: BackgroundTasks
) -> IntentResponse:
    """
    Process a network intent.
    
    Args:
        request: Intent request
        background_tasks: FastAPI background tasks
        
    Returns:
        IntentResponse: Processed intent response
    """
    try:
        logger.info(f"Received intent request: {request.intent_text}")
        
        # Process intent
        response = await intent_processor.process_intent(request)
        
        logger.info(f"Intent processed: {response.intent_id} - {response.status}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=IntentResponse)
async def analyze_intent(request: IntentRequest) -> IntentResponse:
    """
    Analyze a network intent without executing it.
    
    Args:
        request: Intent request
        
    Returns:
        IntentResponse: Analysis results
    """
    try:
        logger.info(f"Received intent analysis request: {request.intent_text}")
        
        # Analyze intent without execution
        response = await intent_processor.analyze_intent(request)
        
        logger.info(f"Intent analyzed: {response.intent_id} - {response.status}")
        return response
        
    except Exception as e:
        logger.error(f"Error analyzing intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{intent_id}", response_model=IntentResponse)
async def get_intent_status(intent_id: str) -> IntentResponse:
    """
    Get status of a specific intent.
    
    Args:
        intent_id: Intent identifier
        
    Returns:
        IntentResponse: Intent status
    """
    try:
        response = await intent_processor.get_intent_status(intent_id)
        
        if not response:
            raise HTTPException(status_code=404, detail="Intent not found")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting intent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[IntentResponse])
async def list_intents(
    status: Optional[str] = None,
    limit: int = 100
) -> List[IntentResponse]:
    """
    List active intents.
    
    Args:
        status: Filter by status (optional)
        limit: Maximum number of intents to return
        
    Returns:
        List of intent responses
    """
    try:
        intents = await intent_processor.list_active_intents()
        
        # Filter by status if specified
        if status:
            intents = [intent for intent in intents if intent.status == status]
        
        # Apply limit
        intents = intents[:limit]
        
        return intents
        
    except Exception as e:
        logger.error(f"Error listing intents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{intent_id}/cancel")
async def cancel_intent(intent_id: str) -> dict:
    """
    Cancel an active intent.
    
    Args:
        intent_id: Intent identifier
        
    Returns:
        Cancellation result
    """
    try:
        success = await intent_processor.cancel_intent(intent_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel intent")
        
        return {"message": f"Intent {intent_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling intent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_intent_only(request: IntentRequest) -> dict:
    """
    Analyze intent without executing actions.
    
    Args:
        request: Intent request
        
    Returns:
        Intent analysis result
    """
    try:
        # Get network topology
        topology = await intent_processor.ryu_service.get_network_topology()
        
        # Analyze intent with LLM
        analysis = await intent_processor.llm_service.analyze_intent(
            intent_text=request.intent_text,
            network_topology=topology,
            context=request.context
        )
        
        return {
            "intent_text": request.intent_text,
            "analysis": {
                "intent_type": analysis.intent_type,
                "confidence": analysis.confidence,
                "extracted_entities": analysis.extracted_entities,
                "suggested_actions": analysis.suggested_actions,
                "reasoning": analysis.reasoning
            },
            "network_context": {
                "devices": len(topology.devices),
                "links": len(topology.links),
                "flow_rules": len(topology.flow_rules)
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing intent: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 