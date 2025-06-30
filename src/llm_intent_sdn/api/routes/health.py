"""Health check API routes."""

import time
from fastapi import APIRouter, HTTPException
from loguru import logger
import httpx

from ...config import settings


router = APIRouter()


@router.get("/")
async def health_check() -> dict:
    """
    Basic health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "service": "llm-intent-sdn"
    }


@router.get("/detailed")
async def detailed_health_check() -> dict:
    """
    Detailed health check including dependencies.
    
    Returns:
        Detailed health status
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "0.1.0",
        "service": "llm-intent-sdn",
        "dependencies": {}
    }
    
    overall_healthy = True
    
    # Check RYU controller
    try:
        ryu_healthy = await _check_ryu_health()
        health_status["dependencies"]["ryu_controller"] = {
            "status": "healthy" if ryu_healthy else "unhealthy",
            "endpoint": f"{settings.ryu_host}:{settings.ryu_port}",
            "last_checked": time.time()
        }
        if not ryu_healthy:
            overall_healthy = False
    except Exception as e:
        health_status["dependencies"]["ryu_controller"] = {
            "status": "error",
            "error": str(e),
            "endpoint": f"{settings.ryu_host}:{settings.ryu_port}",
            "last_checked": time.time()
        }
        overall_healthy = False
    
    # Check LLM API
    try:
        llm_healthy = await _check_llm_health()
        health_status["dependencies"]["llm_api"] = {
            "status": "healthy" if llm_healthy else "unhealthy",
            "endpoint": settings.openai_base_url,
            "model": settings.llm_model,
            "last_checked": time.time()
        }
        if not llm_healthy:
            overall_healthy = False
    except Exception as e:
        health_status["dependencies"]["llm_api"] = {
            "status": "error",
            "error": str(e),
            "endpoint": settings.openai_base_url,
            "model": settings.llm_model,
            "last_checked": time.time()
        }
        overall_healthy = False
    
    # Update overall status
    if not overall_healthy:
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check for deployment orchestration.
    
    Returns:
        Readiness status
    """
    try:
        # Check if critical dependencies are available
        ryu_healthy = await _check_ryu_health()
        
        if not ryu_healthy:
            raise HTTPException(
                status_code=503, 
                detail="Service not ready - RYU controller unavailable"
            )
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "message": "Service is ready to accept requests"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready - {str(e)}"
        )


@router.get("/live")
async def liveness_check() -> dict:
    """
    Liveness check for deployment orchestration.
    
    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": time.time(),
        "uptime": time.time(),  # Simplified uptime
        "message": "Service is alive"
    }


async def _check_ryu_health() -> bool:
    """
    Check if RYU controller is healthy.
    
    Returns:
        bool: True if healthy, False otherwise
    """
    try:
        base_url = f"http://{settings.ryu_host}:{settings.ryu_port}"
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}{settings.ryu_api_base}/switches")
            return response.status_code == 200
            
    except Exception as e:
        logger.warning(f"RYU health check failed: {e}")
        return False


async def _check_llm_health() -> bool:
    """
    Check if LLM API is healthy.
    
    Returns:
        bool: True if healthy, False otherwise
    """
    try:
        # Simple health check - just verify we can connect
        headers = {
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Use a minimal request to check connectivity
        payload = {
            "model": settings.llm_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{settings.openai_base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            return response.status_code in [200, 400]  # 400 might be ok for minimal request
            
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
        return False


@router.get("/metrics")
async def get_health_metrics() -> dict:
    """
    Get health-related metrics.
    
    Returns:
        Health metrics
    """
    try:
        # Basic metrics
        current_time = time.time()
        
        # Check dependencies
        ryu_start_time = time.time()
        ryu_healthy = await _check_ryu_health()
        ryu_response_time = int((time.time() - ryu_start_time) * 1000)
        
        llm_start_time = time.time()
        llm_healthy = await _check_llm_health()
        llm_response_time = int((time.time() - llm_start_time) * 1000)
        
        return {
            "timestamp": current_time,
            "dependency_health": {
                "ryu_controller": {
                    "healthy": ryu_healthy,
                    "response_time_ms": ryu_response_time
                },
                "llm_api": {
                    "healthy": llm_healthy,
                    "response_time_ms": llm_response_time
                }
            },
            "service_metrics": {
                "memory_usage": "not_implemented",  # Could add psutil here
                "cpu_usage": "not_implemented",
                "active_connections": "not_implemented"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting health metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 