"""Main FastAPI application for LLM Intent-based SDN system."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

from ..config import settings
from .routes import intent, network, monitoring, health, verification


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="LLM Intent-based SDN API",
        description="API for intent-based SDN network management using Large Language Models",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Mount static files for Web UI
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        # Create static directory if it doesn't exist
        os.makedirs(static_dir, exist_ok=True)
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Include routers
    app.include_router(intent.router, prefix="/api/v1/intent", tags=["Intent"])
    app.include_router(network.router, prefix="/api/v1/network", tags=["Network"])
    app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])
    app.include_router(verification.router, prefix="/api/v1", tags=["Verification"])
    app.include_router(health.router, prefix="/health", tags=["Health"])
    
    # Configure logging
    _configure_logging()
    
    @app.on_event("startup")
    async def startup_event() -> None:
        """Handle application startup."""
        logger.info("Starting LLM Intent-based SDN API")
        logger.info(f"API Version: 0.1.0")
        logger.info(f"Environment: {'DEBUG' if settings.api_debug else 'PRODUCTION'}")
        logger.info(f"RYU Controller: {settings.ryu_host}:{settings.ryu_port}")
        logger.info(f"LLM Model: {settings.llm_model}")
        
        # Create logs directory if it doesn't exist
        if settings.log_file:
            log_dir = os.path.dirname(settings.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
        # Create static directory for Web UI if it doesn't exist
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)
    
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Handle application shutdown."""
        logger.info("Shutting down LLM Intent-based SDN API")
    
    @app.get("/")
    async def root():
        """Root endpoint - redirect to Web UI."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/static/index.html")
    
    @app.get("/api")
    async def api_info():
        """API information endpoint."""
        return {
            "message": "LLM Intent-based SDN API",
            "version": "0.1.0",
            "docs": "/docs",
            "web_ui": "/static/index.html",
            "status": "running"
        }
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return {"error": "Internal server error", "detail": str(exc)}
    
    return app


def _configure_logging() -> None:
    """Configure application logging."""
    # Custom logging configuration here if needed
    pass


# Create app instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower()
    ) 