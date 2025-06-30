"""Main FastAPI application for LLM Intent-based SDN system."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from loguru import logger
import uvicorn

from ..config import settings
from .routes import intent, network, monitoring, health


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
    
    # Include routers
    app.include_router(intent.router, prefix="/api/v1/intent", tags=["Intent"])
    app.include_router(network.router, prefix="/api/v1/network", tags=["Network"])
    app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])
    app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
    
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
    
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Handle application shutdown."""
        logger.info("Shutting down LLM Intent-based SDN API")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "LLM Intent-based SDN API",
            "version": "0.1.0",
            "docs": "/docs",
            "status": "running"
        }
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        return {"error": "Internal server error", "detail": str(exc)}
    
    return app


def _configure_logging() -> None:
    """Configure logging with loguru."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add file handler if specified
    if settings.log_file:
        logger.add(
            sink=settings.log_file,
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="1 day",
            retention="30 days",
            compression="gz"
        )


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