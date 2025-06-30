#!/usr/bin/env python3
"""
Main entry point for the LLM Intent-based SDN system.
This script starts the FastAPI server with proper configuration.
"""

import asyncio
import os
import sys
from pathlib import Path

import uvicorn
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_intent_sdn.config import settings
from llm_intent_sdn.utils.logging import setup_logging


def main():
    """Main entry point for the application."""
    # Setup logging
    setup_logging()
    
    # Log startup information
    logger.info("Starting LLM Intent-based SDN system")
    logger.info(f"API Host: {settings.api_host}")
    logger.info(f"API Port: {settings.api_port}")
    logger.info(f"RYU Controller: {settings.ryu_host}:{settings.ryu_port}")
    logger.info(f"LLM Model: {settings.llm_model}")
    
    # Ensure logs directory exists
    if settings.log_file:
        log_dir = Path(settings.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Start the FastAPI server
    try:
        uvicorn.run(
            "llm_intent_sdn.api.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_debug,
            log_level=settings.log_level.lower(),
            access_log=True,
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal, stopping server...")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 