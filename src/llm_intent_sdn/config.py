"""Configuration management for the LLM Intent-based SDN system."""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import validator
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    
    # RYU Controller Configuration
    ryu_host: str = "127.0.0.1"
    ryu_port: int = 8080
    ryu_api_base: str = "/stats"
    
    # LLM Configuration
    openai_api_key: str = ""
    openai_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "deepseek/deepseek-chat"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    
    # Ollama Configuration (for local models)
    ollama_host: str = "127.0.0.1"
    ollama_port: int = 11434
    use_ollama: bool = False
    ollama_timeout: int = 120
    
    # Gemini Configuration (for Gemini API)
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-1.5-flash"
    use_gemini: bool = True
    
    # Network Configuration
    mininet_host: str = "127.0.0.1"
    mininet_port: int = 6653
    
    # Security
    allowed_hosts: List[str] = ["*"]
    cors_origins: List[str] = ["*"]
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/llm_intent_sdn.log"
    
    # Intent Processing
    intent_timeout: int = 30
    max_retry_attempts: int = 3
    
    @validator("openai_api_key")
    def validate_api_key(cls, v: str) -> str:
        """Validate that OpenAI API key is provided."""
        if not v:
            # For development, allow empty API key with warning
            import warnings
            warnings.warn("OpenAI API key not provided. Some LLM features may not work.")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @property
    def llm_base_url(self) -> str:
        """Get the appropriate LLM base URL based on configuration."""
        if self.use_ollama:
            return f"http://{self.ollama_host}:{self.ollama_port}"
        elif self.use_gemini:
            return self.gemini_base_url
        return self.openai_base_url
    
    @property
    def llm_model_name(self) -> str:
        """Get the appropriate model name based on configuration."""
        if self.use_ollama:
            return "llama3.2:latest"  # Default Ollama model
        elif self.use_gemini:
            return self.gemini_model
        return self.llm_model
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_prefix = "LLM_SDN_"
        case_sensitive = False


# Global settings instance
settings = Settings() 