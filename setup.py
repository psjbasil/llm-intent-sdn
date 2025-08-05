#!/usr/bin/env python3
"""
Setup script for the LLM Intent-based SDN system.
This script helps set up the development environment.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def run_command(command: str, description: str = None):
    """Run a shell command and handle errors."""
    if description:
        print(f"📦 {description}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is >= 3.8."""
    version = sys.version_info
    if version < (3, 8):
        print(f"❌ Python 3.10+ is required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing project dependencies...")
    
    # Install main dependencies
    if not run_command("pip install -e .", "Installing main package"):
        return False
    
    # Install development dependencies
    if not run_command("pip install -e .[dev]", "Installing development dependencies"):
        print("⚠️  Warning: Failed to install dev dependencies")
    
    return True


def create_env_file():
    """Create .env file from env.example if it doesn't exist."""
    env_file = Path(".env")
    example_file = Path("env.example")
    
    if not env_file.exists() and example_file.exists():
        print("📝 Creating .env file from env.example")
        shutil.copy(example_file, env_file)
        print("⚠️  Please edit .env file and add your API keys!")
    else:
        print("✅ .env file already exists")


def create_directories():
    """Create necessary directories."""
    directories = ["logs", "data", "tests"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")


def setup_pre_commit():
    """Setup pre-commit hooks."""
    if shutil.which("pre-commit"):
        run_command("pre-commit install", "Setting up pre-commit hooks")
    else:
        print("⚠️  pre-commit not found, skipping hook setup")


def main():
    """Main setup function."""
    print("🚀 Setting up LLM Intent-based SDN system")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Create necessary directories
    create_directories()
    
    # Setup pre-commit
    setup_pre_commit()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Start RYU controller: ryu-manager ryu.app.rest_topology ryu.app.ws_topology ryu.app.ofctl_rest")
    print("3. Start the application: python run.py")
    print("4. Access API documentation: http://localhost:8000/docs")


if __name__ == "__main__":
    main() 