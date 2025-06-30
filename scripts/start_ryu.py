#!/usr/bin/env python3
"""
Script to start RYU controller with necessary applications.
This script automates the process of starting RYU with the required apps for SDN management.
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path


class RyuController:
    """RYU controller manager."""
    
    def __init__(self):
        self.process = None
        self.ryu_apps = [
            "ryu.app.rest_topology",
            "ryu.app.ws_topology", 
            "ryu.app.ofctl_rest",
            "ryu.app.rest_conf_switch",
            "ryu.app.simple_switch_13"
        ]
    
    def start(self):
        """Start RYU controller with required applications."""
        print(" Starting RYU controller...")
        print(f"Loading applications: {', '.join(self.ryu_apps)}")
        
        try:
            # Build the command
            cmd = ["ryu-manager"] + self.ryu_apps
            print(f"Command: {' '.join(cmd)}")
            
            # Start RYU controller
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print("RYU controller started successfully!")
            print("REST API available at: http://localhost:8080")
            print("WebSocket API available at: ws://localhost:8080")
            print("\nRYU Controller Output:")
            print("-" * 50)
            
            # Stream output
            for line in self.process.stdout:
                print(line.rstrip())
                
        except FileNotFoundError:
            print("Error: RYU controller not found!")
            print("Please install RYU: pip install ryu")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nReceived shutdown signal...")
            self.stop()
        except Exception as e:
            print(f"Error starting RYU controller: {e}")
            sys.exit(1)
    
    def stop(self):
        """Stop RYU controller."""
        if self.process:
            print("Stopping RYU controller...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                print("RYU controller stopped successfully!")
            except subprocess.TimeoutExpired:
                print("Force killing RYU controller...")
                self.process.kill()
                self.process.wait()
    
    def is_running(self):
        """Check if RYU controller is running."""
        return self.process and self.process.poll() is None


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main function."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start RYU controller
    controller = RyuController()
    
    try:
        controller.start()
    finally:
        controller.stop()


if __name__ == "__main__":
    main() 