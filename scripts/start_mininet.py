#!/usr/bin/env python3
"""
Script to create and start Mininet topology for testing SDN applications.
This script creates a sample network topology that can be used with the LLM Intent-based SDN system.
"""

import sys
import time
import signal
from pathlib import Path

try:
    from mininet.net import Mininet
    from mininet.node import Controller, RemoteController, OVSKernelSwitch
    from mininet.cli import CLI
    from mininet.log import setLogLevel, info
    from mininet.link import TCLink
    from mininet.topo import Topo
except ImportError:
    print("Error: Mininet not found!")
    print("Please install Mininet: sudo apt-get install mininet")
    sys.exit(1)


class CustomTopology(Topo):
    """Custom network topology for testing."""
    
    def build(self):
        """Build the network topology."""
        info("*** Creating custom topology\n")
        
        # Add switches
        s1 = self.addSwitch('s1', cls=OVSKernelSwitch, protocols='OpenFlow13')
        s2 = self.addSwitch('s2', cls=OVSKernelSwitch, protocols='OpenFlow13')
        s3 = self.addSwitch('s3', cls=OVSKernelSwitch, protocols='OpenFlow13')
        
        # Add hosts
        h1 = self.addHost('h1', ip='10.0.0.1/24')
        h2 = self.addHost('h2', ip='10.0.0.2/24')
        h3 = self.addHost('h3', ip='10.0.0.3/24')
        h4 = self.addHost('h4', ip='10.0.0.4/24')
        
        # Add links with bandwidth constraints
        # Host to switch links
        self.addLink(h1, s1, cls=TCLink, bw=10)
        self.addLink(h2, s1, cls=TCLink, bw=10)
        self.addLink(h3, s2, cls=TCLink, bw=10)
        self.addLink(h4, s3, cls=TCLink, bw=10)
        
        # Switch to switch links
        self.addLink(s1, s2, cls=TCLink, bw=5)
        self.addLink(s2, s3, cls=TCLink, bw=5)
        self.addLink(s1, s3, cls=TCLink, bw=8)


class MininetManager:
    """Mininet network manager."""
    
    def __init__(self, controller_ip='127.0.0.1', controller_port=6653):
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.net = None
        
    def create_network(self):
        """Create and start the network."""
        info("*** Creating network\n")
        
        # Create topology
        topo = CustomTopology()
        
        # Create network with remote controller
        self.net = Mininet(
            topo=topo,
            controller=RemoteController,
            switch=OVSKernelSwitch,
            link=TCLink,
            autoSetMacs=True,
            autoStaticArp=True
        )
        
        # Add remote controller
        controller = self.net.addController(
            'c0',
            controller=RemoteController,
            ip=self.controller_ip,
            port=self.controller_port
        )
        
        return self.net
    
    def start_network(self):
        """Start the network."""
        if not self.net:
            self.create_network()
            
        info("*** Starting network\n")
        self.net.start()
        
        # Wait for switches to connect
        info("*** Waiting for switches to connect to controller\n")
        time.sleep(3)
        
        # Test connectivity
        info("*** Testing connectivity\n")
        self.net.pingAll()
        
        # Display network information
        self.show_network_info()
        
    def show_network_info(self):
        """Display network information."""
        info("\n*** Network Information ***\n")
        info("Switches:\n")
        for switch in self.net.switches:
            info(f"  {switch.name}: {switch.IP()}\n")
            
        info("\nHosts:\n")
        for host in self.net.hosts:
            info(f"  {host.name}: {host.IP()}\n")
            
        info("\nController:\n")
        for controller in self.net.controllers:
            info(f"  {controller.name}: {controller.IP()}:{controller.port}\n")
    
    def stop_network(self):
        """Stop the network."""
        if self.net:
            info("*** Stopping network\n")
            self.net.stop()
    
    def start_cli(self):
        """Start Mininet CLI."""
        if self.net:
            info("*** Starting CLI\n")
            info("*** Available commands:\n")
            info("    pingall - Test connectivity between all hosts\n")
            info("    iperf h1 h3 - Test bandwidth between h1 and h3\n")
            info("    h1 ping h3 - Ping from h1 to h3\n")
            info("    exit - Exit CLI\n")
            CLI(self.net)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)


def main():
    """Main function."""
    print(" Starting Mininet topology for LLM Intent-based SDN")
    print("=" * 60)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set log level
    setLogLevel('info')
    
    # Create network manager
    manager = MininetManager()
    
    try:
        # Start network
        manager.start_network()
        
        print("\n Network started successfully!")
        print("Use 'pingall' to test connectivity")
        print("Use 'iperf h1 h3' to test bandwidth")
        print("Press Ctrl+C to stop the network")
        
        # Start CLI
        manager.start_cli()
        
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        manager.stop_network()
        print("Network stopped successfully!")


if __name__ == "__main__":
    # Check if running as root
    import os
    if os.geteuid() != 0:
        print("Error: This script must be run as root!")
        print("Please run: sudo python scripts/start_mininet.py")
        sys.exit(1)
    
    main() 