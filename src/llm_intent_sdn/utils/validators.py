"""Validation utilities."""

import re
import ipaddress
from typing import Union


def validate_ip_address(ip: str) -> bool:
    """
    Validate IP address format.
    
    Args:
        ip: IP address string
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def validate_mac_address(mac: str) -> bool:
    """
    Validate MAC address format.
    
    Args:
        mac: MAC address string
        
    Returns:
        bool: True if valid, False otherwise
    """
    # MAC address pattern: XX:XX:XX:XX:XX:XX or XX-XX-XX-XX-XX-XX
    pattern = r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
    return bool(re.match(pattern, mac))


def validate_dpid(dpid: Union[str, int]) -> bool:
    """
    Validate datapath ID.
    
    Args:
        dpid: Datapath ID
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        dpid_int = int(dpid)
        return 0 <= dpid_int <= 0xFFFFFFFFFFFFFFFF  # 64-bit max
    except (ValueError, TypeError):
        return False


def validate_port_number(port: Union[str, int]) -> bool:
    """
    Validate port number.
    
    Args:
        port: Port number
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        port_int = int(port)
        return 1 <= port_int <= 65535
    except (ValueError, TypeError):
        return False 