"""Network-related data models."""

from typing import Any, Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class PortState(str, Enum):
    """Port state enumeration."""
    
    UP = "up"
    DOWN = "down"
    BLOCKED = "blocked"


class DeviceType(str, Enum):
    """Network device type enumeration."""
    
    SWITCH = "switch"
    HOST = "host"
    CONTROLLER = "controller"


class FlowAction(BaseModel):
    """Flow action model."""
    
    type: str = Field(..., description="Action type (e.g., OUTPUT, DROP, SET_VLAN)")
    port: Optional[int] = Field(None, description="Output port number")
    value: Optional[Any] = Field(None, description="Action value")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "type": "OUTPUT",
                "port": 2
            }
        }


class FlowMatch(BaseModel):
    """Flow match criteria model."""
    
    in_port: Optional[int] = Field(None, description="Input port")
    eth_src: Optional[str] = Field(None, description="Source MAC address")
    eth_dst: Optional[str] = Field(None, description="Destination MAC address")
    eth_type: Optional[int] = Field(None, description="Ethernet type")
    ip_src: Optional[str] = Field(None, description="Source IP address")
    ip_dst: Optional[str] = Field(None, description="Destination IP address")
    ip_proto: Optional[int] = Field(None, description="IP protocol")
    tcp_src: Optional[int] = Field(None, description="TCP source port")
    tcp_dst: Optional[int] = Field(None, description="TCP destination port")
    udp_src: Optional[int] = Field(None, description="UDP source port")
    udp_dst: Optional[int] = Field(None, description="UDP destination port")
    vlan_vid: Optional[int] = Field(None, description="VLAN ID")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "in_port": 1,
                "eth_type": 2048,
                "ip_dst": "10.0.0.2"
            }
        }


class FlowRule(BaseModel):
    """Flow rule model."""
    
    dpid: int = Field(..., description="Datapath ID of the switch")
    table_id: int = Field(default=0, description="Flow table ID")
    priority: int = Field(default=100, description="Flow priority")
    match: FlowMatch = Field(..., description="Match criteria")
    actions: List[FlowAction] = Field(..., description="Actions to perform")
    idle_timeout: int = Field(default=0, description="Idle timeout in seconds")
    hard_timeout: int = Field(default=0, description="Hard timeout in seconds")
    cookie: int = Field(default=0, description="Flow cookie")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "dpid": 1,
                "table_id": 0,
                "priority": 100,
                "match": {
                    "in_port": 1,
                    "eth_type": 2048,
                    "ip_dst": "10.0.0.2"
                },
                "actions": [
                    {
                        "type": "OUTPUT",
                        "port": 2
                    }
                ],
                "idle_timeout": 0,
                "hard_timeout": 0
            }
        }


class NetworkPort(BaseModel):
    """Network port model."""
    
    port_no: int = Field(..., description="Port number")
    name: str = Field(..., description="Port name")
    hw_addr: str = Field(..., description="Hardware address")
    state: PortState = Field(..., description="Port state")
    curr_speed: int = Field(default=0, description="Current speed in Kbps")
    max_speed: int = Field(default=0, description="Maximum speed in Kbps")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "port_no": 1,
                "name": "s1-eth1",
                "hw_addr": "00:00:00:00:00:01",
                "state": "up",
                "curr_speed": 10000000,
                "max_speed": 10000000
            }
        }


class NetworkDevice(BaseModel):
    """Network device model."""
    
    dpid: Optional[int] = Field(None, description="Datapath ID (for switches)")
    name: str = Field(..., description="Device name")
    device_type: DeviceType = Field(..., description="Device type")
    ip_address: Optional[str] = Field(None, description="IP address")
    mac_address: Optional[str] = Field(None, description="MAC address")
    ports: List[NetworkPort] = Field(default_factory=list, description="Device ports")
    connected_to: List[str] = Field(
        default_factory=list, 
        description="List of connected device names"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional device metadata"
    )
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "dpid": 1,
                "name": "s1",
                "device_type": "switch",
                "ip_address": "127.0.0.1",
                "ports": [
                    {
                        "port_no": 1,
                        "name": "s1-eth1", 
                        "hw_addr": "00:00:00:00:00:01",
                        "state": "up",
                        "curr_speed": 10000000,
                        "max_speed": 10000000
                    }
                ],
                "connected_to": ["h1", "s2"]
            }
        }


class TrafficStats(BaseModel):
    """Traffic statistics model."""
    
    dpid: int = Field(..., description="Datapath ID")
    port_no: Optional[int] = Field(None, description="Port number (None for aggregate)")
    
    # Packet statistics
    rx_packets: int = Field(default=0, description="Received packets")
    tx_packets: int = Field(default=0, description="Transmitted packets")
    rx_bytes: int = Field(default=0, description="Received bytes")
    tx_bytes: int = Field(default=0, description="Transmitted bytes")
    
    # Error statistics
    rx_dropped: int = Field(default=0, description="Dropped received packets")
    tx_dropped: int = Field(default=0, description="Dropped transmitted packets")
    rx_errors: int = Field(default=0, description="Receive errors")
    tx_errors: int = Field(default=0, description="Transmit errors")
    
    # Timing
    duration_sec: int = Field(default=0, description="Duration in seconds")
    duration_nsec: int = Field(default=0, description="Duration nanoseconds")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "dpid": 1,
                "port_no": 1,
                "rx_packets": 1000,
                "tx_packets": 950,
                "rx_bytes": 64000,
                "tx_bytes": 60800,
                "rx_dropped": 0,
                "tx_dropped": 0,
                "rx_errors": 0,
                "tx_errors": 0,
                "duration_sec": 3600
            }
        }


class NetworkLink(BaseModel):
    """Network link model."""
    
    src_dpid: int = Field(..., description="Source switch DPID")
    src_port_no: int = Field(..., description="Source port number")
    dst_dpid: int = Field(..., description="Destination switch DPID")
    dst_port_no: int = Field(..., description="Destination port number")
    bandwidth: Optional[int] = Field(None, description="Link bandwidth in Mbps")
    latency: Optional[float] = Field(None, description="Link latency in ms")
    utilization: Optional[float] = Field(None, description="Link utilization percentage")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "src_dpid": 1,
                "src_port_no": 2,
                "dst_dpid": 2,
                "dst_port_no": 1,
                "bandwidth": 1000,
                "latency": 0.5,
                "utilization": 25.5
            }
        }


class NetworkTopology(BaseModel):
    """Network topology model."""
    
    devices: List[NetworkDevice] = Field(
        default_factory=list,
        description="List of network devices"
    )
    links: List[NetworkLink] = Field(
        default_factory=list,
        description="List of network links"
    )
    flow_rules: List[FlowRule] = Field(
        default_factory=list,
        description="List of active flow rules"
    )
    traffic_stats: List[TrafficStats] = Field(
        default_factory=list,
        description="List of traffic statistics"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Topology snapshot timestamp"
    )
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "devices": [
                    {
                        "dpid": 1,
                        "name": "s1",
                        "device_type": "switch",
                        "connected_to": ["h1", "s2"]
                    }
                ],
                "links": [
                    {
                        "src_dpid": 1,
                        "src_port_no": 2,
                        "dst_dpid": 2,
                        "dst_port_no": 1,
                        "bandwidth": 1000
                    }
                ],
                "flow_rules": [],
                "traffic_stats": []
            }
        } 