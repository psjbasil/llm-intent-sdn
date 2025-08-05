"""RYU controller service for SDN network management."""

import json
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger

from ..config import settings
from ..models.network import (
    NetworkTopology, NetworkDevice, NetworkPort, NetworkLink,
    FlowRule, FlowMatch, FlowAction, TrafficStats, DeviceType, PortState
)


class RyuService:
    """Service for interacting with RYU SDN controller."""
    
    def __init__(self) -> None:
        """Initialize RYU service."""
        self.base_url = f"http://{settings.ryu_host}:{settings.ryu_port}"
        self.api_base = settings.ryu_api_base
        
    async def get_network_topology(self) -> NetworkTopology:
        """
        Get current network topology from RYU controller.
        
        Returns:
            NetworkTopology: Current network topology
        """
        try:
            # Get switches
            switches = await self._get_switches()
            
            # Get ports for each switch
            devices = []
            for switch_data in switches:
                # Handle different possible data formats from RYU
                if isinstance(switch_data, dict):
                    dpid = switch_data.get("dpid", switch_data)
                else:
                    dpid = switch_data
                
                # Convert dpid to string if it's an integer
                if isinstance(dpid, int):
                    dpid_str = str(dpid)
                else:
                    dpid_str = str(dpid)
                
                ports = await self._get_switch_ports(dpid_str)
                
                device = NetworkDevice(
                    dpid=dpid_str,
                    name=f"s{dpid_str}",
                    device_type=DeviceType.SWITCH,
                    ports=ports
                )
                devices.append(device)
            
            # Get links
            links = await self._get_links()
            
            # Get flow rules
            flow_rules = await self._get_all_flow_rules()
            
            # Get traffic stats
            traffic_stats = await self._get_all_traffic_stats()
            
            topology = NetworkTopology(
                devices=devices,
                links=links,
                flow_rules=flow_rules,
                traffic_stats=traffic_stats
            )
            
            logger.info(f"Retrieved topology: {len(devices)} devices, {len(links)} links")
            return topology
            
        except Exception as e:
            logger.error(f"Error getting network topology: {e}")
            # Return a mock topology for development/demo purposes
            return self._get_mock_topology()
    
    def _get_mock_topology(self) -> NetworkTopology:
        """
        Return a mock topology for development/demo purposes.
        
        Returns:
            NetworkTopology: Mock network topology
        """
        from ..models.network import NetworkDevice, NetworkPort, NetworkLink, DeviceType, PortState
        
        # Create mock switches
        switch1_ports = [
            NetworkPort(
                port_no=1,
                name="port_1",
                hw_addr="00:00:00:00:00:01",
                state=PortState.UP,
                curr_speed=1000000000,  # 1Gbps
                max_speed=1000000000
            ),
            NetworkPort(
                port_no=2,
                name="port_2", 
                hw_addr="00:00:00:00:00:02",
                state=PortState.UP,
                curr_speed=1000000000,
                max_speed=1000000000
            )
        ]
        
        switch2_ports = [
            NetworkPort(
                port_no=1,
                name="port_1",
                hw_addr="00:00:00:00:00:03",
                state=PortState.UP,
                curr_speed=1000000000,
                max_speed=1000000000
            ),
            NetworkPort(
                port_no=2,
                name="port_2",
                hw_addr="00:00:00:00:00:04",
                state=PortState.UP,
                curr_speed=1000000000,
                max_speed=1000000000
            )
        ]
        
        devices = [
            NetworkDevice(
                dpid="0000000000000001",
                name="switch1",
                device_type=DeviceType.SWITCH,
                ports=switch1_ports
            ),
            NetworkDevice(
                dpid="0000000000000002",
                name="switch2",
                device_type=DeviceType.SWITCH,
                ports=switch2_ports
            )
        ]
        
        # Create mock links
        links = [
            NetworkLink(
                src_dpid="0000000000000001",
                src_port=2,
                dst_dpid="0000000000000002",
                dst_port=1
            )
        ]
        
        logger.info("Returning mock topology (3 devices, 1 link)")
        return NetworkTopology(
            devices=devices,
            links=links,
            flow_rules=[],
            traffic_stats={}
        )
    
    async def _get_switches(self) -> List[Dict[str, Any]]:
        """Get list of switches from RYU controller."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}{self.api_base}/switches")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting switches: {e}")
            return []
    
    async def _get_switch_ports(self, dpid: str) -> List[NetworkPort]:
        """
        Get ports for a specific switch.
        
        Args:
            dpid: Datapath ID of the switch
            
        Returns:
            List of NetworkPort objects
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}{self.api_base}/port/{dpid}"
                )
                response.raise_for_status()
                ports_data = response.json()
                
                ports = []
                # Handle different response formats from RYU
                if isinstance(ports_data, dict):
                    # If ports_data is a dict with dpid keys
                    for dpid_key, port_list in ports_data.items():
                        if isinstance(port_list, list):
                            for port_data in port_list:
                                if isinstance(port_data, dict):
                                    port = NetworkPort(
                                        port_no=port_data.get("port_no", 0),
                                        name=port_data.get("name", f"port_{port_data.get('port_no', 0)}"),
                                        hw_addr=port_data.get("hw_addr", "00:00:00:00:00:00"),
                                        state=PortState.UP if port_data.get("state", 0) == 0 else PortState.DOWN,
                                        curr_speed=port_data.get("curr_speed", 0),
                                        max_speed=port_data.get("max_speed", 0)
                                    )
                                    ports.append(port)
                elif isinstance(ports_data, list):
                    # If ports_data is directly a list of ports
                    for port_data in ports_data:
                        if isinstance(port_data, dict):
                            port = NetworkPort(
                                port_no=port_data.get("port_no", 0),
                                name=port_data.get("name", f"port_{port_data.get('port_no', 0)}"),
                                hw_addr=port_data.get("hw_addr", "00:00:00:00:00:00"),
                                state=PortState.UP if port_data.get("state", 0) == 0 else PortState.DOWN,
                                curr_speed=port_data.get("curr_speed", 0),
                                max_speed=port_data.get("max_speed", 0)
                            )
                            ports.append(port)
                
                return ports
                
        except Exception as e:
            logger.error(f"Error getting ports for switch {dpid}: {e}")
            logger.debug(f"Raw response data type: {type(ports_data) if 'ports_data' in locals() else 'No data'}")
            if 'ports_data' in locals():
                logger.debug(f"Raw response data: {ports_data}")
            return []
    
    async def _get_links(self) -> List[NetworkLink]:
        """Get network links from RYU controller."""
        try:
            # Note: RYU topology discovery might not be enabled by default
            # This is a placeholder implementation
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(f"{self.base_url}/v1.0/topology/links")
                    response.raise_for_status()
                    links_data = response.json()
                    
                    links = []
                    for link_data in links_data:
                        src = link_data.get("src", {})
                        dst = link_data.get("dst", {})
                        
                        link = NetworkLink(
                            src_dpid=src.get("dpid", 0),
                            src_port_no=src.get("port_no", 0),
                            dst_dpid=dst.get("dpid", 0),
                            dst_port_no=dst.get("port_no", 0)
                        )
                        links.append(link)
                    
                    return links
                    
                except httpx.HTTPStatusError:
                    # Topology API might not be available
                    logger.warning("Topology API not available, returning empty links")
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting links: {e}")
            return []
    
    async def _get_all_flow_rules(self) -> List[FlowRule]:
        """Get all flow rules from all switches."""
        try:
            switches = await self._get_switches()
            all_flow_rules = []
            
            for switch_data in switches:
                # Handle different possible data formats from RYU
                if isinstance(switch_data, dict):
                    dpid = switch_data.get("dpid", switch_data)
                else:
                    dpid = switch_data
                
                # Convert dpid to string if it's an integer
                if isinstance(dpid, int):
                    dpid_str = str(dpid)
                else:
                    dpid_str = str(dpid)
                
                flow_rules = await self._get_flow_rules(dpid_str)
                all_flow_rules.extend(flow_rules)
            
            return all_flow_rules
            
        except Exception as e:
            logger.error(f"Error getting all flow rules: {e}")
            return []
    
    async def _get_flow_rules(self, dpid: str) -> List[FlowRule]:
        """
        Get flow rules for a specific switch.
        
        Args:
            dpid: Datapath ID of the switch
            
        Returns:
            List of FlowRule objects
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}{self.api_base}/flow/{dpid}"
                )
                response.raise_for_status()
                flows_data = response.json()
                
                flow_rules = []
                for dpid_str, flow_list in flows_data.items():
                    for flow_data in flow_list:
                        # Parse match criteria
                        match_data = flow_data.get("match", {})
                        match = FlowMatch(
                            in_port=match_data.get("in_port"),
                            eth_src=match_data.get("eth_src"),
                            eth_dst=match_data.get("eth_dst"),
                            eth_type=match_data.get("eth_type"),
                            ip_src=match_data.get("ipv4_src"),
                            ip_dst=match_data.get("ipv4_dst"),
                            ip_proto=match_data.get("ip_proto"),
                            tcp_src=match_data.get("tcp_src"),
                            tcp_dst=match_data.get("tcp_dst"),
                            udp_src=match_data.get("udp_src"),
                            udp_dst=match_data.get("udp_dst"),
                            vlan_vid=match_data.get("vlan_vid")
                        )
                        
                        # Parse actions
                        actions_data = flow_data.get("actions", [])
                        actions = []
                        for action_data in actions_data:
                            action = FlowAction(
                                type=action_data.get("type", "OUTPUT"),
                                port=action_data.get("port"),
                                value=action_data.get("value")
                            )
                            actions.append(action)
                        
                        flow_rule = FlowRule(
                            dpid=int(dpid_str),
                            table_id=flow_data.get("table_id", 0),
                            priority=flow_data.get("priority", 0),
                            match=match,
                            actions=actions,
                            idle_timeout=flow_data.get("idle_timeout", 0),
                            hard_timeout=flow_data.get("hard_timeout", 0),
                            cookie=flow_data.get("cookie", 0)
                        )
                        flow_rules.append(flow_rule)
                
                return flow_rules
                
        except Exception as e:
            logger.error(f"Error getting flow rules for switch {dpid}: {e}")
            return []
    
    async def _get_all_traffic_stats(self) -> List[TrafficStats]:
        """Get traffic statistics from all switches."""
        try:
            switches = await self._get_switches()
            all_stats = []
            
            for switch_data in switches:
                dpid = switch_data["dpid"]
                stats = await self._get_traffic_stats(dpid)
                all_stats.extend(stats)
            
            return all_stats
            
        except Exception as e:
            logger.error(f"Error getting all traffic stats: {e}")
            return []
    
    async def _get_traffic_stats(self, dpid: int) -> List[TrafficStats]:
        """
        Get traffic statistics for a specific switch.
        
        Args:
            dpid: Datapath ID of the switch
            
        Returns:
            List of TrafficStats objects
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}{self.api_base}/port/{dpid}"
                )
                response.raise_for_status()
                stats_data = response.json()
                
                traffic_stats = []
                for dpid_str, port_list in stats_data.items():
                    for port_data in port_list:
                        stats = TrafficStats(
                            dpid=int(dpid_str),
                            port_no=port_data.get("port_no"),
                            rx_packets=port_data.get("rx_packets", 0),
                            tx_packets=port_data.get("tx_packets", 0),
                            rx_bytes=port_data.get("rx_bytes", 0),
                            tx_bytes=port_data.get("tx_bytes", 0),
                            rx_dropped=port_data.get("rx_dropped", 0),
                            tx_dropped=port_data.get("tx_dropped", 0),
                            rx_errors=port_data.get("rx_errors", 0),
                            tx_errors=port_data.get("tx_errors", 0),
                            duration_sec=port_data.get("duration_sec", 0),
                            duration_nsec=port_data.get("duration_nsec", 0)
                        )
                        traffic_stats.append(stats)
                
                return traffic_stats
                
        except Exception as e:
            logger.error(f"Error getting traffic stats for switch {dpid}: {e}")
            return []
    
    async def install_flow_rule(self, flow_rule: FlowRule) -> bool:
        """
        Install a flow rule on a switch.
        
        Args:
            flow_rule: FlowRule object to install
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert FlowRule to RYU API format
            payload = {
                "dpid": flow_rule.dpid,
                "table_id": flow_rule.table_id,
                "priority": flow_rule.priority,
                "idle_timeout": flow_rule.idle_timeout,
                "hard_timeout": flow_rule.hard_timeout,
                "cookie": flow_rule.cookie,
                "match": {},
                "actions": []
            }
            
            # Build match criteria
            if flow_rule.match.in_port is not None:
                payload["match"]["in_port"] = flow_rule.match.in_port
            if flow_rule.match.eth_src:
                payload["match"]["eth_src"] = flow_rule.match.eth_src
            if flow_rule.match.eth_dst:
                payload["match"]["eth_dst"] = flow_rule.match.eth_dst
            if flow_rule.match.eth_type is not None:
                payload["match"]["eth_type"] = flow_rule.match.eth_type
            if flow_rule.match.ip_src:
                payload["match"]["ipv4_src"] = flow_rule.match.ip_src
            if flow_rule.match.ip_dst:
                payload["match"]["ipv4_dst"] = flow_rule.match.ip_dst
            if flow_rule.match.ip_proto is not None:
                payload["match"]["ip_proto"] = flow_rule.match.ip_proto
            if flow_rule.match.tcp_src is not None:
                payload["match"]["tcp_src"] = flow_rule.match.tcp_src
            if flow_rule.match.tcp_dst is not None:
                payload["match"]["tcp_dst"] = flow_rule.match.tcp_dst
            if flow_rule.match.udp_src is not None:
                payload["match"]["udp_src"] = flow_rule.match.udp_src
            if flow_rule.match.udp_dst is not None:
                payload["match"]["udp_dst"] = flow_rule.match.udp_dst
            if flow_rule.match.vlan_vid is not None:
                payload["match"]["vlan_vid"] = flow_rule.match.vlan_vid
            
            # Build actions
            for action in flow_rule.actions:
                action_dict = {"type": action.type}
                if action.port is not None:
                    action_dict["port"] = action.port
                if action.value is not None:
                    action_dict["value"] = action.value
                payload["actions"].append(action_dict)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}{self.api_base}/flowentry/add",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Flow rule installed successfully on switch {flow_rule.dpid}")
                return True
                
        except Exception as e:
            logger.error(f"Error installing flow rule: {e}")
            return False
    
    async def delete_flow_rule(self, dpid: int, flow_rule: Optional[FlowRule] = None) -> bool:
        """
        Delete flow rule(s) from a switch.
        
        Args:
            dpid: Datapath ID of the switch
            flow_rule: Specific flow rule to delete (None to delete all)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if flow_rule:
                # Delete specific flow rule
                payload = {
                    "dpid": dpid,
                    "table_id": flow_rule.table_id,
                    "priority": flow_rule.priority,
                    "match": {}
                }
                
                # Add match criteria for specific rule deletion
                if flow_rule.match.in_port is not None:
                    payload["match"]["in_port"] = flow_rule.match.in_port
                # Add other match criteria as needed...
                
            else:
                # Delete all flow rules
                payload = {"dpid": dpid}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}{self.api_base}/flowentry/delete",
                    json=payload
                )
                response.raise_for_status()
                
                logger.info(f"Flow rule(s) deleted successfully from switch {dpid}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting flow rule: {e}")
            return False
    
    async def modify_flow_rule(self, flow_rule: FlowRule) -> bool:
        """
        Modify an existing flow rule.
        
        Args:
            flow_rule: FlowRule object with modifications
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # RYU typically handles flow modification via the same add API
            # with the OFPFC_MODIFY command
            return await self.install_flow_rule(flow_rule)
            
        except Exception as e:
            logger.error(f"Error modifying flow rule: {e}")
            return False 