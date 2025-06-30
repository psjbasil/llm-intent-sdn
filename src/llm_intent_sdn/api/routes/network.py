"""Network management API routes."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from loguru import logger

from ...models.network import NetworkTopology, FlowRule
from ...services.ryu_service import RyuService


router = APIRouter()
ryu_service = RyuService()


@router.get("/topology", response_model=NetworkTopology)
async def get_network_topology() -> NetworkTopology:
    """
    Get current network topology.
    
    Returns:
        NetworkTopology: Current network topology
    """
    try:
        topology = await ryu_service.get_network_topology()
        return topology
        
    except Exception as e:
        logger.error(f"Error getting network topology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/devices")
async def get_network_devices() -> dict:
    """
    Get network devices information.
    
    Returns:
        Network devices summary
    """
    try:
        topology = await ryu_service.get_network_topology()
        
        devices_summary = []
        for device in topology.devices:
            device_info = {
                "dpid": device.dpid,
                "name": device.name,
                "device_type": device.device_type,
                "ip_address": device.ip_address,
                "port_count": len(device.ports),
                "connected_to": device.connected_to
            }
            devices_summary.append(device_info)
        
        return {
            "total_devices": len(devices_summary),
            "devices": devices_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting network devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flows")
async def get_flow_rules(dpid: Optional[int] = None) -> dict:
    """
    Get flow rules from switches.
    
    Args:
        dpid: Optional switch DPID to filter by
        
    Returns:
        Flow rules information
    """
    try:
        topology = await ryu_service.get_network_topology()
        
        flow_rules = topology.flow_rules
        if dpid is not None:
            flow_rules = [rule for rule in flow_rules if rule.dpid == dpid]
        
        flow_summary = []
        for rule in flow_rules:
            rule_info = {
                "dpid": rule.dpid,
                "table_id": rule.table_id,
                "priority": rule.priority,
                "match": rule.match.dict(exclude_none=True),
                "actions": [action.dict(exclude_none=True) for action in rule.actions],
                "idle_timeout": rule.idle_timeout,
                "hard_timeout": rule.hard_timeout
            }
            flow_summary.append(rule_info)
        
        return {
            "total_rules": len(flow_summary),
            "rules": flow_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting flow rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flows/install")
async def install_flow_rule(flow_rule: FlowRule) -> dict:
    """
    Install a flow rule on a switch.
    
    Args:
        flow_rule: Flow rule to install
        
    Returns:
        Installation result
    """
    try:
        success = await ryu_service.install_flow_rule(flow_rule)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to install flow rule")
        
        return {
            "message": f"Flow rule installed successfully on switch {flow_rule.dpid}",
            "dpid": flow_rule.dpid,
            "priority": flow_rule.priority
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error installing flow rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/flows/{dpid}")
async def delete_flow_rules(dpid: int, all_rules: bool = False) -> dict:
    """
    Delete flow rules from a switch.
    
    Args:
        dpid: Switch DPID
        all_rules: Whether to delete all rules
        
    Returns:
        Deletion result
    """
    try:
        success = await ryu_service.delete_flow_rule(dpid, None if all_rules else None)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to delete flow rules")
        
        message = f"All flow rules deleted from switch {dpid}" if all_rules else f"Flow rules deleted from switch {dpid}"
        
        return {"message": message, "dpid": dpid}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting flow rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats/traffic")
async def get_traffic_stats(dpid: Optional[int] = None) -> dict:
    """
    Get traffic statistics.
    
    Args:
        dpid: Optional switch DPID to filter by
        
    Returns:
        Traffic statistics
    """
    try:
        topology = await ryu_service.get_network_topology()
        
        traffic_stats = topology.traffic_stats
        if dpid is not None:
            traffic_stats = [stats for stats in traffic_stats if stats.dpid == dpid]
        
        stats_summary = []
        total_rx_bytes = 0
        total_tx_bytes = 0
        
        for stats in traffic_stats:
            stat_info = {
                "dpid": stats.dpid,
                "port_no": stats.port_no,
                "rx_packets": stats.rx_packets,
                "tx_packets": stats.tx_packets,
                "rx_bytes": stats.rx_bytes,
                "tx_bytes": stats.tx_bytes,
                "rx_dropped": stats.rx_dropped,
                "tx_dropped": stats.tx_dropped,
                "rx_errors": stats.rx_errors,
                "tx_errors": stats.tx_errors,
                "duration_sec": stats.duration_sec,
                "timestamp": stats.timestamp
            }
            stats_summary.append(stat_info)
            total_rx_bytes += stats.rx_bytes
            total_tx_bytes += stats.tx_bytes
        
        return {
            "total_stats": len(stats_summary),
            "total_rx_bytes": total_rx_bytes,
            "total_tx_bytes": total_tx_bytes,
            "stats": stats_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting traffic stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/links")
async def get_network_links() -> dict:
    """
    Get network links information.
    
    Returns:
        Network links summary
    """
    try:
        topology = await ryu_service.get_network_topology()
        
        links_summary = []
        for link in topology.links:
            link_info = {
                "src_dpid": link.src_dpid,
                "src_port_no": link.src_port_no,
                "dst_dpid": link.dst_dpid,
                "dst_port_no": link.dst_port_no,
                "bandwidth": link.bandwidth,
                "latency": link.latency,
                "utilization": link.utilization
            }
            links_summary.append(link_info)
        
        return {
            "total_links": len(links_summary),
            "links": links_summary
        }
        
    except Exception as e:
        logger.error(f"Error getting network links: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 