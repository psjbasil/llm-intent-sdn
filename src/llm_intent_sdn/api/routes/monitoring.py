"""Network monitoring API routes."""

from fastapi import APIRouter, HTTPException
from loguru import logger

from ...services.network_monitor import NetworkMonitor
from ...services.ryu_service import RyuService


router = APIRouter()
network_monitor = NetworkMonitor()
ryu_service = RyuService()


@router.get("/anomalies")
async def get_anomalies() -> dict:
    """
    Detect and return network anomalies.
    
    Returns:
        Detected anomalies
    """
    try:
        # Get current traffic statistics
        topology = await ryu_service.get_network_topology()
        
        # Detect anomalies
        anomalies = await network_monitor.detect_anomalies(topology.traffic_stats)
        
        # Group anomalies by severity
        anomaly_groups = {"error": [], "warning": [], "info": []}
        for anomaly in anomalies:
            severity = anomaly.get("severity", "info")
            anomaly_groups[severity].append(anomaly)
        
        return {
            "total_anomalies": len(anomalies),
            "anomaly_counts": {
                "error": len(anomaly_groups["error"]),
                "warning": len(anomaly_groups["warning"]),
                "info": len(anomaly_groups["info"])
            },
            "anomalies": anomalies,
            "anomalies_by_severity": anomaly_groups
        }
        
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/load-analysis")
async def get_load_analysis() -> dict:
    """
    Get network load distribution analysis.
    
    Returns:
        Load analysis results
    """
    try:
        # Get current traffic statistics
        topology = await ryu_service.get_network_topology()
        
        # Analyze load distribution
        load_analysis = await network_monitor.analyze_load_distribution(
            topology.traffic_stats
        )
        
        return load_analysis
        
    except Exception as e:
        logger.error(f"Error getting load analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_network_health() -> dict:
    """
    Get overall network health summary.
    
    Returns:
        Network health summary
    """
    try:
        # Get current traffic statistics
        topology = await ryu_service.get_network_topology()
        
        # Get health summary
        health_summary = await network_monitor.get_network_health_summary(
            topology.traffic_stats
        )
        
        return health_summary
        
    except Exception as e:
        logger.error(f"Error getting network health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_network_metrics() -> dict:
    """
    Get key network metrics.
    
    Returns:
        Network metrics summary
    """
    try:
        # Get current topology and statistics
        topology = await ryu_service.get_network_topology()
        
        # Calculate basic metrics
        total_devices = len(topology.devices)
        total_links = len(topology.links)
        total_flow_rules = len(topology.flow_rules)
        
        # Calculate traffic metrics
        total_rx_bytes = sum(stats.rx_bytes for stats in topology.traffic_stats)
        total_tx_bytes = sum(stats.tx_bytes for stats in topology.traffic_stats)
        total_rx_packets = sum(stats.rx_packets for stats in topology.traffic_stats)
        total_tx_packets = sum(stats.tx_packets for stats in topology.traffic_stats)
        
        # Calculate error metrics
        total_rx_errors = sum(stats.rx_errors for stats in topology.traffic_stats)
        total_tx_errors = sum(stats.tx_errors for stats in topology.traffic_stats)
        total_rx_dropped = sum(stats.rx_dropped for stats in topology.traffic_stats)
        total_tx_dropped = sum(stats.tx_dropped for stats in topology.traffic_stats)
        
        # Get recent anomalies
        anomalies = await network_monitor.detect_anomalies(topology.traffic_stats)
        
        return {
            "topology_metrics": {
                "devices": total_devices,
                "links": total_links,
                "flow_rules": total_flow_rules
            },
            "traffic_metrics": {
                "total_rx_bytes": total_rx_bytes,
                "total_tx_bytes": total_tx_bytes,
                "total_rx_packets": total_rx_packets,
                "total_tx_packets": total_tx_packets
            },
            "error_metrics": {
                "total_rx_errors": total_rx_errors,
                "total_tx_errors": total_tx_errors,
                "total_rx_dropped": total_rx_dropped,
                "total_tx_dropped": total_tx_dropped
            },
            "anomaly_metrics": {
                "total_anomalies": len(anomalies),
                "recent_anomalies": len([a for a in anomalies if a.get("severity") in ["error", "warning"]])
            },
            "timestamp": topology.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error getting network metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard_data() -> dict:
    """
    Get comprehensive dashboard data.
    
    Returns:
        Dashboard data including metrics, health, and recent activity
    """
    try:
        # Get current topology and statistics
        topology = await ryu_service.get_network_topology()
        
        # Get health summary
        health_summary = await network_monitor.get_network_health_summary(
            topology.traffic_stats
        )
        
        # Get load analysis
        load_analysis = await network_monitor.analyze_load_distribution(
            topology.traffic_stats
        )
        
        # Get recent anomalies
        anomalies = await network_monitor.detect_anomalies(topology.traffic_stats)
        recent_anomalies = sorted(
            anomalies, 
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )[:10]  # Last 10 anomalies
        
        # Calculate utilization statistics
        utilizations = []
        for stats in topology.traffic_stats:
            if stats.port_no is not None and stats.duration_sec > 0:
                total_bytes = stats.rx_bytes + stats.tx_bytes
                bytes_per_second = total_bytes / stats.duration_sec
                # Assume 1Gbps interface
                interface_capacity = 1_000_000_000 / 8
                utilization = min(bytes_per_second / interface_capacity * 100, 100)
                utilizations.append(utilization)
        
        avg_utilization = sum(utilizations) / len(utilizations) if utilizations else 0
        max_utilization = max(utilizations) if utilizations else 0
        
        return {
            "overview": {
                "devices": len(topology.devices),
                "links": len(topology.links),
                "flow_rules": len(topology.flow_rules),
                "health_status": health_summary.get("health_status", "unknown"),
                "total_anomalies": len(anomalies)
            },
            "performance": {
                "average_utilization": round(avg_utilization, 2),
                "max_utilization": round(max_utilization, 2),
                "load_balance_score": round(load_analysis.get("load_balance_score", 0), 2),
                "bottleneck_count": len(load_analysis.get("bottlenecks", []))
            },
            "recent_activity": {
                "recent_anomalies": recent_anomalies,
                "recommendations": health_summary.get("recommendations", [])
            },
            "traffic_summary": {
                "total_bytes": sum(stats.rx_bytes + stats.tx_bytes for stats in topology.traffic_stats),
                "total_packets": sum(stats.rx_packets + stats.tx_packets for stats in topology.traffic_stats),
                "error_rate": sum(stats.rx_errors + stats.tx_errors for stats in topology.traffic_stats)
            },
            "timestamp": topology.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 