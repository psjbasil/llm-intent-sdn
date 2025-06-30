"""Network monitoring service for anomaly detection and analysis."""

import statistics
from typing import Any, Dict, List, Optional
from loguru import logger

from ..models.network import TrafficStats


class NetworkMonitor:
    """Service for monitoring network conditions and detecting anomalies."""
    
    def __init__(self) -> None:
        """Initialize network monitor."""
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._alert_thresholds = {
            "high_packet_loss": 0.05,  # 5% packet loss
            "high_error_rate": 0.01,   # 1% error rate
            "high_utilization": 0.80,  # 80% utilization
            "low_utilization": 0.05,   # 5% utilization
        }
    
    async def detect_anomalies(self, traffic_stats: List[TrafficStats]) -> List[Dict[str, Any]]:
        """
        Detect network anomalies from traffic statistics.
        
        Args:
            traffic_stats: List of traffic statistics
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            for stats in traffic_stats:
                dpid = stats.dpid
                port_no = stats.port_no
                
                # Calculate metrics
                metrics = self._calculate_metrics(stats)
                
                # Check for anomalies
                port_anomalies = await self._check_port_anomalies(
                    dpid, port_no, metrics, stats
                )
                anomalies.extend(port_anomalies)
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _calculate_metrics(self, stats: TrafficStats) -> Dict[str, float]:
        """
        Calculate network metrics from traffic statistics.
        
        Args:
            stats: Traffic statistics
            
        Returns:
            Dictionary of calculated metrics
        """
        total_rx = stats.rx_packets
        total_tx = stats.tx_packets
        total_packets = total_rx + total_tx
        
        # Packet loss rate
        packet_loss_rate = 0.0
        if total_packets > 0:
            dropped_packets = stats.rx_dropped + stats.tx_dropped
            packet_loss_rate = dropped_packets / total_packets
        
        # Error rate
        error_rate = 0.0
        if total_packets > 0:
            error_packets = stats.rx_errors + stats.tx_errors
            error_rate = error_packets / total_packets
        
        # Utilization (simplified calculation)
        # This would typically require interface speed information
        total_bytes = stats.rx_bytes + stats.tx_bytes
        duration = max(stats.duration_sec, 1)  # Avoid division by zero
        bytes_per_second = total_bytes / duration
        
        # Assume 1Gbps interface for utilization calculation
        interface_capacity = 1_000_000_000 / 8  # 1Gbps in bytes per second
        utilization = min(bytes_per_second / interface_capacity, 1.0)
        
        return {
            "packet_loss_rate": packet_loss_rate,
            "error_rate": error_rate,
            "utilization": utilization,
            "rx_packets": total_rx,
            "tx_packets": total_tx,
            "rx_bytes": stats.rx_bytes,
            "tx_bytes": stats.tx_bytes,
            "bytes_per_second": bytes_per_second
        }
    
    async def _check_port_anomalies(
        self,
        dpid: int,
        port_no: Optional[int],
        metrics: Dict[str, float],
        stats: TrafficStats
    ) -> List[Dict[str, Any]]:
        """
        Check for anomalies on a specific port.
        
        Args:
            dpid: Datapath ID
            port_no: Port number
            metrics: Calculated metrics
            stats: Original statistics
            
        Returns:
            List of anomalies for this port
        """
        anomalies = []
        port_key = f"{dpid}:{port_no}"
        
        # High packet loss
        if metrics["packet_loss_rate"] > self._alert_thresholds["high_packet_loss"]:
            anomalies.append({
                "type": "high_packet_loss",
                "severity": "warning",
                "dpid": dpid,
                "port_no": port_no,
                "value": metrics["packet_loss_rate"],
                "threshold": self._alert_thresholds["high_packet_loss"],
                "description": f"High packet loss rate: {metrics['packet_loss_rate']:.2%}",
                "timestamp": stats.timestamp
            })
        
        # High error rate
        if metrics["error_rate"] > self._alert_thresholds["high_error_rate"]:
            anomalies.append({
                "type": "high_error_rate",
                "severity": "error",
                "dpid": dpid,
                "port_no": port_no,
                "value": metrics["error_rate"],
                "threshold": self._alert_thresholds["high_error_rate"],
                "description": f"High error rate: {metrics['error_rate']:.2%}",
                "timestamp": stats.timestamp
            })
        
        # High utilization
        if metrics["utilization"] > self._alert_thresholds["high_utilization"]:
            anomalies.append({
                "type": "high_utilization",
                "severity": "warning",
                "dpid": dpid,
                "port_no": port_no,
                "value": metrics["utilization"],
                "threshold": self._alert_thresholds["high_utilization"],
                "description": f"High port utilization: {metrics['utilization']:.2%}",
                "timestamp": stats.timestamp
            })
        
        # Unexpected traffic (baseline comparison)
        baseline = self._baseline_stats.get(port_key, {})
        if baseline:
            # Check for significant deviations from baseline
            for metric_name in ["bytes_per_second", "rx_packets", "tx_packets"]:
                current_value = metrics.get(metric_name, 0)
                baseline_value = baseline.get(metric_name, 0)
                
                if baseline_value > 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                    if deviation > 2.0:  # 200% deviation threshold
                        anomalies.append({
                            "type": "traffic_deviation",
                            "severity": "info",
                            "dpid": dpid,
                            "port_no": port_no,
                            "metric": metric_name,
                            "current_value": current_value,
                            "baseline_value": baseline_value,
                            "deviation": deviation,
                            "description": f"Significant traffic deviation in {metric_name}: {deviation:.1f}x baseline",
                            "timestamp": stats.timestamp
                        })
        
        # Update baseline
        self._update_baseline(port_key, metrics)
        
        return anomalies
    
    def _update_baseline(self, port_key: str, metrics: Dict[str, float]) -> None:
        """
        Update baseline statistics for a port.
        
        Args:
            port_key: Port identifier
            metrics: Current metrics
        """
        if port_key not in self._baseline_stats:
            self._baseline_stats[port_key] = {}
        
        # Simple exponential moving average for baseline update
        alpha = 0.1  # Smoothing factor
        
        for metric_name, current_value in metrics.items():
            if metric_name in ["bytes_per_second", "rx_packets", "tx_packets"]:
                baseline_value = self._baseline_stats[port_key].get(metric_name, current_value)
                new_baseline = alpha * current_value + (1 - alpha) * baseline_value
                self._baseline_stats[port_key][metric_name] = new_baseline
    
    async def analyze_load_distribution(
        self, 
        traffic_stats: List[TrafficStats]
    ) -> Dict[str, Any]:
        """
        Analyze load distribution across the network.
        
        Args:
            traffic_stats: List of traffic statistics
            
        Returns:
            Load distribution analysis
        """
        try:
            if not traffic_stats:
                return {"links": [], "total_traffic": 0, "average_utilization": 0}
            
            # Group statistics by switch
            switch_stats = {}
            for stats in traffic_stats:
                dpid = stats.dpid
                if dpid not in switch_stats:
                    switch_stats[dpid] = []
                switch_stats[dpid].append(stats)
            
            # Calculate link utilizations
            links = []
            total_bytes = 0
            utilizations = []
            
            for dpid, stats_list in switch_stats.items():
                for stats in stats_list:
                    if stats.port_no is not None:  # Skip aggregate stats
                        metrics = self._calculate_metrics(stats)
                        
                        link_info = {
                            "dpid": dpid,
                            "port_no": stats.port_no,
                            "utilization": metrics["utilization"] * 100,  # Convert to percentage
                            "bytes_per_second": metrics["bytes_per_second"],
                            "rx_packets": stats.rx_packets,
                            "tx_packets": stats.tx_packets,
                            "rx_bytes": stats.rx_bytes,
                            "tx_bytes": stats.tx_bytes
                        }
                        
                        links.append(link_info)
                        total_bytes += stats.rx_bytes + stats.tx_bytes
                        utilizations.append(metrics["utilization"])
            
            # Calculate statistics
            avg_utilization = statistics.mean(utilizations) if utilizations else 0
            max_utilization = max(utilizations) if utilizations else 0
            min_utilization = min(utilizations) if utilizations else 0
            
            # Identify bottlenecks
            bottlenecks = [
                link for link in links 
                if link["utilization"] > 70  # 70% threshold for bottleneck
            ]
            
            # Identify underutilized links
            underutilized = [
                link for link in links 
                if link["utilization"] < 10  # 10% threshold for underutilization
            ]
            
            return {
                "links": links,
                "total_traffic_bytes": total_bytes,
                "average_utilization": avg_utilization * 100,
                "max_utilization": max_utilization * 100,
                "min_utilization": min_utilization * 100,
                "bottlenecks": bottlenecks,
                "underutilized_links": underutilized,
                "load_balance_score": self._calculate_load_balance_score(utilizations)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing load distribution: {e}")
            return {"error": str(e)}
    
    def _calculate_load_balance_score(self, utilizations: List[float]) -> float:
        """
        Calculate a load balance score (0-100, higher is better).
        
        Args:
            utilizations: List of link utilizations
            
        Returns:
            Load balance score
        """
        if not utilizations:
            return 100.0
        
        # Calculate coefficient of variation (CV)
        # Lower CV indicates better load balance
        mean_util = statistics.mean(utilizations)
        
        if mean_util == 0:
            return 100.0
        
        variance = statistics.variance(utilizations)
        std_dev = variance ** 0.5
        cv = std_dev / mean_util
        
        # Convert CV to score (0-100)
        # CV of 0 = perfect balance (score 100)
        # CV of 1 = poor balance (score 0)
        score = max(0, 100 * (1 - min(cv, 1)))
        
        return score
    
    async def get_network_health_summary(
        self, 
        traffic_stats: List[TrafficStats]
    ) -> Dict[str, Any]:
        """
        Get overall network health summary.
        
        Args:
            traffic_stats: List of traffic statistics
            
        Returns:
            Network health summary
        """
        try:
            anomalies = await self.detect_anomalies(traffic_stats)
            load_analysis = await self.analyze_load_distribution(traffic_stats)
            
            # Count anomalies by severity
            anomaly_counts = {"error": 0, "warning": 0, "info": 0}
            for anomaly in anomalies:
                severity = anomaly.get("severity", "info")
                anomaly_counts[severity] = anomaly_counts.get(severity, 0) + 1
            
            # Determine overall health status
            if anomaly_counts["error"] > 0:
                health_status = "critical"
            elif anomaly_counts["warning"] > 0:
                health_status = "warning" 
            elif anomaly_counts["info"] > 0:
                health_status = "attention"
            else:
                health_status = "healthy"
            
            return {
                "health_status": health_status,
                "anomaly_counts": anomaly_counts,
                "total_anomalies": len(anomalies),
                "load_balance_score": load_analysis.get("load_balance_score", 0),
                "average_utilization": load_analysis.get("average_utilization", 0),
                "bottleneck_count": len(load_analysis.get("bottlenecks", [])),
                "recent_anomalies": anomalies[-5:] if anomalies else [],
                "recommendations": self._generate_recommendations(anomalies, load_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error getting network health summary: {e}")
            return {"error": str(e), "health_status": "unknown"}
    
    def _generate_recommendations(
        self, 
        anomalies: List[Dict[str, Any]], 
        load_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Generate network optimization recommendations.
        
        Args:
            anomalies: List of detected anomalies
            load_analysis: Load distribution analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Anomaly-based recommendations
        error_count = sum(1 for a in anomalies if a.get("severity") == "error")
        if error_count > 0:
            recommendations.append(f"Investigate {error_count} critical network errors")
        
        high_util_count = sum(1 for a in anomalies if a.get("type") == "high_utilization")
        if high_util_count > 0:
            recommendations.append(f"Consider load balancing for {high_util_count} overutilized links")
        
        # Load balancing recommendations
        load_score = load_analysis.get("load_balance_score", 100)
        if load_score < 50:
            recommendations.append("Poor load distribution detected - consider traffic engineering")
        
        bottlenecks = load_analysis.get("bottlenecks", [])
        if bottlenecks:
            recommendations.append(f"Address {len(bottlenecks)} network bottlenecks")
        
        underutilized = load_analysis.get("underutilized_links", [])
        if len(underutilized) > len(load_analysis.get("links", [])) / 2:
            recommendations.append("Many underutilized links - optimize traffic distribution")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Network is operating within normal parameters")
        
        return recommendations 