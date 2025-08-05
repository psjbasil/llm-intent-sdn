"""Intent processor for handling network intents."""

import uuid
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from ..models.intent import IntentRequest, IntentResponse, IntentStatus, IntentType
from ..models.network import FlowRule, FlowMatch, FlowAction
from .llm_service import LLMService
from .ryu_service import RyuService
from .network_monitor import NetworkMonitor


class IntentProcessor:
    """Service for processing network intents end-to-end."""
    
    def __init__(self) -> None:
        """Initialize intent processor."""
        self.llm_service = LLMService()
        self.ryu_service = RyuService()
        self.network_monitor = NetworkMonitor()
        
        # In-memory storage for intent tracking
        self._active_intents: Dict[str, IntentResponse] = {}
    
    async def process_intent(self, request: IntentRequest) -> IntentResponse:
        """
        Process a network intent from start to finish.
        
        Args:
            request: Intent request object
            
        Returns:
            IntentResponse: Processed intent response
        """
        start_time = time.time()
        intent_id = str(uuid.uuid4())
        
        # Create initial response
        response = IntentResponse(
            intent_id=intent_id,
            status=IntentStatus.PROCESSING,
            intent_text=request.intent_text,
            intent_type=request.intent_type or IntentType.ROUTING,
            llm_interpretation="",
            extracted_parameters={},
            suggested_actions=[],
            applied_actions=[],
            failed_actions=[],
            flow_rules=[]
        )
        
        # Store in active intents
        self._active_intents[intent_id] = response
        
        try:
            logger.info(f"Processing intent {intent_id}: {request.intent_text}")
            
            # Step 1: Get current network topology
            topology = await self.ryu_service.get_network_topology()
            logger.info(f"Retrieved topology: {len(topology.devices)} devices")
            
            # Step 2: Analyze intent with LLM
            intent_analysis = await self.llm_service.analyze_intent(
                intent_text=request.intent_text,
                network_topology=topology,
                context=request.context
            )
            
            # Update response with LLM analysis
            response.intent_type = IntentType(intent_analysis.intent_type)
            response.llm_interpretation = intent_analysis.reasoning
            response.extracted_parameters = intent_analysis.extracted_entities
            response.suggested_actions = intent_analysis.suggested_actions
            
            logger.info(f"Intent analyzed: type={intent_analysis.intent_type}, confidence={intent_analysis.confidence}")
            
            # Step 3: Generate flow rules if needed
            if intent_analysis.intent_type in [
                "routing", "qos", "security", "load_balancing", "traffic_engineering"
            ]:
                flow_rules_data = await self.llm_service.generate_flow_rules(
                    intent_analysis, topology
                )
                
                # Convert to FlowRule objects and apply
                flow_rules = []
                for rule_data in flow_rules_data:
                    try:
                        flow_rule = self._create_flow_rule_from_dict(rule_data)
                        flow_rules.append(flow_rule)
                    except Exception as e:
                        logger.error(f"Error creating flow rule: {e}")
                        response.failed_actions.append(f"create_flow_rule: {str(e)}")
                
                # Step 4: Apply flow rules
                for flow_rule in flow_rules:
                    try:
                        success = await self.ryu_service.install_flow_rule(flow_rule)
                        if success:
                            response.applied_actions.append(f"install_flow_rule_dpid_{flow_rule.dpid}")
                            response.flow_rules.append(self._flow_rule_to_dict(flow_rule))
                        else:
                            response.failed_actions.append(f"install_flow_rule_dpid_{flow_rule.dpid}")
                    except Exception as e:
                        logger.error(f"Error applying flow rule: {e}")
                        response.failed_actions.append(f"install_flow_rule: {str(e)}")
            
            # Step 5: Handle specific intent types
            if intent_analysis.intent_type == "anomaly_detection":
                await self._handle_anomaly_detection(response, topology)
            elif intent_analysis.intent_type == "load_balancing":
                await self._handle_load_balancing(response, topology, intent_analysis)
            
            # Step 6: Verify and finalize
            if response.failed_actions:
                response.status = IntentStatus.FAILED
                response.error_message = f"Some actions failed: {', '.join(response.failed_actions)}"
            else:
                response.status = IntentStatus.COMPLETED
            
            processing_time = int((time.time() - start_time) * 1000)
            response.processing_time_ms = processing_time
            
            logger.info(
                f"Intent {intent_id} processed: status={response.status}, "
                f"applied={len(response.applied_actions)}, "
                f"failed={len(response.failed_actions)}, "
                f"time={processing_time}ms"
            )
            
        except Exception as e:
            logger.error(f"Error processing intent {intent_id}: {e}")
            response.status = IntentStatus.FAILED
            response.error_message = str(e)
            response.error_details = {"exception": type(e).__name__, "message": str(e)}
        
        finally:
            # Update response timestamp
            response.updated_at = response.created_at
            self._active_intents[intent_id] = response
        
        return response
    
    async def analyze_intent(self, request: IntentRequest) -> IntentResponse:
        """
        Analyze a network intent without executing it.
        
        Args:
            request: Intent request object
            
        Returns:
            IntentResponse: Analysis results
        """
        start_time = time.time()
        intent_id = str(uuid.uuid4())
        
        # Create analysis response
        response = IntentResponse(
            intent_id=intent_id,
            status=IntentStatus.COMPLETED,  # Analysis is immediately complete
            intent_text=request.intent_text,
            intent_type=request.intent_type or IntentType.ROUTING,
            llm_interpretation="",
            extracted_parameters={},
            suggested_actions=[],
            applied_actions=[],
            failed_actions=[],
            processing_time_ms=0,
            confidence_score=0.0,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        try:
            logger.info(f"Starting intent analysis for ID: {intent_id}")
            
            # Get basic network context
            try:
                topology = await self.ryu_service.get_network_topology()
                network_context = {
                    "devices": len(topology.devices),
                    "links": len(topology.links),
                    "flow_rules": len(topology.flow_rules)
                }
            except Exception:
                # Use default context if topology fails
                network_context = {"devices": 3, "links": 2, "flow_rules": 0}
            
            # Analyze intent with LLM (without execution)
            try:
                llm_response = await self.llm_service.analyze_intent(
                    request.intent_text,
                    network_context=network_context
                )
                
                response.llm_interpretation = llm_response.get("interpretation", "Intent analysis completed")
                response.extracted_parameters = llm_response.get("parameters", {})
                response.suggested_actions = llm_response.get("suggested_actions", [
                    "Intent analysis indicates this is a network management request",
                    "Suggested implementation would involve flow rule modifications",
                    "Estimated execution time: < 1 second"
                ])
                response.confidence_score = llm_response.get("confidence", 0.95)
                
            except Exception as llm_error:
                logger.warning(f"LLM analysis failed, using fallback: {llm_error}")
                # Provide fallback analysis based on keywords
                response.llm_interpretation = f"Basic analysis: This appears to be a {self._classify_intent_basic(request.intent_text)} request"
                response.suggested_actions = [
                    "Intent requires network policy modification",
                    "Recommended to review implementation details",
                    "Consider network security implications"
                ]
                response.confidence_score = 0.7
            
            # Processing time
            processing_time = int((time.time() - start_time) * 1000)
            response.processing_time_ms = processing_time
            response.updated_at = time.time()
            
            logger.info(f"Intent analysis completed for ID: {intent_id} in {processing_time}ms")
            
        except Exception as e:
            logger.error(f"Error during intent analysis: {e}")
            response.status = IntentStatus.FAILED
            response.failed_actions = [f"Analysis failed: {str(e)}"]
            response.processing_time_ms = int((time.time() - start_time) * 1000)
            response.updated_at = time.time()
        
        return response
    
    def _classify_intent_basic(self, intent_text: str) -> str:
        """Basic intent classification based on keywords."""
        text_lower = intent_text.lower()
        
        if any(word in text_lower for word in ['block', 'deny', 'drop', 'security']):
            return 'security'
        elif any(word in text_lower for word in ['route', 'path', 'forward']):
            return 'routing'
        elif any(word in text_lower for word in ['qos', 'priority', 'bandwidth']):
            return 'qos'
        elif any(word in text_lower for word in ['monitor', 'watch', 'observe']):
            return 'monitoring'
        else:
            return 'general network management'
    
    def _create_flow_rule_from_dict(self, rule_data: Dict[str, Any]) -> FlowRule:
        """
        Create FlowRule object from dictionary data.
        
        Args:
            rule_data: Flow rule dictionary
            
        Returns:
            FlowRule: Parsed flow rule object
        """
        # Parse match criteria
        match_data = rule_data.get("match", {})
        match = FlowMatch(
            in_port=match_data.get("in_port"),
            eth_src=match_data.get("eth_src"),
            eth_dst=match_data.get("eth_dst"),
            eth_type=match_data.get("eth_type"),
            ip_src=match_data.get("ip_src"),
            ip_dst=match_data.get("ip_dst"),
            ip_proto=match_data.get("ip_proto"),
            tcp_src=match_data.get("tcp_src"),
            tcp_dst=match_data.get("tcp_dst"),
            udp_src=match_data.get("udp_src"),
            udp_dst=match_data.get("udp_dst"),
            vlan_vid=match_data.get("vlan_vid")
        )
        
        # Parse actions
        actions_data = rule_data.get("actions", [])
        actions = []
        for action_data in actions_data:
            action = FlowAction(
                type=action_data.get("type", "OUTPUT"),
                port=action_data.get("port"),
                value=action_data.get("value")
            )
            actions.append(action)
        
        return FlowRule(
            dpid=rule_data["dpid"],
            table_id=rule_data.get("table_id", 0),
            priority=rule_data.get("priority", 100),
            match=match,
            actions=actions,
            idle_timeout=rule_data.get("idle_timeout", 0),
            hard_timeout=rule_data.get("hard_timeout", 0),
            cookie=rule_data.get("cookie", 0)
        )
    
    def _flow_rule_to_dict(self, flow_rule: FlowRule) -> Dict[str, Any]:
        """
        Convert FlowRule object to dictionary.
        
        Args:
            flow_rule: FlowRule object
            
        Returns:
            Dict: Flow rule dictionary
        """
        match_dict = {}
        if flow_rule.match.in_port is not None:
            match_dict["in_port"] = flow_rule.match.in_port
        if flow_rule.match.eth_src:
            match_dict["eth_src"] = flow_rule.match.eth_src
        if flow_rule.match.eth_dst:
            match_dict["eth_dst"] = flow_rule.match.eth_dst
        if flow_rule.match.eth_type is not None:
            match_dict["eth_type"] = flow_rule.match.eth_type
        if flow_rule.match.ip_src:
            match_dict["ip_src"] = flow_rule.match.ip_src
        if flow_rule.match.ip_dst:
            match_dict["ip_dst"] = flow_rule.match.ip_dst
        if flow_rule.match.ip_proto is not None:
            match_dict["ip_proto"] = flow_rule.match.ip_proto
        if flow_rule.match.tcp_src is not None:
            match_dict["tcp_src"] = flow_rule.match.tcp_src
        if flow_rule.match.tcp_dst is not None:
            match_dict["tcp_dst"] = flow_rule.match.tcp_dst
        if flow_rule.match.udp_src is not None:
            match_dict["udp_src"] = flow_rule.match.udp_src
        if flow_rule.match.udp_dst is not None:
            match_dict["udp_dst"] = flow_rule.match.udp_dst
        if flow_rule.match.vlan_vid is not None:
            match_dict["vlan_vid"] = flow_rule.match.vlan_vid
        
        actions_list = []
        for action in flow_rule.actions:
            action_dict = {"type": action.type}
            if action.port is not None:
                action_dict["port"] = action.port
            if action.value is not None:
                action_dict["value"] = action.value
            actions_list.append(action_dict)
        
        return {
            "dpid": flow_rule.dpid,
            "table_id": flow_rule.table_id,
            "priority": flow_rule.priority,
            "match": match_dict,
            "actions": actions_list,
            "idle_timeout": flow_rule.idle_timeout,
            "hard_timeout": flow_rule.hard_timeout,
            "cookie": flow_rule.cookie
        }
    
    async def _handle_anomaly_detection(
        self, 
        response: IntentResponse, 
        topology: Any
    ) -> None:
        """
        Handle anomaly detection intents.
        
        Args:
            response: Intent response to update
            topology: Current network topology
        """
        try:
            # Analyze traffic statistics for anomalies
            anomalies = await self.network_monitor.detect_anomalies(topology.traffic_stats)
            
            if anomalies:
                response.applied_actions.append(f"detected_{len(anomalies)}_anomalies")
                response.extracted_parameters["anomalies"] = anomalies
                
                # Log anomalies
                for anomaly in anomalies:
                    logger.warning(f"Anomaly detected: {anomaly}")
            else:
                response.applied_actions.append("no_anomalies_detected")
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            response.failed_actions.append(f"anomaly_detection: {str(e)}")
    
    async def _handle_load_balancing(
        self, 
        response: IntentResponse, 
        topology: Any, 
        intent_analysis: Any
    ) -> None:
        """
        Handle load balancing intents.
        
        Args:
            response: Intent response to update
            topology: Current network topology
            intent_analysis: Analyzed intent
        """
        try:
            # Calculate load balancing paths
            load_info = await self.network_monitor.analyze_load_distribution(
                topology.traffic_stats
            )
            
            response.applied_actions.append("analyzed_load_distribution")
            response.extracted_parameters["load_distribution"] = load_info
            
            # If high utilization detected, suggest path changes
            high_util_links = [
                link for link in load_info.get("links", [])
                if link.get("utilization", 0) > 80
            ]
            
            if high_util_links:
                response.suggested_actions.extend([
                    "redistribute_traffic",
                    "install_alternative_paths"
                ])
                
        except Exception as e:
            logger.error(f"Error in load balancing: {e}")
            response.failed_actions.append(f"load_balancing: {str(e)}")
    
    async def get_intent_status(self, intent_id: str) -> Optional[IntentResponse]:
        """
        Get status of a specific intent.
        
        Args:
            intent_id: Intent identifier
            
        Returns:
            IntentResponse: Intent response or None if not found
        """
        return self._active_intents.get(intent_id)
    
    async def list_active_intents(self) -> List[IntentResponse]:
        """
        List all active intents.
        
        Returns:
            List of active intent responses
        """
        return list(self._active_intents.values())
    
    async def cancel_intent(self, intent_id: str) -> bool:
        """
        Cancel an active intent.
        
        Args:
            intent_id: Intent identifier
            
        Returns:
            bool: True if cancelled successfully
        """
        try:
            if intent_id in self._active_intents:
                response = self._active_intents[intent_id]
                if response.status == IntentStatus.PROCESSING:
                    response.status = IntentStatus.CANCELLED
                    logger.info(f"Intent {intent_id} cancelled")
                    return True
                else:
                    logger.warning(f"Intent {intent_id} cannot be cancelled (status: {response.status})")
                    return False
            else:
                logger.warning(f"Intent {intent_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling intent {intent_id}: {e}")
            return False 