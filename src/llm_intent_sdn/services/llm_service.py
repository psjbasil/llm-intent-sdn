"""LLM service for processing network intents."""

import json
import re
import time
from typing import Any, Dict, List, Optional
import httpx
from loguru import logger

from ..config import settings
from ..models.llm import LLMMessage, LLMRequest, LLMResponse, IntentAnalysis
from ..models.network import NetworkTopology


class LLMService:
    """Service for interacting with Large Language Models."""
    
    def __init__(self) -> None:
        """Initialize LLM service."""
        self.base_url = settings.openai_base_url
        self.api_key = settings.openai_api_key
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        
        self._system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load the system prompt for intent processing."""
        return """You are an expert network engineer and AI assistant specializing in Software Defined Networking (SDN) and intent-based networking. Your role is to interpret natural language network intents and translate them into specific network actions using RYU controller APIs.

CAPABILITIES:
- Analyze network topology and traffic patterns
- Interpret user intents in natural language
- Generate flow rules and network configurations
- Detect and respond to network anomalies
- Optimize traffic routing and load balancing

RESPONSE FORMAT:
You MUST respond with ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.

The JSON object must contain:
{
    "intent_type": "routing|qos|security|load_balancing|anomaly_detection|traffic_engineering",
    "confidence": 0.0-1.0,
    "extracted_entities": {
        "source": "source host/switch",
        "destination": "destination host/switch", 
        "parameters": "additional parameters"
    },
    "suggested_actions": ["list", "of", "actions"],
    "reasoning": "Your analysis and reasoning process"
}

CRITICAL: Ensure all JSON strings are properly quoted with double quotes. No trailing commas allowed.

IMPORTANT RULES:
1. Respond ONLY with JSON - no other text
2. Ensure the JSON is valid and complete
3. Use proper JSON syntax with double quotes
4. Do not include markdown code blocks or ```json``` tags
5. For security intents like blocking traffic, use intent_type: "security"
6. For routing intents, use intent_type: "routing"

EXAMPLES:
For "Block all traffic from host h1 to host h2":
{
    "intent_type": "security",
    "confidence": 0.95,
    "extracted_entities": {
        "source": "h1",
        "destination": "h2",
        "action": "block"
    },
    "suggested_actions": ["Create flow rule to drop traffic from h1 to h2"],
    "reasoning": "This is a security intent to block traffic between specific hosts"
}

For "Route traffic from h1 to h3 through the fastest path":
{
    "intent_type": "routing",
    "confidence": 0.90,
    "extracted_entities": {
        "source": "h1",
        "destination": "h3",
        "optimization": "fastest_path"
    },
    "suggested_actions": ["Calculate shortest path using Dijkstra algorithm", "Install flow rules along the calculated path"],
    "reasoning": "This is a routing intent to optimize path between hosts"
}

For "Set high priority QoS for video traffic on port 80":
{
    "intent_type": "qos",
    "confidence": 0.85,
    "extracted_entities": {
        "traffic_type": "video",
        "port": "80",
        "priority": "high"
    },
    "suggested_actions": ["Create QoS flow rules with high priority", "Set DSCP marking for video traffic"],
    "reasoning": "This is a QoS intent to prioritize specific traffic type"
}

For "Monitor bandwidth usage on all network links":
{
    "intent_type": "monitoring",
    "confidence": 0.88,
    "extracted_entities": {
        "metric": "bandwidth",
        "scope": "all_links"
    },
    "suggested_actions": ["Enable link monitoring", "Collect traffic statistics", "Generate performance report"],
    "reasoning": "This is a monitoring intent to track network performance"
}"""

    def _clean_json_response(self, content: str) -> str:
        """Clean and extract JSON from LLM response."""
        if not content:
            return "{}"
        
        # Remove markdown code blocks
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        # Try to find JSON array first (for flow rules)
        try:
            # Find the first [ and last ]
            start = content.find('[')
            if start != -1:
                # Find matching closing bracket
                bracket_count = 0
                end = start
                for i, char in enumerate(content[start:], start):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end = i + 1
                            break
                
                if bracket_count == 0:
                    json_content = content[start:end]
                    # Basic JSON validation
                    parsed = json.loads(json_content)  # Test if it's valid JSON
                    logger.debug(f"Successfully parsed JSON array with {len(parsed) if isinstance(parsed, list) else 'unknown'} items")
                    return json_content
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse JSON array: {e}")
            pass
        
        # Try to find JSON object with better regex
        # Look for content between { and } with proper nesting
        try:
            # Find the first { and last }
            start = content.find('{')
            if start == -1:
                return "{}"
            
            # Find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(content[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if brace_count == 0:
                json_content = content[start:end]
                # Basic JSON validation
                json.loads(json_content)  # Test if it's valid JSON
                return json_content
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: try to find any JSON-like structure
        json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # If no JSON found, return empty object
        return "{}"

    async def analyze_intent(
        self, 
        intent_text: str, 
        network_topology: Optional[NetworkTopology] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentAnalysis:
        """
        Analyze a network intent using LLM.
        
        Args:
            intent_text: Natural language intent description
            network_topology: Current network topology 
            context: Additional context information
            
        Returns:
            IntentAnalysis: Parsed intent analysis
        """
        try:
            start_time = time.time()
            
            # Prepare context information
            topology_context = ""
            if network_topology:
                topology_context = f"""
CURRENT NETWORK TOPOLOGY:
Devices: {len(network_topology.devices)} switches/hosts
Links: {len(network_topology.links)} connections
Active Flow Rules: {len(network_topology.flow_rules)}
"""
            
            additional_context = ""
            if context:
                additional_context = f"\nADDITIONAL CONTEXT: {json.dumps(context, indent=2)}"
            
            # Create messages
            messages = [
                LLMMessage(role="system", content=self._system_prompt),
                LLMMessage(
                    role="user", 
                    content=f"""NETWORK INTENT: {intent_text}
{topology_context}{additional_context}

Please analyze this intent and provide a structured response."""
                )
            ]
            
            # Create LLM request
            llm_request = LLMRequest(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Get LLM response
            llm_response = await self._call_llm(llm_request)
            
            # Parse response
            try:
                # Clean and extract JSON from response
                cleaned_content = self._clean_json_response(llm_response.content)
                
                # Check if content is valid for JSON parsing
                if not cleaned_content or cleaned_content.strip() == "":
                    raise json.JSONDecodeError("Empty content", "", 0)
                
                response_data = json.loads(cleaned_content)
                
                # Handle case where LLM returns a list instead of dict
                if isinstance(response_data, list):
                    logger.warning(f"LLM returned list instead of dict for intent analysis: {response_data}")
                    # Try to extract the first dict item from the list
                    if response_data and isinstance(response_data[0], dict):
                        response_data = response_data[0]
                        logger.info("Successfully extracted intent analysis from first list item")
                    elif response_data and isinstance(response_data[0], str):
                        # LLM returned a list of strings (suggested actions)
                        response_data = {
                            "intent_type": "security" if "block" in intent_text.lower() else "routing",
                            "confidence": 0.8,
                            "extracted_entities": {},
                            "suggested_actions": response_data,  # Use the list as suggested actions
                            "reasoning": f"Intent analysis based on action list: {', '.join(response_data[:3])}"
                        }
                        logger.info("Converted LLM action list to intent analysis structure")
                    else:
                        # Fallback to default values
                        response_data = {
                            "intent_type": "security" if "block" in intent_text.lower() else "routing",
                            "confidence": 0.5,
                            "extracted_entities": {},
                            "suggested_actions": ["Manual review required"],
                            "reasoning": "LLM returned unexpected list format"
                        }
                
                # Create IntentAnalysis object
                # Safely get intent_type with validation
                intent_type_raw = response_data.get("intent_type", "routing")
                try:
                    # Validate intent_type is in allowed values
                    from ..models.intent import IntentType
                    intent_type = intent_type_raw if intent_type_raw in [e.value for e in IntentType] else "routing"
                except:
                    intent_type = "routing"
                
                analysis = IntentAnalysis(
                    intent_type=intent_type,
                    confidence=response_data.get("confidence", 0.5),
                    extracted_entities=response_data.get("extracted_entities", {}),
                    suggested_actions=response_data.get("suggested_actions", []),
                    reasoning=response_data.get("reasoning", "")
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                logger.info(
                    f"Intent analyzed successfully in {processing_time}ms: {intent_text[:50]}..."
                )
                
                return analysis
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                logger.error(f"Raw LLM response: {llm_response.content}")
                logger.error(f"Cleaned content: {cleaned_content}")
                
                # Try to extract any useful information from the response
                fallback_reasoning = f"Failed to parse LLM response: {llm_response.content[:200]}..."
                
                # Check if we can extract intent type from the raw response
                intent_type = "routing"
                if "security" in llm_response.content.lower():
                    intent_type = "security"
                elif "block" in llm_response.content.lower() or "drop" in llm_response.content.lower():
                    intent_type = "security"
                
                # Return fallback analysis
                return IntentAnalysis(
                    intent_type=intent_type,
                    confidence=0.1,
                    extracted_entities={},
                    suggested_actions=["Manual review required due to parsing error"],
                    reasoning=fallback_reasoning
                )
                
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return IntentAnalysis(
                intent_type="routing",  # Default to routing instead of "error"
                confidence=0.0,
                extracted_entities={},
                suggested_actions=["Manual review required due to analysis error"],
                reasoning=f"Error during analysis: {str(e)}"
            )
    
    async def _call_llm(self, request: LLMRequest) -> LLMResponse:
        """
        Make API call to LLM service.
        
        Args:
            request: LLM request object
            
        Returns:
            LLMResponse: LLM response object
        """
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content} 
                for msg in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        async with httpx.AsyncClient(timeout=settings.intent_timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                
                # Check if content is empty or None
                if not content or content.strip() == "":
                    logger.warning("LLM returned empty content")
                    content = "Empty response from LLM"
                
                usage = data.get("usage", {})
                
                response_time = int((time.time() - start_time) * 1000)
                
                return LLMResponse(
                    content=content,
                    model=request.model,
                    usage=usage,
                    finish_reason=data["choices"][0].get("finish_reason", "stop"),
                    response_time_ms=response_time
                )
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error calling LLM API: {e}")
                raise
            except httpx.TimeoutException:
                logger.error("Timeout calling LLM API")
                raise
            except Exception as e:
                logger.error(f"Unexpected error calling LLM API: {e}")
                raise
    
    async def generate_flow_rules(
        self,
        intent_analysis: IntentAnalysis,
        network_topology: NetworkTopology
    ) -> List[Dict[str, Any]]:
        """
        Generate specific flow rules based on intent analysis.
        
        Args:
            intent_analysis: Analyzed intent
            network_topology: Current network topology
            
        Returns:
            List of flow rule dictionaries
        """
        try:
            logger.info(f"Generating flow rules for intent type: {intent_analysis.intent_type}")
            logger.info(f"Intent entities: {intent_analysis.extracted_entities}")
            logger.info(f"Topology has {len(network_topology.devices)} devices and {len(network_topology.links)} links")
            
            # Create a focused prompt for flow rule generation based on intent type
            prompt = self._generate_flow_rule_prompt(intent_analysis, network_topology)
            
            messages = [
                LLMMessage(role="system", content="You are an OpenFlow expert. Generate flow rules as a JSON array. Return ONLY the JSON array, no other text."),
                LLMMessage(role="user", content=prompt)
            ]
            
            request = LLMRequest(
                messages=messages,
                model=self.model,
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=2000
            )
            
            response = await self._call_llm(request)
            
            # Parse flow rules
            try:
                # Clean the response first
                cleaned_content = self._clean_json_response(response.content)
                logger.info(f"Raw LLM response: {response.content}")
                logger.info(f"Cleaned content: {cleaned_content}")
                
                # Check if cleaned content is empty or invalid
                if not cleaned_content or cleaned_content.strip() == "" or cleaned_content.strip() == "{}":
                    logger.error("LLM returned empty or invalid content")
                    return []
                
                flow_rules = json.loads(cleaned_content)
                if isinstance(flow_rules, list):
                    logger.info(f"Successfully parsed {len(flow_rules)} flow rules")
                    return flow_rules
                elif isinstance(flow_rules, dict):
                    # If LLM returned a dict instead of list, try to extract rules
                    logger.warning(f"LLM returned dict instead of list, attempting to extract rules")
                    if "rules" in flow_rules:
                        rules = flow_rules["rules"]
                        if isinstance(rules, list):
                            logger.info(f"Extracted {len(rules)} flow rules from dict")
                            return rules
                    # If no rules found, return as single rule
                    logger.info("Converting dict to single rule")
                    return [flow_rules]
                else:
                    logger.warning(f"LLM returned non-list flow rules: {type(flow_rules)}")
                    logger.warning(f"Content: {response.content}")
                    return []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse flow rules JSON: {e}")
                logger.error(f"Raw flow rules response: {response.content}")
                logger.error(f"Cleaned content: {cleaned_content}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating flow rules: {e}")
            return []
    
    def _generate_flow_rule_prompt(self, intent_analysis, network_topology) -> str:
        """Generate specialized prompts for different intent types."""
        intent_type = intent_analysis.intent_type.lower()
        topology_str = self._format_topology_for_llm(network_topology)
        entities = intent_analysis.extracted_entities
        
        if intent_type == "security":
            return self._generate_security_prompt(entities, topology_str)
        elif intent_type == "routing":
            return self._generate_routing_prompt(entities, topology_str)
        elif intent_type == "qos":
            return self._generate_qos_prompt(entities, topology_str)
        elif intent_type == "monitoring":
            return self._generate_monitoring_prompt(entities, topology_str)
        else:
            # Fallback to security prompt
            return self._generate_security_prompt(entities, topology_str)
    
    def _generate_security_prompt(self, entities, topology_str) -> str:
        """Generate prompt for security intents."""
        source = entities.get("source", "h1")
        destination = entities.get("destination", "h2")
        
        return f"""
You are an expert OpenFlow engineer. Generate OpenFlow rules for SECURITY intent.

TASK: Block traffic between {source} and {destination}

NETWORK TOPOLOGY:
{topology_str}

REQUIREMENTS:
1. Block traffic BIDIRECTIONALLY ({source}->{destination} AND {destination}->{source})
2. Install rules on ALL switches that connect {source} and {destination}
3. Use IP-based matching with exact IPs
4. Use DROP action for blocking
5. Set priority 1000+
6. Use eth_type: 2048 for IPv4

FORMAT: Return ONLY a JSON array of flow rules, no other text.

RESPOND WITH ONLY THE JSON ARRAY:
"""
    
    def _generate_routing_prompt(self, entities, topology_str) -> str:
        """Generate prompt for routing intents."""
        source = entities.get("source", "h1")
        destination = entities.get("destination", "h3")
        optimization = entities.get("optimization", "shortest_path")
        
        return f"""
You are an expert OpenFlow engineer. Generate OpenFlow rules for ROUTING optimization.

TASK: Route traffic from {source} to {destination} using {optimization}

NETWORK TOPOLOGY:
{topology_str}

REQUIREMENTS:
1. Calculate the shortest/fastest path from {source} to {destination}
2. Install forwarding rules along the calculated path
3. Set appropriate output ports for each switch
4. Use higher priority (800+) to override default rules
5. Use IP-based matching for specific host traffic

EXAMPLE ROUTING RULE:
{{"dpid": 1, "table_id": 0, "priority": 800, "match": {{"eth_type": 2048, "ip_src": "10.0.0.1", "ip_dst": "10.0.0.3"}}, "actions": [{{"type": "OUTPUT", "port": 3}}], "idle_timeout": 300, "hard_timeout": 0}}

FORMAT: Return ONLY a JSON array of flow rules, no other text.

RESPOND WITH ONLY THE JSON ARRAY:
"""
    
    def _generate_qos_prompt(self, entities, topology_str) -> str:
        """Generate prompt for QoS intents."""
        traffic_type = entities.get("traffic_type", "video")
        port = entities.get("port", "80")
        priority = entities.get("priority", "high")
        
        priority_value = 900 if priority == "high" else 700 if priority == "medium" else 500
        
        return f"""
You are an expert OpenFlow engineer. Generate OpenFlow rules for QoS control.

TASK: Set {priority} priority for {traffic_type} traffic on port {port}

NETWORK TOPOLOGY:
{topology_str}

REQUIREMENTS:
1. Create rules to identify {traffic_type} traffic on TCP port {port}
2. Set priority {priority_value} for {priority} priority traffic
3. Use queue actions or DSCP marking if available
4. Install on all switches to ensure QoS throughout network
5. Match both TCP source and destination ports

EXAMPLE QoS RULE:
{{"dpid": 1, "table_id": 0, "priority": {priority_value}, "match": {{"eth_type": 2048, "ip_proto": 6, "tcp_dst": {port}}}, "actions": [{{"type": "SET_QUEUE", "queue_id": 1}}, {{"type": "OUTPUT", "port": "NORMAL"}}], "idle_timeout": 0, "hard_timeout": 0}}

FORMAT: Return ONLY a JSON array of flow rules, no other text.

RESPOND WITH ONLY THE JSON ARRAY:
"""
    
    def _generate_monitoring_prompt(self, entities, topology_str) -> str:
        """Generate prompt for monitoring intents."""
        metric = entities.get("metric", "bandwidth")
        scope = entities.get("scope", "all_links")
        
        return f"""
You are an expert OpenFlow engineer. Generate OpenFlow rules for MONITORING.

TASK: Enable {metric} monitoring for {scope}

NETWORK TOPOLOGY:
{topology_str}

REQUIREMENTS:
1. Create monitoring rules to track {metric} usage
2. Use low priority (100) to capture all traffic
3. Forward to controller for statistics collection
4. Install on all switches for comprehensive monitoring
5. Use table-miss rules to catch all unmatched traffic

EXAMPLE MONITORING RULE:
{{"dpid": 1, "table_id": 0, "priority": 100, "match": {{}}, "actions": [{{"type": "OUTPUT", "port": "CONTROLLER"}}, {{"type": "OUTPUT", "port": "NORMAL"}}], "idle_timeout": 0, "hard_timeout": 0}}

FORMAT: Return ONLY a JSON array of flow rules, no other text.

RESPOND WITH ONLY THE JSON ARRAY:
"""
    
    def _format_topology_for_llm(self, topology: NetworkTopology) -> str:
        """Format network topology for LLM consumption."""
        devices_info = []
        host_mappings = {}
        host_ip_mappings = {}
        
        # Process switches and their ports
        for device in topology.devices:
            if device.device_type.value == "switch":
                device_info = f"Switch {device.name} (DPID: {device.dpid}):"
                if device.ports:
                    for port in device.ports:
                        device_info += f"\n  - Port {port.port_no}: {port.name}"
                        logger.info(f"Processing port {port.port_no}: {port.name} on switch {device.name}")
                        # Extract host information from port names
                        # Try multiple patterns for host port detection
                        host_detected = False
                        
                        # Pattern 1: s1-eth1, s1-eth2 format (Mininet standard)
                        if "eth" in port.name.lower() and "-" in port.name:
                            logger.info(f"Found eth port: {port.name} on device {device.name}")
                            port_parts = port.name.split('-')
                            if len(port_parts) >= 2:
                                switch_name = port_parts[0]
                                eth_part = port_parts[1]
                                if eth_part.startswith('eth'):
                                    host_num = eth_part.replace('eth', '')
                                    if host_num.isdigit():
                                        # Only map if this is a host port (not inter-switch link)
                                        # Based on Mininet topology:
                                        # s1-eth1:h1, s1-eth2:h5 (host ports)
                                        # s1-eth3:s2 (inter-switch link)
                                        # s2-eth1:h2, s2-eth2:h4 (host ports)
                                        # s2-eth3:s1, s2-eth4:s3 (inter-switch links)
                                        # s3-eth1:h3, s3-eth2:h6 (host ports)
                                        # s3-eth3:s2 (inter-switch link)
                                        
                                        # Check if this is a host port based on topology
                                        is_host_port = False
                                        if switch_name == "s1" and host_num in ["1", "2"]:  # s1-eth1:h1, s1-eth2:h5
                                            is_host_port = True
                                        elif switch_name == "s2" and host_num in ["1", "2"]:  # s2-eth1:h2, s2-eth2:h4
                                            is_host_port = True
                                        elif switch_name == "s3" and host_num in ["1", "2"]:  # s3-eth1:h3, s3-eth2:h6
                                            is_host_port = True
                                        
                                        if is_host_port:
                                            # Map host number to actual host name based on topology
                                            if switch_name == "s1" and host_num == "1":
                                                host_name = "h1"
                                            elif switch_name == "s1" and host_num == "2":
                                                host_name = "h5"
                                            elif switch_name == "s2" and host_num == "1":
                                                host_name = "h2"
                                            elif switch_name == "s2" and host_num == "2":
                                                host_name = "h4"
                                            elif switch_name == "s3" and host_num == "1":
                                                host_name = "h3"
                                            elif switch_name == "s3" and host_num == "2":
                                                host_name = "h6"
                                            else:
                                                host_name = f"h{host_num}"
                                            
                                            host_ip = f"10.0.0.{host_name[1]}"  # Extract number from h1, h2, etc.
                                            # Map host to this switch
                                            host_mappings[host_name] = device.dpid
                                            host_ip_mappings[host_name] = host_ip
                                            logger.info(f"Mapped {host_name} ({host_ip}) to switch {device.name} (DPID: {device.dpid})")
                                            host_detected = True
                        
                        # Pattern 2: Check if port name contains host reference (e.g., "h1", "h2")
                        if not host_detected:
                            for i in range(1, 10):  # Check for h1 to h9
                                host_pattern = f"h{i}"
                                if host_pattern in port.name.lower():
                                    host_name = f"h{i}"
                                    host_ip = f"10.0.0.{i}"
                                    host_mappings[host_name] = device.dpid
                                    host_ip_mappings[host_name] = host_ip
                                    logger.info(f"Mapped {host_name} ({host_ip}) to switch {device.name} via pattern matching")
                                    host_detected = True
                                    break
                        
                        # Pattern 3: Fallback by port number is unreliable and may misclassify inter-switch links
                        # Disable naive port_no->host mapping to avoid false hosts and missing some hosts
                        # Keep only explicit name-based detections above
                        
                        if not host_detected:
                            logger.info(f"Port {port.name} not identified as host port")
                devices_info.append(device_info)
            elif device.device_type.value == "host":
                # Store host mappings for later use
                host_mappings[device.name] = device.dpid
                devices_info.append(f"Host {device.name} (DPID: {device.dpid})")
        
        # Process inter-switch links
        links_info = []
        for link in topology.links:
            links_info.append(
                f"Switch {link.src_dpid} Port {link.src_port_no} <-> Switch {link.dst_dpid} Port {link.dst_port_no}"
            )
        
        # Build host-to-switch connections
        host_connections = []
        for device in topology.devices:
            if device.device_type.value == "switch":
                for port in device.ports:
                    # Use the same host detection logic as above
                    host_detected = False
                    
                    # Pattern 1: s1-eth1, s1-eth2 format (Mininet standard)
                    if "eth" in port.name.lower() and "-" in port.name:
                        port_parts = port.name.split('-')
                        if len(port_parts) >= 2:
                            switch_name = port_parts[0]
                            eth_part = port_parts[1]
                            if eth_part.startswith('eth'):
                                host_num = eth_part.replace('eth', '')
                                if host_num.isdigit():
                                    # Check if this is a host port based on topology
                                    is_host_port = False
                                    if switch_name == "s1" and host_num in ["1", "2"]:  # s1-eth1:h1, s1-eth2:h5
                                        is_host_port = True
                                    elif switch_name == "s2" and host_num in ["1", "2"]:  # s2-eth1:h2, s2-eth2:h4
                                        is_host_port = True
                                    elif switch_name == "s3" and host_num in ["1", "2"]:  # s3-eth1:h3, s3-eth2:h6
                                        is_host_port = True
                                    
                                    if is_host_port:
                                        # Map host number to actual host name based on topology
                                        if switch_name == "s1" and host_num == "1":
                                            host_name = "h1"
                                        elif switch_name == "s1" and host_num == "2":
                                            host_name = "h5"
                                        elif switch_name == "s2" and host_num == "1":
                                            host_name = "h2"
                                        elif switch_name == "s2" and host_num == "2":
                                            host_name = "h4"
                                        elif switch_name == "s3" and host_num == "1":
                                            host_name = "h3"
                                        elif switch_name == "s3" and host_num == "2":
                                            host_name = "h6"
                                        else:
                                            host_name = f"h{host_num}"
                                        
                                        host_ip = f"10.0.0.{host_name[1]}"  # Extract number from h1, h2, etc.
                                        if host_name in host_mappings and host_mappings[host_name] == device.dpid:
                                            host_connections.append(f"Host {host_name} ({host_ip}) -> Switch {switch_name} (DPID: {device.dpid}) Port {port.port_no}")
                                            host_detected = True
                    
                    # Pattern 2: Check if port name contains host reference
                    if not host_detected:
                        for i in range(1, 10):
                            host_pattern = f"h{i}"
                            if host_pattern in port.name.lower():
                                host_name = f"h{i}"
                                host_ip = f"10.0.0.{i}"
                                if host_name in host_mappings and host_mappings[host_name] == device.dpid:
                                    host_connections.append(f"Host {host_name} ({host_ip}) -> Switch {device.name} (DPID: {device.dpid}) Port {port.port_no}")
                                    host_detected = True
                                    break
                    
                    # Pattern 3: Port number corresponds to host number
                    if not host_detected and port.port_no <= 6:
                        host_name = f"h{port.port_no}"
                        host_ip = f"10.0.0.{port.port_no}"
                        if host_name in host_mappings and host_mappings[host_name] == device.dpid:
                            host_connections.append(f"Host {host_name} ({host_ip}) -> Switch {device.name} (DPID: {device.dpid}) Port {port.port_no}")
                            host_detected = True
        
        # Build detailed topology summary
        topology_str = f"""
NETWORK TOPOLOGY:

SWITCHES:
{chr(10).join(devices_info)}

HOST CONNECTIONS:
{chr(10).join(host_connections)}

HOST IP MAPPINGS:
{chr(10).join([f"- {host}: {ip}" for host, ip in host_ip_mappings.items()])}

INTER-SWITCH LINKS:
{chr(10).join(links_info) if links_info else "No inter-switch links"}

FOR BLOCKING h1<->h2:
- h1 (10.0.0.1) connects to Switch {host_mappings.get('h1', 'unknown')} (DPID: {host_mappings.get('h1', 'unknown')})
- h2 (10.0.0.2) connects to Switch {host_mappings.get('h2', 'unknown')} (DPID: {host_mappings.get('h2', 'unknown')})
- Install DROP rules on switches that connect h1 and h2 for bidirectional blocking
"""
        
        logger.info(f"Formatted topology for LLM with {len(host_mappings)} hosts and {len(links_info)} links")
        return topology_str 