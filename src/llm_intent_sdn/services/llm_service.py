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
    
    def __init__(self):
        """Initialize LLM service."""
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model_name
        
        # Detect API type
        self.is_ollama = settings.use_ollama or "localhost:11434" in self.base_url or "127.0.0.1:11434" in self.base_url
        self.is_gemini = settings.use_gemini or "generativelanguage.googleapis.com" in self.base_url
        
        # Set API key based on type
        if self.is_gemini:
            self.api_key = settings.gemini_api_key
        else:
            self.api_key = settings.openai_api_key
        
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

For "Give video conference traffic highest priority from h1 to h2":
{
    "intent_type": "qos",
    "confidence": 0.92,
    "extracted_entities": {
        "source": "h1",
        "destination": "h2",
        "traffic_type": "video_conference",
        "priority": "highest"
    },
    "suggested_actions": ["Install QoS flow rules with highest priority queue", "Set traffic shaping for video conference traffic"],
    "reasoning": "This is a QoS intent to prioritize video conference traffic between specific hosts"
}

For "Prioritize database traffic from h3 to h4":
{
    "intent_type": "qos",
    "confidence": 0.88,
    "extracted_entities": {
        "source": "h3",
        "destination": "h4",
        "traffic_type": "database",
        "priority": "high"
    },
    "suggested_actions": ["Create QoS flow rules for database traffic", "Apply priority queuing"],
    "reasoning": "This is a QoS intent to prioritize database traffic between hosts"
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
                        # Try to infer source/destination from the intent_text
                        src, dst = self._infer_hosts_from_text(intent_text)
                        response_data = {
                            "intent_type": "security" if any(k in intent_text.lower() for k in ["block", "deny"]) else "routing",
                            "confidence": 0.8,
                            "extracted_entities": {"source": src, "destination": dst} if src and dst else {},
                            "suggested_actions": response_data,
                            "reasoning": f"Intent analysis based on action list: {', '.join(response_data[:3])}"
                        }
                        logger.info("Converted LLM action list to intent analysis structure with inferred hosts")
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
                
                # Enrich entities if missing hosts
                entities = response_data.get("extracted_entities", {}) or {}
                if not entities.get("source") or not entities.get("destination"):
                    src, dst = self._infer_hosts_from_text(intent_text)
                    if src and dst:
                        entities["source"] = entities.get("source", src)
                        entities["destination"] = entities.get("destination", dst)

                analysis = IntentAnalysis(
                    intent_type=intent_type,
                    confidence=response_data.get("confidence", 0.5),
                    extracted_entities=entities,
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
        
        # Prepare headers and payload based on API type
        if self.is_ollama:
            headers = {"Content-Type": "application/json"}
            # Ollama API format
            payload = {
                "model": request.model,
                "messages": [
                    {"role": msg.role, "content": msg.content} 
                    for msg in request.messages
                ],
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }
            endpoint = "/api/chat"
        elif self.is_gemini:
            headers = {"Content-Type": "application/json"}
            # Convert OpenAI format to Gemini format
            gemini_messages = []
            for msg in request.messages:
                if msg.role == "system":
                    # Gemini doesn't have system role, prepend to user message
                    if gemini_messages and gemini_messages[-1]["role"] == "user":
                        gemini_messages[-1]["parts"][0]["text"] = f"{msg.content}\n\n{gemini_messages[-1]['parts'][0]['text']}"
                    else:
                        # If no user message yet, create one
                        gemini_messages.append({
                            "role": "user",
                            "parts": [{"text": msg.content}]
                        })
                else:
                    gemini_messages.append({
                        "role": msg.role,
                        "parts": [{"text": msg.content}]
                    })
            
            payload = {
                "contents": gemini_messages,
                "generationConfig": {
                    "temperature": request.temperature,
                    "maxOutputTokens": request.max_tokens,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            endpoint = f"/{request.model}:generateContent?key={self.api_key}"
        else:
            # OpenAI API format
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
            endpoint = "/chat/completions"
        
        # Use longer timeout for local Ollama if configured, otherwise default intent timeout
        client_timeout = settings.ollama_timeout if getattr(settings, 'ollama_timeout', None) and self.is_ollama else settings.intent_timeout
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Extract content based on API type
                if self.is_ollama:
                    content = data.get("message", {}).get("content", "")
                    # Ollama doesn't provide usage or finish_reason in the same format
                    usage = {}
                    finish_reason = "stop"
                elif self.is_gemini:
                    content = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    # Gemini doesn't provide usage or finish_reason in the same format
                    usage = {}
                    finish_reason = "stop"
                else:
                    content = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    finish_reason = data["choices"][0].get("finish_reason", "stop")
                
                # Check if content is empty or None
                if not content or content.strip() == "":
                    logger.warning("LLM returned empty content")
                    content = "Empty response from LLM"
                
                response_time = int((time.time() - start_time) * 1000)
                
                return LLMResponse(
                    content=content,
                    model=request.model,
                    usage=usage,
                    finish_reason=finish_reason,
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
                logger.debug(f"Raw LLM response: {response.content}")
                logger.debug(f"Cleaned content: {cleaned_content}")
                
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
        entities = intent_analysis.extracted_entities
        
        # Extract source and destination for routing intents
        source = entities.get("source")
        destination = entities.get("destination")
        
        # Format topology with source and destination for routing intents
        topology_str = self._format_topology_for_llm(network_topology, source, destination)
        
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
1. **CRITICAL**: Only use valid switch DPIDs: 1, 2, 3, 4 (NEVER use 5, 6, 7, 8, 9, 10)
2. Block traffic BIDIRECTIONALLY ({source}->{destination} AND {destination}->{source})
3. Install rules on ALL switches that connect {source} and {destination}
4. Use IP-based matching with exact IPs
5. **CRITICAL**: Use DROP action for blocking traffic
6. Set priority 1000+  
7. Use eth_type: 2048 for IPv4

EXAMPLE DROP RULE:
""" + '{"dpid": 1, "table_id": 0, "priority": 1000, "match": {"eth_type": 2048, "ipv4_src": "10.0.0.1", "ipv4_dst": "10.0.0.2"}, "actions": [{"type": "DROP"}], "idle_timeout": 0, "hard_timeout": 0}' + """

FORMAT: Return ONLY a JSON array of flow rules, no other text.

RESPOND WITH ONLY THE JSON ARRAY:
"""
    
    def _generate_routing_prompt(self, entities, topology_str) -> str:
        """Generate enhanced prompt for routing intents with detailed topology analysis."""
        source = entities.get("source")
        destination = entities.get("destination")
        if not source or not destination:
            source = source or "hX"
            destination = destination or "hY"
        optimization = entities.get("optimization", "shortest_path")
        
        return f"""
You are an expert OpenFlow engineer and network path calculation specialist. Generate OpenFlow rules for ROUTING optimization.

TASK: Route traffic from {source} to {destination} using {optimization}

NETWORK TOPOLOGY:
{topology_str}

PATH CALCULATION INSTRUCTIONS:
1. **ANALYZE TOPOLOGY**: First understand the complete network topology including all switches, hosts, and inter-switch links
2. **IDENTIFY HOST LOCATIONS**: Find which switch each host connects to based on the topology information
3. **CHECK SAME-SWITCH CASE**: If both {source} and {destination} connect to the SAME switch, generate rules ONLY for that switch using host ports
4. **CALCULATE SHORTEST PATH**: For different switches, use Dijkstra's algorithm to find the shortest path between source and destination switches
5. **CONSIDER ALTERNATIVES**: If multiple paths exist, choose the one with fewer hops or better performance
6. **VALIDATE PORTS**: Ensure the output ports in your rules match the actual inter-switch connections

REQUIREMENTS:
1. **CRITICAL**: Only use valid switch DPIDs: 1, 2, 3, 4 (NEVER use 5, 6, 7, 8, 9, 10)
2. **SAME-SWITCH DETECTION**: If {source} and {destination} are on the same switch, only generate rules for that switch
3. **PATH ANALYSIS**: For different switches, calculate the exact path from {source} to {destination} through the network
4. **BIDIRECTIONAL RULES**: Generate rules for both directions (source->destination AND destination->source)
5. **PORT MAPPING**: Use correct output ports based on the topology links
6. **PRIORITY**: Use priority 800+ to override default rules
7. **IP MATCHING**: Use exact IP addresses for source and destination hosts

SAME-SWITCH EXAMPLE:
If h1 and h5 are both on switch s1 (DPID: 1):
- h1 connects to port 1, h5 connects to port 2
- Generate rules ONLY for switch 1
- Forward: h1->h5 uses port 2, Reverse: h5->h1 uses port 1
- DO NOT generate rules for other switches

MULTI-HOP EXAMPLE:
For h1->h3 routing (different switches):
- h1 connects to switch s1 (DPID: 1)
- h3 connects to switch s3 (DPID: 3)  
- Path: s1 -> s2 -> s3 (via inter-switch links)
- Generate rules for each switch along the path

EXAMPLE SAME-SWITCH RULES (h1->h5 on s1):
[
  {{"dpid": 1, "table_id": 0, "priority": 800, "match": {{"eth_type": 2048, "ipv4_src": "10.0.0.1", "ipv4_dst": "10.0.0.5"}}, "actions": [{{"type": "OUTPUT", "port": 2}}], "idle_timeout": 300, "hard_timeout": 0}},
  {{"dpid": 1, "table_id": 0, "priority": 800, "match": {{"eth_type": 2048, "ipv4_src": "10.0.0.5", "ipv4_dst": "10.0.0.1"}}, "actions": [{{"type": "OUTPUT", "port": 1}}], "idle_timeout": 300, "hard_timeout": 0}}
]

EXAMPLE MULTI-HOP RULES (h1->h3):
[
  {{"dpid": 1, "table_id": 0, "priority": 800, "match": {{"eth_type": 2048, "ipv4_src": "10.0.0.1", "ipv4_dst": "10.0.0.3"}}, "actions": [{{"type": "OUTPUT", "port": 3}}], "idle_timeout": 300, "hard_timeout": 0}},
  {{"dpid": 2, "table_id": 0, "priority": 800, "match": {{"eth_type": 2048, "ipv4_src": "10.0.0.1", "ipv4_dst": "10.0.0.3"}}, "actions": [{{"type": "OUTPUT", "port": 4}}], "idle_timeout": 300, "hard_timeout": 0}},
  {{"dpid": 3, "table_id": 0, "priority": 800, "match": {{"eth_type": 2048, "ipv4_src": "10.0.0.1", "ipv4_dst": "10.0.0.3"}}, "actions": [{{"type": "OUTPUT", "port": 1}}], "idle_timeout": 300, "hard_timeout": 0}}
]

CRITICAL: For same-switch communication, generate rules ONLY for that switch. Do not include rules for other switches.

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
1. **CRITICAL**: Only use valid switch DPIDs: 1, 2, 3, 4 (NEVER use 5, 6, 7, 8, 9, 10)
2. Create rules to identify {traffic_type} traffic on TCP port {port}
3. Set priority {priority_value} for {priority} priority traffic
4. Use queue actions or DSCP marking if available
5. Install on all switches to ensure QoS throughout network
6. **CRITICAL**: Match only TCP destination port {port}, NOT source port (clients use random source ports)

EXAMPLE QoS RULE:
""" + '{"dpid": 1, "table_id": 0, "priority": 500, "match": {"eth_type": 2048, "ip_proto": 6, "tcp_dst": 80}, "actions": [{"type": "SET_QUEUE", "queue_id": 1}, {"type": "OUTPUT", "port": 1}], "idle_timeout": 0, "hard_timeout": 0}' + """

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
1. **CRITICAL**: Only use valid switch DPIDs: 1, 2, 3, 4 (NEVER use 5, 6, 7, 8, 9, 10)
2. Create monitoring rules to track {metric} usage
3. Use low priority (100) to capture all traffic
4. Forward to controller for statistics collection
5. Install on all switches for comprehensive monitoring
6. Use table-miss rules to catch all unmatched traffic

EXAMPLE MONITORING RULE:
""" + '{"dpid": 1, "table_id": 0, "priority": 100, "match": {}, "actions": [{"type": "OUTPUT", "port": "CONTROLLER"}, {"type": "OUTPUT", "port": "NORMAL"}], "idle_timeout": 0, "hard_timeout": 0}' + """

FORMAT: Return ONLY a JSON array of flow rules, no other text.

RESPOND WITH ONLY THE JSON ARRAY:
"""
    
    def _format_topology_for_llm(self, topology: Any, source: str = None, destination: str = None) -> str:
        """Format network topology information for LLM consumption."""
        # Initialize host mappings
        host_mappings = {}
        host_ip_mappings = {}
        
        # Process devices to build host mappings (deterministic, name-based only)
        devices_info = []
        for device in topology.devices:
            if device.device_type.value == "switch":
                device_info = f"Switch {device.name} (DPID: {device.dpid}) with {len(device.ports)} ports"
                
                # Process each port to identify host connections
                for port in device.ports:
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
                                    host_name = None
                                    if switch_name == "s1" and host_num == "1":  # s1 port 1 -> h1
                                        is_host_port = True
                                        host_name = "h1"
                                    elif switch_name == "s1" and host_num == "2":  # s1 port 2 -> h5
                                        is_host_port = True
                                        host_name = "h5"
                                    elif switch_name == "s2" and host_num == "1":  # s2 port 1 -> h2
                                        is_host_port = True
                                        host_name = "h2"
                                    elif switch_name == "s2" and host_num == "2":  # s2 port 2 -> h4
                                        is_host_port = True
                                        host_name = "h4"
                                    elif switch_name == "s3" and host_num == "1":  # s3 port 1 -> h3
                                        is_host_port = True
                                        host_name = "h3"
                                    elif switch_name == "s4" and host_num == "1":  # s4 port 1 -> h6
                                        is_host_port = True
                                        host_name = "h6"
                                    
                                    if is_host_port and host_name:
                                        
                                        host_ip = f"10.0.0.{host_name.replace('h', '')}"  # Extract number from h1, h2, etc.
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
                    
                    # Do not infer hosts from numeric port numbers to avoid false positives
                    
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
        
        # Build host-to-switch connections (for prompt context only)
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
                    
                    # Skip numeric port-to-host inference
        
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

SAME-SWITCH HOSTS:
{chr(10).join([f"- Switch {device.name} (DPID: {device.dpid}) has hosts: {', '.join([host for host, dpid in host_mappings.items() if dpid == device.dpid])}" for device in topology.devices if device.device_type.value == "switch" and len([host for host, dpid in host_mappings.items() if dpid == device.dpid]) >= 2]) if any(len([host for host, dpid in host_mappings.items() if dpid == device.dpid]) >= 2 for device in topology.devices if device.device_type.value == "switch") else "None"}
"""

        # Add routing-specific information if source and destination are provided
        if source and destination:
            topology_str += f"""

FOR ROUTING {source}<->{destination}:
- {source} (10.0.0.{source[1:]}) connects to Switch {host_mappings.get(source, 'unknown')} (DPID: {host_mappings.get(source, 'unknown')})
- {destination} (10.0.0.{destination[1:]}) connects to Switch {host_mappings.get(destination, 'unknown')} (DPID: {host_mappings.get(destination, 'unknown')})
- If same switch: Generate rules ONLY for that switch
- If different switches: Calculate shortest path and generate rules for each switch along the path
"""
        
        logger.info(f"Formatted topology for LLM with {len(host_mappings)} hosts and {len(links_info)} links")
        return topology_str

    def _infer_hosts_from_text(self, intent_text: str):
        """Infer source/destination hosts like h1/h2 from free text."""
        try:
            import re
            text = intent_text.lower()
            # common patterns: "h1 to h3", "h2 reach h4", "allow h5 -> h6"
            m = re.search(r"h(\d+)\s*(?:->|to|reach|communicat[e|ion]*\s*with)\s*h(\d+)", text)
            if m:
                return f"h{m.group(1)}", f"h{m.group(2)}"
            # fallback: first two host tokens in text
            hosts = re.findall(r"h(\d+)", text)
            if len(hosts) >= 2:
                return f"h{hosts[0]}", f"h{hosts[1]}"
        except Exception:
            pass
        return None, None