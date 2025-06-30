"""LLM service for processing network intents."""

import json
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
Always respond with a JSON object containing:
{
    "intent_type": "routing|qos|security|load_balancing|anomaly_detection|traffic_engineering",
    "confidence": 0.0-1.0,
    "extracted_entities": {
        "source": "source host/switch",
        "destination": "destination host/switch", 
        "parameters": "additional parameters"
    },
    "suggested_actions": ["list", "of", "actions"],
    "reasoning": "Your analysis and reasoning process",
    "flow_rules": [
        {
            "dpid": 1,
            "table_id": 0,
            "priority": 100,
            "match": {"field": "value"},
            "actions": [{"type": "OUTPUT", "port": 2}]
        }
    ]
}

GUIDELINES:
- Always consider network state and topology
- Prioritize network stability and performance
- Provide detailed reasoning for your decisions
- Include specific flow rules when applicable
- Handle errors and edge cases gracefully
"""

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
                response_data = json.loads(llm_response.content)
                
                # Create IntentAnalysis object
                analysis = IntentAnalysis(
                    intent_type=response_data.get("intent_type", "unknown"),
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
                # Return fallback analysis
                return IntentAnalysis(
                    intent_type="unknown",
                    confidence=0.1,
                    extracted_entities={},
                    suggested_actions=[],
                    reasoning=f"Failed to parse LLM response: {llm_response.content}"
                )
                
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            return IntentAnalysis(
                intent_type="error",
                confidence=0.0,
                extracted_entities={},
                suggested_actions=[],
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
            # Create a focused prompt for flow rule generation
            prompt = f"""
Based on the following intent analysis and network topology, generate specific OpenFlow rules:

INTENT ANALYSIS:
Type: {intent_analysis.intent_type}
Entities: {json.dumps(intent_analysis.extracted_entities, indent=2)}
Actions: {intent_analysis.suggested_actions}

NETWORK TOPOLOGY:
{self._format_topology_for_llm(network_topology)}

Generate OpenFlow rules in this exact JSON format:
[
    {{
        "dpid": 1,
        "table_id": 0,
        "priority": 100,
        "match": {{"in_port": 1, "eth_type": 2048}},
        "actions": [{{"type": "OUTPUT", "port": 2}}],
        "idle_timeout": 0,
        "hard_timeout": 0
    }}
]
"""
            
            messages = [
                LLMMessage(role="system", content="You are an OpenFlow expert. Generate only valid JSON flow rules."),
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
                flow_rules = json.loads(response.content)
                if isinstance(flow_rules, list):
                    return flow_rules
                else:
                    logger.warning("LLM returned non-list flow rules")
                    return []
            except json.JSONDecodeError:
                logger.error("Failed to parse flow rules JSON")
                return []
                
        except Exception as e:
            logger.error(f"Error generating flow rules: {e}")
            return []
    
    def _format_topology_for_llm(self, topology: NetworkTopology) -> str:
        """Format network topology for LLM consumption."""
        devices_info = []
        for device in topology.devices:
            device_info = f"- {device.name} (DPID: {device.dpid}, Type: {device.device_type})"
            if device.ports:
                ports = [f"Port {p.port_no}" for p in device.ports]
                device_info += f" Ports: {', '.join(ports)}"
            devices_info.append(device_info)
        
        links_info = []
        for link in topology.links:
            links_info.append(
                f"- Switch {link.src_dpid}:{link.src_port_no} <-> Switch {link.dst_dpid}:{link.dst_port_no}"
            )
        
        return f"""
DEVICES:
{chr(10).join(devices_info)}

LINKS:
{chr(10).join(links_info)}
""" 