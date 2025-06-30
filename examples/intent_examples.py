#!/usr/bin/env python3
"""
Intent examples for the LLM Intent-based SDN system.
This script demonstrates various types of network intents and how to process them.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import httpx
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_intent_sdn.config import settings


class IntentExamples:
    """Examples of different types of network intents."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or f"http://{settings.api_host}:{settings.api_port}"
        
    async def process_intent(self, intent_text: str, context: str = "") -> Dict[str, Any]:
        """Process a single intent."""
        intent_data = {
            "text": intent_text,
            "context": context
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/intent/process",
                    json=intent_data
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Failed to process intent: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error processing intent: {e}")
            return {}
    
    async def security_policy_examples(self):
        """Examples of security policy intents."""
        logger.info("🔒 Security Policy Examples")
        logger.info("=" * 40)
        
        security_intents = [
            {
                "text": "Block all traffic from host 10.0.0.1 to host 10.0.0.2",
                "context": "Security incident response - compromised host isolation"
            },
            {
                "text": "Allow only HTTP and HTTPS traffic between web servers and clients",
                "context": "Web application security policy"
            },
            {
                "text": "Deny SSH access from external networks to internal servers",
                "context": "Remote access security hardening"
            },
            {
                "text": "Create firewall rule to block all traffic on port 23 (Telnet)",
                "context": "Legacy protocol security removal"
            },
            {
                "text": "Isolate host 10.0.0.3 from the network due to malware detection",
                "context": "Automated incident response"
            }
        ]
        
        for i, intent in enumerate(security_intents, 1):
            logger.info(f"\n📝 Security Example {i}:")
            logger.info(f"Intent: {intent['text']}")
            logger.info(f"Context: {intent['context']}")
            
            result = await self.process_intent(intent['text'], intent['context'])
            if result:
                logger.success(f"✅ Type: {result.get('intent_type', 'Unknown')}")
                logger.info(f"📊 Analysis: {result.get('analysis', 'No analysis')[:100]}...")
                if result.get('flow_rules'):
                    logger.info(f"🔧 Generated {len(result['flow_rules'])} flow rules")
            else:
                logger.error("❌ Failed to process intent")
            
            await asyncio.sleep(1)  # Rate limiting
    
    async def traffic_engineering_examples(self):
        """Examples of traffic engineering intents."""
        logger.info("\n🚦 Traffic Engineering Examples")
        logger.info("=" * 40)
        
        traffic_intents = [
            {
                "text": "Route traffic from 10.0.0.1 to 10.0.0.4 through the fastest path",
                "context": "Performance optimization for critical application"
            },
            {
                "text": "Balance load between two paths for traffic to subnet 10.0.1.0/24",
                "context": "Load balancing for high-traffic network segment"
            },
            {
                "text": "Set bandwidth limit of 10 Mbps for traffic from host 10.0.0.2",
                "context": "QoS policy for guest network users"
            },
            {
                "text": "Prioritize video streaming traffic over file downloads",
                "context": "QoS optimization for multimedia applications"
            },
            {
                "text": "Route backup traffic through secondary links during maintenance",
                "context": "Planned maintenance traffic rerouting"
            }
        ]
        
        for i, intent in enumerate(traffic_intents, 1):
            logger.info(f"\n📝 Traffic Example {i}:")
            logger.info(f"Intent: {intent['text']}")
            logger.info(f"Context: {intent['context']}")
            
            result = await self.process_intent(intent['text'], intent['context'])
            if result:
                logger.success(f"✅ Type: {result.get('intent_type', 'Unknown')}")
                logger.info(f"📊 Analysis: {result.get('analysis', 'No analysis')[:100]}...")
                if result.get('flow_rules'):
                    logger.info(f"🔧 Generated {len(result['flow_rules'])} flow rules")
            else:
                logger.error("❌ Failed to process intent")
            
            await asyncio.sleep(1)
    
    async def monitoring_examples(self):
        """Examples of monitoring and alerting intents."""
        logger.info("\n📊 Monitoring Examples")
        logger.info("=" * 40)
        
        monitoring_intents = [
            {
                "text": "Monitor bandwidth usage on link between switch s1 and s2",
                "context": "Capacity planning for network upgrade"
            },
            {
                "text": "Alert when packet loss exceeds 1% on any link",
                "context": "Network performance monitoring"
            },
            {
                "text": "Track top 5 applications by bandwidth consumption",
                "context": "Application performance monitoring"
            },
            {
                "text": "Monitor for DDoS attack patterns and auto-block suspicious traffic",
                "context": "Automated security monitoring"
            },
            {
                "text": "Generate report of network utilization for the past hour",
                "context": "Network operations reporting"
            }
        ]
        
        for i, intent in enumerate(monitoring_intents, 1):
            logger.info(f"\n📝 Monitoring Example {i}:")
            logger.info(f"Intent: {intent['text']}")
            logger.info(f"Context: {intent['context']}")
            
            result = await self.process_intent(intent['text'], intent['context'])
            if result:
                logger.success(f"✅ Type: {result.get('intent_type', 'Unknown')}")
                logger.info(f"📊 Analysis: {result.get('analysis', 'No analysis')[:100]}...")
            else:
                logger.error("❌ Failed to process intent")
            
            await asyncio.sleep(1)
    
    async def configuration_examples(self):
        """Examples of network configuration intents."""
        logger.info("\n⚙️ Configuration Examples")
        logger.info("=" * 40)
        
        config_intents = [
            {
                "text": "Add new VLAN 100 for guest network access",
                "context": "Network segmentation for guest users"
            },
            {
                "text": "Configure port mirroring on switch s1 for traffic analysis",
                "context": "Network troubleshooting and monitoring setup"
            },
            {
                "text": "Set up redundant paths between data center switches",
                "context": "High availability network design"
            },
            {
                "text": "Enable spanning tree protocol on all switches",
                "context": "Loop prevention in redundant topology"
            },
            {
                "text": "Configure QoS classes for voice, data, and video traffic",
                "context": "Multimedia network optimization"
            }
        ]
        
        for i, intent in enumerate(config_intents, 1):
            logger.info(f"\n📝 Configuration Example {i}:")
            logger.info(f"Intent: {intent['text']}")
            logger.info(f"Context: {intent['context']}")
            
            result = await self.process_intent(intent['text'], intent['context'])
            if result:
                logger.success(f"✅ Type: {result.get('intent_type', 'Unknown')}")
                logger.info(f"📊 Analysis: {result.get('analysis', 'No analysis')[:100]}...")
            else:
                logger.error("❌ Failed to process intent")
            
            await asyncio.sleep(1)
    
    async def troubleshooting_examples(self):
        """Examples of network troubleshooting intents."""
        logger.info("\n🔧 Troubleshooting Examples")
        logger.info("=" * 40)
        
        troubleshooting_intents = [
            {
                "text": "Diagnose connectivity issues between host 10.0.0.1 and 10.0.0.3",
                "context": "User reported application connectivity problems"
            },
            {
                "text": "Find the root cause of high latency in the network",
                "context": "Performance degradation investigation"
            },
            {
                "text": "Identify bottlenecks causing packet drops on switch s2",
                "context": "Network performance troubleshooting"
            },
            {
                "text": "Check if there are any loops in the network topology",
                "context": "Network stability verification"
            },
            {
                "text": "Analyze traffic patterns to find unusual behavior",
                "context": "Security anomaly detection"
            }
        ]
        
        for i, intent in enumerate(troubleshooting_intents, 1):
            logger.info(f"\n📝 Troubleshooting Example {i}:")
            logger.info(f"Intent: {intent['text']}")
            logger.info(f"Context: {intent['context']}")
            
            result = await self.process_intent(intent['text'], intent['context'])
            if result:
                logger.success(f"✅ Type: {result.get('intent_type', 'Unknown')}")
                logger.info(f"📊 Analysis: {result.get('analysis', 'No analysis')[:100]}...")
            else:
                logger.error("❌ Failed to process intent")
            
            await asyncio.sleep(1)
    
    async def run_all_examples(self):
        """Run all intent examples."""
        logger.info("🚀 Running Intent Processing Examples")
        logger.info("=" * 50)
        logger.info("This demonstrates various types of network intents that the LLM system can process.")
        logger.info("Note: Some intents may require actual network topology to generate flow rules.\n")
        
        # Run different categories of examples
        await self.security_policy_examples()
        await self.traffic_engineering_examples()
        await self.monitoring_examples()
        await self.configuration_examples()
        await self.troubleshooting_examples()
        
        logger.info("\n" + "=" * 50)
        logger.success("🎉 All intent examples completed!")
        logger.info("💡 Try creating your own intents using natural language!")


async def interactive_mode():
    """Interactive mode for testing custom intents."""
    logger.info("🎯 Interactive Intent Testing Mode")
    logger.info("=" * 40)
    logger.info("Enter your network intents in natural language.")
    logger.info("Type 'quit' or 'exit' to stop.\n")
    
    examples = IntentExamples()
    
    while True:
        try:
            intent_text = input("🔤 Enter intent: ").strip()
            
            if intent_text.lower() in ['quit', 'exit', 'q']:
                logger.info("👋 Goodbye!")
                break
            
            if not intent_text:
                continue
            
            context = input("📝 Enter context (optional): ").strip()
            
            logger.info("\n🔄 Processing intent...")
            result = await examples.process_intent(intent_text, context)
            
            if result:
                logger.success(f"✅ Intent Type: {result.get('intent_type', 'Unknown')}")
                logger.info(f"📊 Analysis: {result.get('analysis', 'No analysis')}")
                
                if result.get('flow_rules'):
                    logger.info(f"🔧 Generated {len(result['flow_rules'])} flow rules:")
                    for i, rule in enumerate(result['flow_rules'][:3], 1):  # Show first 3 rules
                        logger.info(f"   Rule {i}: {rule.get('match', {})} -> {rule.get('actions', [])}")
                
                if result.get('status') == 'completed':
                    logger.success("🎉 Intent processed successfully!")
                else:
                    logger.warning(f"⚠️ Status: {result.get('status', 'Unknown')}")
            else:
                logger.error("❌ Failed to process intent")
            
            print("\n" + "-" * 40 + "\n")
            
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")


async def main():
    """Main function."""
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", 
              format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_mode()
    else:
        # Run predefined examples
        examples = IntentExamples()
        await examples.run_all_examples()


if __name__ == "__main__":
    asyncio.run(main()) 