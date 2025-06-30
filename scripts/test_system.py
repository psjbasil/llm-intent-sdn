#!/usr/bin/env python3
"""
Test script for the LLM Intent-based SDN system.
This script tests various components and API endpoints to ensure the system is working correctly.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import httpx
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_intent_sdn.config import settings


class SystemTester:
    """System testing utility."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or f"http://{settings.api_host}:{settings.api_port}"
        self.ryu_url = f"http://{settings.ryu_host}:{settings.ryu_port}"
        
    async def test_api_health(self):
        """Test API health endpoint."""
        logger.info("Testing API health...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/health")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.success(f"API Health: {data['status']}")
                    return True
                else:
                    logger.error(f"API Health check failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"API Health check error: {e}")
            return False
    
    async def test_ryu_connectivity(self):
        """Test RYU controller connectivity."""
        logger.info("Testing RYU controller connectivity...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ryu_url}/stats/switches")
                
                if response.status_code == 200:
                    switches = response.json()
                    logger.success(f"RYU Connected: {len(switches)} switches found")
                    return True
                else:
                    logger.error(f"RYU connection failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"RYU connection error: {e}")
            return False
    
    async def test_intent_processing(self):
        """Test intent processing endpoint."""
        logger.info("Testing intent processing...")
        
        test_intent = {
            "text": "Block all traffic from host 10.0.0.1 to host 10.0.0.2",
            "context": "Security policy enforcement"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/intent/process",
                    json=test_intent
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.success(f"Intent processed: {data['intent_type']}")
                    logger.info(f"Analysis: {data['analysis'][:100]}...")
                    return True
                else:
                    logger.error(f"Intent processing failed: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Intent processing error: {e}")
            return False
    
    async def test_network_topology(self):
        """Test network topology endpoint."""
        logger.info("Testing network topology...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/network/topology")
                
                if response.status_code == 200:
                    data = response.json()
                    switches = data.get('switches', [])
                    hosts = data.get('hosts', [])
                    logger.success(f"Topology: {len(switches)} switches, {len(hosts)} hosts")
                    return True
                else:
                    logger.error(f"Topology fetch failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Topology fetch error: {e}")
            return False
    
    async def test_flow_rules(self):
        """Test flow rules endpoint."""
        logger.info("Testing flow rules...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/network/flows")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.success(f"Flow rules retrieved: {len(data)} rules")
                    return True
                else:
                    logger.error(f"Flow rules fetch failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Flow rules fetch error: {e}")
            return False
    
    async def test_monitoring(self):
        """Test monitoring endpoints."""
        logger.info("Testing monitoring...")
        
        endpoints = [
            "/api/v1/monitoring/anomalies",
            "/api/v1/monitoring/health",
            "/api/v1/monitoring/stats"
        ]
        
        results = []
        async with httpx.AsyncClient() as client:
            for endpoint in endpoints:
                try:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    if response.status_code == 200:
                        logger.success(f"{endpoint} - OK")
                        results.append(True)
                    else:
                        logger.error(f"{endpoint} - Failed ({response.status_code})")
                        results.append(False)
                except Exception as e:
                    logger.error(f"{endpoint} - Error: {e}")
                    results.append(False)
        
        return all(results)
    
    async def test_api_documentation(self):
        """Test API documentation endpoint."""
        logger.info("Testing API documentation...")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/docs")
                
                if response.status_code == 200:
                    logger.success("API documentation available")
                    return True
                else:
                    logger.error(f"API documentation failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"API documentation error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests."""
        logger.info(" Starting system tests...")
        logger.info("=" * 50)
        
        tests = [
            ("API Health", self.test_api_health),
            ("RYU Connectivity", self.test_ryu_connectivity),
            ("Network Topology", self.test_network_topology),
            ("Flow Rules", self.test_flow_rules),
            ("Monitoring", self.test_monitoring),
            ("API Documentation", self.test_api_documentation),
            ("Intent Processing", self.test_intent_processing),
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name} test...")
            try:
                result = await test_func()
                results[test_name] = result
            except Exception as e:
                logger.error(f"{test_name} test failed with exception: {e}")
                results[test_name] = False
            
            # Wait between tests
            time.sleep(1)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("Test Results Summary:")
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"  {test_name}: {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.success("All tests passed! System is working correctly.")
            return True
        else:
            logger.error("Some tests failed. Please check the system configuration.")
            return False


async def main():
    """Main test function."""
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    
    # Create tester
    tester = SystemTester()
    
    # Run tests
    success = await tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 