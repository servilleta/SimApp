#!/usr/bin/env python3
"""
Load Testing Plan for Multi-Instance Production Deployment
Monte Carlo Platform - Performance Validation
"""

import asyncio
import aiohttp
import json
import time
import random
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("load_testing")

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios"""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int
    target_rps: int  # requests per second
    endpoints: List[str]
    test_data: Dict[str, Any]

@dataclass
class LoadTestResult:
    """Results from load testing"""
    scenario_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    error_rate: float
    errors: List[str]

class LoadTestClient:
    """Async HTTP client for load testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.response_times = []
        self.errors = []
        self.successful_requests = 0
        self.failed_requests = 0
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=1000, limit_per_host=100)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request and record metrics"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint}"
            async with self.session.request(method, url, **kwargs) as response:
                response_time = time.time() - start_time
                self.response_times.append(response_time)
                
                if response.status < 400:
                    self.successful_requests += 1
                    return {
                        "status": "success",
                        "status_code": response.status,
                        "response_time": response_time,
                        "data": await response.json() if response.content_type == 'application/json' else await response.text()
                    }
                else:
                    self.failed_requests += 1
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    self.errors.append(error_msg)
                    return {
                        "status": "error",
                        "status_code": response.status,
                        "response_time": response_time,
                        "error": error_msg
                    }
        
        except Exception as e:
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            self.failed_requests += 1
            error_msg = f"Request failed: {str(e)}"
            self.errors.append(error_msg)
            return {
                "status": "error",
                "response_time": response_time,
                "error": error_msg
            }

class MonteCarloLoadTester:
    """Load testing specifically for Monte Carlo platform"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_configs = self._create_test_scenarios()
    
    def _create_test_scenarios(self) -> List[LoadTestConfig]:
        """Define load testing scenarios for production validation"""
        
        # Sample Excel file data for testing
        sample_simulation_config = {
            "variables": [
                {
                    "cell": "B5",
                    "name": "Market_Volatility",
                    "distribution": {
                        "type": "triangular",
                        "min": 0.05,
                        "mode": 0.15,
                        "max": 0.35
                    }
                }
            ],
            "output_cells": ["J25"],
            "iterations": 1000,
            "engine": "ultra"
        }
        
        return [
            # Normal load - typical business hours
            LoadTestConfig(
                name="normal_load",
                concurrent_users=500,
                duration_seconds=3600,  # 1 hour
                ramp_up_seconds=300,    # 5 minutes
                target_rps=100,
                endpoints=["/api/health", "/api/simulations", "/api/auth/me"],
                test_data={"simulation_config": sample_simulation_config}
            ),
            
            # Peak load - high traffic periods
            LoadTestConfig(
                name="peak_load",
                concurrent_users=2000,
                duration_seconds=1800,  # 30 minutes
                ramp_up_seconds=600,    # 10 minutes
                target_rps=400,
                endpoints=["/api/health", "/api/simulations", "/api/files/upload"],
                test_data={"simulation_config": sample_simulation_config}
            ),
            
            # Stress test - beyond normal capacity
            LoadTestConfig(
                name="stress_test",
                concurrent_users=5000,
                duration_seconds=900,   # 15 minutes
                ramp_up_seconds=900,    # 15 minutes
                target_rps=1000,
                endpoints=["/api/health", "/api/simulations"],
                test_data={"simulation_config": sample_simulation_config}
            ),
            
            # Endurance test - sustained load
            LoadTestConfig(
                name="endurance_test",
                concurrent_users=1000,
                duration_seconds=86400, # 24 hours
                ramp_up_seconds=1800,   # 30 minutes
                target_rps=200,
                endpoints=["/api/health", "/api/simulations", "/api/auth/me"],
                test_data={"simulation_config": sample_simulation_config}
            ),
            
            # Simulation-focused load test
            LoadTestConfig(
                name="simulation_intensive",
                concurrent_users=200,
                duration_seconds=3600,  # 1 hour
                ramp_up_seconds=300,
                target_rps=50,
                endpoints=["/api/simulations", "/api/simulations/{id}/status", "/api/simulations/{id}/results"],
                test_data={"simulation_config": sample_simulation_config}
            )
        ]
    
    async def simulate_user_session(self, client: LoadTestClient, config: LoadTestConfig, user_id: int):
        """Simulate a realistic user session"""
        session_start = time.time()
        
        # Simulate user login
        auth_result = await client.make_request(
            "POST",
            "/api/auth/login",
            json={
                "email": f"loadtest_user_{user_id}@example.com",
                "password": "loadtest_password"
            }
        )
        
        if auth_result["status"] != "success":
            logger.warning(f"User {user_id} failed to authenticate")
            return
        
        # Extract auth token (mock)
        auth_token = "mock_auth_token"
        headers = {"Authorization": f"Bearer {auth_token}"}
        
        # Simulate user workflow
        while time.time() - session_start < config.duration_seconds:
            # Random endpoint selection based on realistic usage patterns
            endpoint_weights = {
                "/api/health": 0.1,
                "/api/auth/me": 0.1,
                "/api/simulations": 0.4,
                "/api/files/upload": 0.2,
                "/api/simulations/{id}/status": 0.15,
                "/api/simulations/{id}/results": 0.05
            }
            
            endpoint = random.choices(
                list(endpoint_weights.keys()),
                weights=list(endpoint_weights.values())
            )[0]
            
            # Execute request based on endpoint
            if endpoint == "/api/health":
                await client.make_request("GET", endpoint)
            
            elif endpoint == "/api/auth/me":
                await client.make_request("GET", endpoint, headers=headers)
            
            elif endpoint == "/api/simulations":
                await client.make_request(
                    "POST",
                    endpoint,
                    json=config.test_data["simulation_config"],
                    headers=headers
                )
            
            elif endpoint == "/api/files/upload":
                # Simulate file upload with mock data
                fake_excel_data = b"PK\x03\x04" + b"0" * 1000  # Mock Excel file
                await client.make_request(
                    "POST",
                    endpoint,
                    data={"file": fake_excel_data},
                    headers=headers
                )
            
            # Realistic delay between requests
            await asyncio.sleep(random.uniform(1, 5))
    
    async def run_load_test(self, config: LoadTestConfig) -> LoadTestResult:
        """Execute load test scenario"""
        logger.info(f"Starting load test: {config.name}")
        logger.info(f"Users: {config.concurrent_users}, Duration: {config.duration_seconds}s")
        
        start_time = time.time()
        
        async with LoadTestClient(self.base_url) as client:
            # Create tasks for concurrent users
            tasks = []
            
            # Gradual ramp-up
            users_per_second = config.concurrent_users / config.ramp_up_seconds
            
            for i in range(config.concurrent_users):
                # Stagger user starts for realistic ramp-up
                delay = i / users_per_second
                task = asyncio.create_task(
                    self._delayed_user_session(client, config, i, delay)
                )
                tasks.append(task)
            
            # Wait for all users to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate results
            total_time = time.time() - start_time
            total_requests = client.successful_requests + client.failed_requests
            
            if client.response_times:
                response_times_sorted = sorted(client.response_times)
                avg_response_time = statistics.mean(client.response_times)
                p95_response_time = response_times_sorted[int(0.95 * len(response_times_sorted))]
                p99_response_time = response_times_sorted[int(0.99 * len(response_times_sorted))]
                max_response_time = max(client.response_times)
                min_response_time = min(client.response_times)
            else:
                avg_response_time = p95_response_time = p99_response_time = 0
                max_response_time = min_response_time = 0
            
            result = LoadTestResult(
                scenario_name=config.name,
                total_requests=total_requests,
                successful_requests=client.successful_requests,
                failed_requests=client.failed_requests,
                avg_response_time=avg_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                max_response_time=max_response_time,
                min_response_time=min_response_time,
                requests_per_second=total_requests / total_time if total_time > 0 else 0,
                error_rate=(client.failed_requests / total_requests * 100) if total_requests > 0 else 0,
                errors=list(set(client.errors))  # Unique errors
            )
            
            logger.info(f"Load test {config.name} completed:")
            logger.info(f"  Total requests: {result.total_requests}")
            logger.info(f"  Success rate: {(result.successful_requests / result.total_requests * 100):.2f}%")
            logger.info(f"  Avg response time: {result.avg_response_time:.3f}s")
            logger.info(f"  P95 response time: {result.p95_response_time:.3f}s")
            logger.info(f"  Requests/second: {result.requests_per_second:.2f}")
            
            return result
    
    async def _delayed_user_session(self, client: LoadTestClient, config: LoadTestConfig, user_id: int, delay: float):
        """Start user session with delay for gradual ramp-up"""
        await asyncio.sleep(delay)
        await self.simulate_user_session(client, config, user_id)
    
    async def run_all_scenarios(self) -> List[LoadTestResult]:
        """Run all load test scenarios"""
        results = []
        
        for config in self.test_configs:
            try:
                result = await self.run_load_test(config)
                results.append(result)
                
                # Break between tests
                logger.info(f"Waiting 5 minutes before next test...")
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Load test {config.name} failed: {str(e)}")
                # Create error result
                error_result = LoadTestResult(
                    scenario_name=config.name,
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0,
                    avg_response_time=0,
                    p95_response_time=0,
                    p99_response_time=0,
                    max_response_time=0,
                    min_response_time=0,
                    requests_per_second=0,
                    error_rate=100,
                    errors=[str(e)]
                )
                results.append(error_result)
        
        return results
    
    def generate_report(self, results: List[LoadTestResult]) -> Dict[str, Any]:
        """Generate comprehensive load test report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_scenarios": len(results),
                "passed_scenarios": len([r for r in results if r.error_rate < 5]),
                "overall_success": all(r.error_rate < 5 and r.p95_response_time < 2.0 for r in results)
            },
            "scenarios": []
        }
        
        for result in results:
            scenario_report = {
                "name": result.scenario_name,
                "status": "PASS" if result.error_rate < 5 and result.p95_response_time < 2.0 else "FAIL",
                "metrics": {
                    "total_requests": result.total_requests,
                    "success_rate": f"{(result.successful_requests / result.total_requests * 100):.2f}%" if result.total_requests > 0 else "0%",
                    "avg_response_time": f"{result.avg_response_time:.3f}s",
                    "p95_response_time": f"{result.p95_response_time:.3f}s",
                    "p99_response_time": f"{result.p99_response_time:.3f}s",
                    "requests_per_second": f"{result.requests_per_second:.2f}",
                    "error_rate": f"{result.error_rate:.2f}%"
                },
                "sla_compliance": {
                    "response_time_sla": result.p95_response_time < 2.0,  # <2s P95
                    "error_rate_sla": result.error_rate < 5,  # <5% error rate
                    "throughput_sla": result.requests_per_second > 50  # >50 RPS
                }
            }
            
            if result.errors:
                scenario_report["errors"] = result.errors[:10]  # Top 10 errors
            
            report["scenarios"].append(scenario_report)
        
        # Overall recommendations
        report["recommendations"] = []
        
        for result in results:
            if result.p95_response_time > 2.0:
                report["recommendations"].append(f"Optimize response time for {result.scenario_name} (P95: {result.p95_response_time:.3f}s)")
            
            if result.error_rate > 5:
                report["recommendations"].append(f"Investigate errors in {result.scenario_name} (Error rate: {result.error_rate:.2f}%)")
        
        if not report["recommendations"]:
            report["recommendations"].append("All scenarios passed SLA requirements - ready for production!")
        
        return report


async def main():
    """Main function to run load tests"""
    tester = MonteCarloLoadTester()
    
    logger.info("🚀 Starting Monte Carlo Platform Load Testing")
    logger.info("=" * 60)
    
    # Run specific scenario or all scenarios
    import sys
    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        config = next((c for c in tester.test_configs if c.name == scenario_name), None)
        if config:
            result = await tester.run_load_test(config)
            results = [result]
        else:
            logger.error(f"Scenario '{scenario_name}' not found")
            return
    else:
        # Run limited scenarios for initial testing
        quick_scenarios = ["normal_load", "peak_load"]
        results = []
        for scenario_name in quick_scenarios:
            config = next(c for c in tester.test_configs if c.name == scenario_name)
            result = await tester.run_load_test(config)
            results.append(result)
    
    # Generate and save report
    report = tester.generate_report(results)
    
    # Save report to file
    report_filename = f"load_test_report_{int(time.time())}.json"
    with open(report_filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Load test report saved to: {report_filename}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {'✅ PASS' if report['summary']['overall_success'] else '❌ FAIL'}")
    print(f"Scenarios Passed: {report['summary']['passed_scenarios']}/{report['summary']['total_scenarios']}")
    
    for scenario in report["scenarios"]:
        status_emoji = "✅" if scenario["status"] == "PASS" else "❌"
        print(f"{status_emoji} {scenario['name']}: {scenario['metrics']['success_rate']} success, {scenario['metrics']['p95_response_time']} P95")
    
    if report["recommendations"]:
        print("\n🔧 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
