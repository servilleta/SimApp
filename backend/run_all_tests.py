#!/usr/bin/env python3
"""
Master Test Runner for Monte Carlo Simulation Platform
Runs all validation test suites and generates comprehensive report
"""

import os
import sys
import json
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path


class TestRunner:
    """Runs all test suites and collects results"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
        
    async def run_all_tests(self):
        """Execute all test suites"""
        print("\n" + "="*80)
        print("🚀 MONTE CARLO SIMULATION - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Define test suites
        test_suites = [
            {
                "name": "Unit Tests - Monte Carlo Validation",
                "file": "test_monte_carlo_validation.py",
                "async": True
            },
            {
                "name": "Integration Tests - Simulation Service",
                "file": "test_simulation_integration.py",
                "async": True
            },
            {
                "name": "End-to-End Tests - API Testing",
                "file": "test_monte_carlo_e2e.py",
                "async": True
            },
            {
                "name": "Statistical Validation Tests",
                "file": "test_statistical_validation.py",
                "async": False
            }
        ]
        
        total_passed = 0
        total_failed = 0
        
        for suite in test_suites:
            print(f"\n{'='*60}")
            print(f"🧪 Running: {suite['name']}")
            print(f"{'='*60}")
            
            result = await self.run_test_suite(suite['file'], suite['async'])
            self.results[suite['name']] = result
            
            if result['status'] == 'passed':
                total_passed += 1
                print(f"\n✅ {suite['name']} - PASSED")
            else:
                total_failed += 1
                print(f"\n❌ {suite['name']} - FAILED")
                
        # Generate final report
        self.generate_report(total_passed, total_failed)
        
        return total_failed == 0
        
    async def run_test_suite(self, test_file, is_async):
        """Run a single test suite"""
        test_path = Path(test_file)
        
        if not test_path.exists():
            return {
                "status": "failed",
                "error": f"Test file not found: {test_file}",
                "duration": 0
            }
            
        start = datetime.now()
        
        try:
            # Run the test
            cmd = [sys.executable, str(test_path)]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            duration = (datetime.now() - start).total_seconds()
            
            # Parse output
            output = stdout.decode('utf-8')
            print(output)  # Show test output
            
            if stderr:
                print(f"STDERR: {stderr.decode('utf-8')}")
                
            return {
                "status": "passed" if process.returncode == 0 else "failed",
                "duration": duration,
                "return_code": process.returncode,
                "output_lines": len(output.splitlines())
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "duration": (datetime.now() - start).total_seconds()
            }
            
    def generate_report(self, passed, failed):
        """Generate comprehensive test report"""
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"\n📈 Overall Statistics:")
        print(f"  • Total Test Suites: {passed + failed}")
        print(f"  • Passed: {passed}")
        print(f"  • Failed: {failed}")
        print(f"  • Success Rate: {(passed/(passed+failed)*100):.1f}%")
        print(f"  • Total Duration: {total_duration:.2f} seconds")
        
        print(f"\n📋 Individual Results:")
        for suite_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'passed' else "❌"
            print(f"  {status_icon} {suite_name}")
            print(f"     Duration: {result.get('duration', 0):.2f}s")
            if result.get('error'):
                print(f"     Error: {result['error']}")
                
        # Save detailed report
        report_data = {
            "test_run": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": total_duration
            },
            "summary": {
                "total_suites": passed + failed,
                "passed": passed,
                "failed": failed,
                "success_rate": (passed/(passed+failed)*100) if (passed+failed) > 0 else 0
            },
            "results": self.results
        }
        
        report_file = f"monte_carlo_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Generate recommendations
        print("\n💡 Recommendations:")
        if failed == 0:
            print("  ✅ All tests passed! The Monte Carlo simulation platform is working correctly.")
            print("  ✅ The FULL_EVALUATION approach is processing all formulas as expected.")
            print("  ✅ Statistical results are mathematically sound.")
        else:
            print("  ⚠️  Some tests failed. Please review the failures above.")
            print("  📝 Check the backend logs for detailed error information.")
            print("  🔧 Ensure Docker containers are running: docker-compose up -d")
            

async def check_prerequisites():
    """Check that all prerequisites are met"""
    print("\n🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
        
    # Check required packages
    required_packages = ['numpy', 'scipy', 'aiohttp', 'asyncio']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
            
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print(f"   Install with: pip install {' '.join(missing_packages)}")
        return False
        
    print("✅ All prerequisites met")
    return True


async def main():
    """Main entry point"""
    print("\n🎯 Monte Carlo Simulation Platform - Test Validation Suite")
    print("📍 This will validate that the Monte Carlo simulation is working correctly")
    
    # Check prerequisites
    if not await check_prerequisites():
        return 1
        
    # Check if backend is running (optional)
    print("\n🔍 Checking backend status...")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health", timeout=5) as resp:
                if resp.status == 200:
                    print("✅ Backend is running")
                else:
                    print("⚠️  Backend returned status:", resp.status)
    except Exception as e:
        print("⚠️  Cannot connect to backend - E2E tests may fail")
        print("   Ensure backend is running: docker-compose up -d")
        
    # Run all tests
    runner = TestRunner()
    success = await runner.run_all_tests()
    
    # Final summary
    print("\n" + "="*80)
    if success:
        print("✅ ALL TESTS PASSED - Monte Carlo simulation is working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Please review the errors above")
    print("="*80)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 