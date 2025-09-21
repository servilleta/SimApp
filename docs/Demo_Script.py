#!/usr/bin/env python3
"""
Monte Carlo API Demo Script
Simple demonstration for potential customers
"""

import requests
import json
import time

# Your API configuration
API_KEY = "ak_4e968d72ca45909d97624140f9ba5d4a_sk_a73a4b3849a834e6bae20c23cccb074ef0ad2af04cc4cd155841484f52120323"  # Demo/test key
BASE_URL = "http://209.51.170.185:8000/simapp-api"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def demo_health_check():
    """Demo: Test API connectivity"""
    print("🔍 Testing API Connection...")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API Status: {data['status']}")
        print(f"🎮 GPU Available: {data['gpu_available']}")
        print(f"⚡ Version: {data['version']}")
        return True
    else:
        print(f"❌ API Connection Failed: {response.status_code}")
        return False

def demo_list_models():
    """Demo: List uploaded models"""
    print("\n📁 Checking Available Models...")
    
    response = requests.get(f"{BASE_URL}/models", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"📊 Found {len(data['models'])} models")
        
        for model in data['models']:
            print(f"  - {model['filename']} ({model['formulas_count']} formulas)")
            
        return data['models']
    else:
        print(f"❌ Failed to list models: {response.status_code}")
        return []

def demo_simulation_example():
    """Demo: Show what a simulation request looks like"""
    print("\n🎲 Example Monte Carlo Simulation Request:")
    
    example_request = {
        "model_id": "mdl_example123456",
        "simulation_config": {
            "iterations": 100000,
            "variables": [
                {
                    "cell": "B5",
                    "name": "Market_Volatility",
                    "distribution": {
                        "type": "triangular",
                        "min": 0.05,     # 5% minimum
                        "mode": 0.15,    # 15% expected
                        "max": 0.35      # 35% maximum
                    }
                },
                {
                    "cell": "C7", 
                    "name": "Interest_Rate",
                    "distribution": {
                        "type": "normal",
                        "mean": 0.03,    # 3% average
                        "std": 0.01      # 1% std deviation
                    }
                }
            ],
            "output_cells": ["J25", "K25"],  # Results we want to analyze
            "confidence_levels": [0.95, 0.99],
            "webhook_url": "https://customer-app.com/simulation-complete"
        }
    }
    
    print(json.dumps(example_request, indent=2))
    
    print("\n📈 This would return:")
    print("  - Statistical analysis (mean, std, percentiles)")
    print("  - Value at Risk (VaR) calculations")
    print("  - Distribution histograms")
    print("  - Correlation analysis")

def demo_results_example():
    """Demo: Show what results look like"""
    print("\n📊 Example Results Format:")
    
    example_results = {
        "simulation_id": "sim_abc123",
        "status": "completed",
        "execution_time": "42.7 seconds",
        "iterations_completed": 100000,
        "results": {
            "J25": {
                "cell_name": "Portfolio_NPV",
                "statistics": {
                    "mean": 1250000,
                    "std": 340000,
                    "min": 420000,
                    "max": 2180000,
                    "percentiles": {
                        "5": 680000,    # 5% chance of being below this
                        "25": 1020000,
                        "50": 1240000,  # Median
                        "75": 1480000,
                        "95": 1820000   # 95% confidence upper bound
                    },
                    "var_95": 680000,  # Value at Risk (95%)
                    "var_99": 540000   # Value at Risk (99%)
                }
            }
        }
    }
    
    print(json.dumps(example_results, indent=2))

def main():
    """Run the complete demo"""
    print("🚀 Monte Carlo Simulation API Demo")
    print("=" * 50)
    
    # Test connectivity
    if not demo_health_check():
        return
    
    # Show available models
    demo_list_models()
    
    # Show how to request simulation
    demo_simulation_example()
    
    # Show results format
    demo_results_example()
    
    print("\n" + "=" * 50)
    print("🎯 Demo Complete!")
    print("\n📋 Next Steps:")
    print("  1. Get your API key: contact sales team")
    print("  2. Upload your Excel model via API")
    print("  3. Run your first Monte Carlo simulation")
    print("  4. Integrate results into your application")
    
    print(f"\n📚 Documentation: https://docs.your-api.com")
    print(f"💬 Support: api-support@your-company.com")
    print(f"🎮 Try it live: {BASE_URL}/health")

if __name__ == "__main__":
    main()
