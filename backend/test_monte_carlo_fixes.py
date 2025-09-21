#!/usr/bin/env python3
'''
Test script for Monte Carlo connection fixes
Run this after applying the fixes to verify debugging is working.
'''

import requests
import json
import time

def test_enhanced_debugging():
    '''Test the enhanced debugging in a B12 simulation'''
    
    print("🧪 TESTING ENHANCED DEBUGGING")
    print("=" * 50)
    
    # Simulation request for B12 (the problematic target)
    test_request = {
        "file_id": "c9ebace1-dd72-4a9f-92da-62375ee630cd",
        "targets": ["B12"],
        "variables": [
            {
                "name": "F4",
                "sheet_name": "WIZEMICE Likest",
                "min_value": 0.08,
                "most_likely": 0.10,
                "max_value": 0.12
            },
            {
                "name": "F5", 
                "sheet_name": "WIZEMICE Likest",
                "min_value": 0.12,
                "most_likely": 0.15,
                "max_value": 0.18
            }
        ],
        "iterations": 10,
        "engine_type": "ultra"
    }
    
    print("📤 Sending test simulation request...")
    print("   Target: B12 (NPV formula)")
    print("   Variables: F4, F5")
    print("   Iterations: 10 (small test)")
    print("   Engine: Ultra")
    
    print("\n🔍 Check backend logs for enhanced debugging output:")
    print("   - [ULTRA_DEBUG] iteration details")
    print("   - [VAR_INJECT] variable injection")
    print("   - [NPV_DEBUG] NPV function calls")
    print("   - [CASH_FLOW] cash flow variations")
    
    print("\n💻 To run this test:")
    print("   curl -X POST http://localhost:8000/api/simulation/run \\")
    print('        -H "Content-Type: application/json" \\')
    print(f'        -d \'{json.dumps(test_request, indent=2)}\'')

if __name__ == "__main__":
    test_enhanced_debugging()
