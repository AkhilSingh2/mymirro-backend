#!/usr/bin/env python3
"""
Demonstration of the Dual-Mode Color Analysis API

This script shows how to use both modes:
1. Photo upload for automatic skin tone detection
2. Manual hex color selection
"""

import requests
import json

def demo_hex_analysis():
    """Demonstrate hex color analysis."""
    print("🎨 Demo: Manual Hex Color Analysis")
    print("-" * 40)
    
    # Test different skin tones
    skin_tones = [
        ("#F8D7C4", "Very Light"),
        ("#E8C5A0", "Light"),
        ("#C8967B", "Medium"),
        ("#A67857", "Medium-Dark"),
        ("#6B4226", "Dark")
    ]
    
    for hex_color, description in skin_tones:
        url = "http://localhost:8000/api/v1/color/analyze"
        payload = {"hex_color": hex_color}
        
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"   {description} ({hex_color}):")
                print(f"      → Undertone: {result['undertone']}")
                print(f"      → Fitzpatrick: {result['fitzpatrick_scale']}")
                print(f"      → Lightness: {result['lightness']:.1f}")
                print(f"      → Formal colors: {len(result['recommended_colours']['Formal'])} options")
            else:
                print(f"   {description} ({hex_color}): Error - {response.status_code}")
                
        except Exception as e:
            print(f"   {description} ({hex_color}): Exception - {e}")
    
    print()

def demo_unified_api():
    """Demonstrate the unified API endpoint."""
    print("🔄 Demo: Unified API (Both Modes)")
    print("-" * 40)
    
    # Test 1: Hex mode
    print("   Test 1: Using hex color")
    url = "http://localhost:8000/api/v1/color/analyze"
    payload = {"hex_color": "#D4A574"}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"      ✅ {result['undertone']} undertone, Type {result['fitzpatrick_scale']}")
            print(f"      💡 Recommended for Streetwear: {len(result['recommended_colours']['Streetwear'])} colors")
        else:
            print(f"      ❌ Error: {response.status_code}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")
    
    # Test 2: Error handling
    print("   Test 2: Error handling (no input)")
    payload = {}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 400:
            result = response.json()
            print(f"      ✅ Correctly rejected: {result.get('error', 'No error message')}")
        else:
            print(f"      ❌ Unexpected status: {response.status_code}")
    except Exception as e:
        print(f"      ❌ Exception: {e}")
    
    print()

def show_api_endpoints():
    """Show available API endpoints."""
    print("📡 Available API Endpoints")
    print("-" * 40)
    print("   1. Unified Analysis (Recommended):")
    print("      POST /api/v1/color/analyze")
    print("      Body: {\"hex_color\": \"#C8967B\"} OR {\"image\": \"base64...\"}")
    print()
    print("   2. Hex-only Analysis:")
    print("      POST /api/v1/color/analyze-hex")
    print("      Body: {\"hex_color\": \"#C8967B\"}")
    print()
    print("   3. Photo-only Analysis:")
    print("      POST /api/v1/color/analyze-photo")
    print("      Body: {\"image\": \"data:image/jpeg;base64,...\"}")
    print()
    print("   4. Swagger UI: http://localhost:8000/swagger/")
    print()

def main():
    """Run the demonstration."""
    print("🎯 MyMirror Color Analysis API - Dual Mode Demo")
    print("=" * 50)
    print()
    
    # Check if API is running
    try:
        health_response = requests.get("http://localhost:8000/api/v1/health")
        if health_response.status_code == 200:
            print("✅ API is running!")
            print()
        else:
            print("❌ API health check failed!")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("💡 Make sure the Flask app is running: python app.py")
        return
    
    # Show endpoints
    show_api_endpoints()
    
    # Run demos
    demo_hex_analysis()
    demo_unified_api()
    
    print("🎉 Demo completed!")
    print("💡 Try the Swagger UI at http://localhost:8000/swagger/ for interactive testing")

if __name__ == "__main__":
    main() 