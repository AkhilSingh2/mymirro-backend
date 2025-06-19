#!/usr/bin/env python3
"""
Test script for MyMirror Backend API
"""

import requests
import base64
import json
import os
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health")
        
        if response.status_code == 200:
            print("✅ Health check passed!")
            print(f"   Status: {response.json()}")
        else:
            print(f"❌ Health check failed! Status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on port 5000")
    except Exception as e:
        print(f"❌ Health check error: {e}")

def test_api_docs():
    """Test the API documentation endpoint."""
    print("\n📚 Testing API documentation endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/swagger.json")
        
        if response.status_code == 200:
            print("✅ API docs accessible!")
            docs = response.json()
            print(f"   Title: {docs.get('title', 'Unknown')}")
            print(f"   Version: {docs.get('version', 'Unknown')}")
            print(f"   Endpoints: {len(docs.get('endpoints', {}))}")
        else:
            print(f"❌ API docs failed! Status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API docs error: {e}")

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"❌ Error encoding image: {e}")
        return None

def test_color_analysis(image_path: str = None):
    """Test the color analysis endpoint."""
    print("\n🎨 Testing color analysis endpoint...")
    
    if image_path and os.path.exists(image_path):
        print(f"   Using image: {image_path}")
        base64_image = encode_image_to_base64(image_path)
        
        if not base64_image:
            print("❌ Failed to encode image")
            return
            
    else:
        print("   Using dummy base64 data (will likely fail face detection)")
        # Create a small dummy base64 image for testing API structure
        base64_image = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD//gA7Q1JFQVRPUjogZ2QtanBlZyB2MS4wICh1c2luZyBJSkcgSlBFRyB2ODApLCBxdWFsaXR5ID0gODUK/9sAQwAGBAUGBQQGBgUGBwcGCAoQCgoJCQoUDg0NDhQUERYWGhUVFhYeHBseHBgiGhsYHh0eIh4fHh0jIBsZJBgeJx0e/9sAQwEHBwcKCAoTCgoTHhseHB4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4e/8AAEQgAAQABAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBkQgUobHwFcHRU9Ph8RQ0YnKCkqOywuLy0tPy4uPz8/f08/P"
    
    try:
        # Test with invalid request first
        print("   Testing invalid request...")
        response = requests.post(f"{API_BASE_URL}/api/v1/color/analyze", json={})
        if response.status_code == 400:
            print("   ✅ Correctly rejected invalid request")
        else:
            print(f"   ⚠️ Unexpected status code for invalid request: {response.status_code}")
        
        # Test with image data
        print("   Testing with image data...")
        response = requests.post(
            f"{API_BASE_URL}/api/v1/color/analyze",
            json={"image": base64_image}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("   ✅ Color analysis successful!")
                analysis = result.get("analysis", {})
                recommendations = result.get("recommendations", {})
                
                print(f"   Undertone: {analysis.get('undertone', 'Unknown')}")
                print(f"   Contrast: {analysis.get('contrast', 'Unknown')}")
                print(f"   Dominant colors: {len(analysis.get('dominant_skin_colors', []))}")
                print(f"   Recommended colors: {len(recommendations.get('colors', []))}")
            else:
                print("   ✅ API responded but analysis failed (expected for dummy data)")
        else:
            print(f"   ❌ Color analysis failed! Status code: {response.status_code}")
            try:
                error_msg = response.json().get('error', 'Unknown error')
                print(f"   Error: {error_msg}")
            except:
                print(f"   Raw response: {response.text}")
                
    except Exception as e:
        print(f"❌ Color analysis error: {e}")

def test_invalid_endpoints():
    """Test invalid endpoints to ensure proper error handling."""
    print("\n🚫 Testing invalid endpoints...")
    
    # Test non-existent endpoint
    try:
        response = requests.get(f"{API_BASE_URL}/nonexistent")
        if response.status_code == 404:
            print("✅ Correctly returns 404 for non-existent endpoints")
        else:
            print(f"⚠️ Unexpected status code for invalid endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing invalid endpoint: {e}")

def run_all_tests(image_path: str = None):
    """Run all API tests."""
    print("🚀 Starting MyMirror Backend API Tests")
    print("=" * 50)
    
    test_health_check()
    test_api_docs()
    test_color_analysis(image_path)
    test_invalid_endpoints()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("\n💡 To test with a real image:")
    print("   python test_api.py path/to/your/image.jpg")

def main():
    """Main function to run tests."""
    import sys
    
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return
    
    run_all_tests(image_path)

if __name__ == "__main__":
    main() 