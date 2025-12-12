#!/usr/bin/env python3
"""
Test script for ViT Classifier deployment
Tests both local and remote deployments
"""

import requests
import sys
import json
from pathlib import Path

def test_health(base_url):
    """Test the health endpoint"""
    print(f"\n🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Device: {data.get('device')}")
            print(f"   Model: {data.get('model')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_prediction(base_url, image_path=None):
    """Test the prediction endpoint"""
    print(f"\n🔍 Testing prediction endpoint...")
    
    # Create a simple test image if none provided
    if image_path is None or not Path(image_path).exists():
        print("   Creating test image...")
        try:
            from PIL import Image
            import io
            
            # Create a simple red square
            img = Image.new('RGB', (224, 224), color='red')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            files = {'image': ('test.png', img_bytes, 'image/png')}
        except ImportError:
            print("❌ PIL not installed. Please provide an image path.")
            return False
    else:
        print(f"   Using image: {image_path}")
        files = {'image': open(image_path, 'rb')}
    
    try:
        response = requests.post(f"{base_url}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            
            print(f"✅ Prediction successful!")
            print(f"\n   Top 5 predictions:")
            for i, pred in enumerate(predictions, 1):
                class_name = pred.get('class', 'Unknown')
                probability = pred.get('probability', 0) * 100
                print(f"   {i}. {class_name.replace('_', ' ').title()}: {probability:.2f}%")
            return True
        else:
            print(f"❌ Prediction failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_web_interface(base_url):
    """Test if web interface loads"""
    print(f"\n🌐 Testing web interface...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200 and 'CIFAR-100' in response.text:
            print(f"✅ Web interface loads successfully!")
            return True
        else:
            print(f"❌ Web interface test failed")
            return False
    except Exception as e:
        print(f"❌ Web interface test failed: {e}")
        return False

def main():
    print("="*60)
    print("Vision Transformer Deployment Test Suite")
    print("="*60)
    
    # Get base URL
    if len(sys.argv) > 1:
        base_url = sys.argv[1].rstrip('/')
    else:
        base_url = input("\nEnter your service URL (or press Enter for localhost:8080): ").strip().rstrip('/')
        if not base_url:
            base_url = "http://localhost:8080"
    
    print(f"\n🎯 Testing service at: {base_url}")
    
    # Get optional image path
    image_path = None
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
    
    # Run tests
    results = {
        'Health Check': test_health(base_url),
        'Web Interface': test_web_interface(base_url),
        'Prediction': test_prediction(base_url, image_path)
    }
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:<20} {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your deployment is working correctly.")
        print(f"\n🌐 Access your app at: {base_url}")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")
        print(f"\nView logs with:")
        print(f"  gcloud run services logs tail vit-classifier --region us-central1")
        sys.exit(1)

if __name__ == "__main__":
    main()