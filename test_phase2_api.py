#!/usr/bin/env python3
"""
Test script for Phase 2 AI capabilities
Tests enhanced search and RAG Q&A functionality
"""

import requests
import json
import sys
from requests.auth import HTTPBasicAuth

# Configuration
BASE_URL = "http://localhost:8003"
USERNAME = "admin"  # User with videos
PASSWORD = "admin"  # You might need to set this

def get_csrf_token(session):
    """Get CSRF token for authenticated requests"""
    response = session.get(f"{BASE_URL}/")
    if 'csrftoken' in session.cookies:
        return session.cookies['csrftoken']
    return None

def login_user(session, username, password):
    """Login user and get session"""
    # Get login page to get CSRF token
    login_page = session.get(f"{BASE_URL}/accounts/login/")
    csrf_token = None
    
    if 'csrftoken' in session.cookies:
        csrf_token = session.cookies['csrftoken']
    
    # Login
    login_data = {
        'username': username,
        'password': password,
        'csrfmiddlewaretoken': csrf_token
    }
    
    response = session.post(f"{BASE_URL}/accounts/login/", data=login_data)
    return response.status_code in [200, 302]

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health/")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Health Status: {data['status']}")
        print(f"   Phase 2 AI: {data.get('phase2_ai', 'unknown')}")
        print(f"   Enhanced Search: {data.get('enhanced_search', 'unknown')}")
        print(f"   RAG Q&A: {data.get('rag_qa', 'unknown')}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_enhanced_search(session):
    """Test enhanced search API"""
    print("\n🧠 Testing Enhanced Search API...")
    
    csrf_token = get_csrf_token(session)
    
    search_data = {
        "query": "How to improve communication skills?",
        "k": 5,
        "min_similarity": 0.3
    }
    
    headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrf_token,
        'Referer': BASE_URL
    }
    
    response = session.post(
        f"{BASE_URL}/api/enhanced-search/",
        json=search_data,
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            print(f"✅ Enhanced Search Success!")
            print(f"   Query: '{data['query']}'")
            print(f"   Results: {data['total_results']}")
            print(f"   Search Type: {data['search_type']}")
            print(f"   Model: {data['model_info']['model_name']}")
            
            # Show sample results
            for i, result in enumerate(data['results'][:2]):
                print(f"\n   Result {i+1}:")
                print(f"     Video: {result['video_title']}")
                print(f"     Text: {result['text'][:100]}...")
                print(f"     Confidence: {result['confidence']:.2f}")
                print(f"     Time: {result.get('timestamp_formatted', 'N/A')}")
            
            return True
        else:
            print(f"❌ Enhanced Search Error: {data['message']}")
    else:
        print(f"❌ Enhanced Search Failed: {response.status_code}")
        try:
            error_data = response.json()
            print(f"   Error: {error_data}")
        except:
            print(f"   Raw response: {response.text}")
    
    return False

def test_rag_qa(session):
    """Test RAG Q&A API"""
    print("\n🤖 Testing RAG Q&A API...")
    
    csrf_token = get_csrf_token(session)
    
    qa_data = {
        "question": "What are the key principles for personal growth?",
        "method": "auto",
        "include_sources": True
    }
    
    headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrf_token,
        'Referer': BASE_URL
    }
    
    response = session.post(
        f"{BASE_URL}/api/rag-qa/",
        json=qa_data,
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            print(f"✅ RAG Q&A Success!")
            print(f"   Question: '{data['question']}'")
            print(f"   Answer: {data['answer']}")
            print(f"   Confidence: {data['confidence']:.2f}")
            print(f"   Method: {data['method']}")
            print(f"   Sources: {len(data.get('sources', []))}")
            
            # Show sample sources
            for i, source in enumerate(data.get('sources', [])[:2]):
                print(f"\n   Source {i+1}:")
                print(f"     Video: {source['video_title']}")
                print(f"     Text: {source['text'][:100]}...")
                print(f"     Time: {source.get('timestamp_formatted', 'N/A')}")
            
            return True
        else:
            print(f"❌ RAG Q&A Error: {data['message']}")
    else:
        print(f"❌ RAG Q&A Failed: {response.status_code}")
        try:
            error_data = response.json()
            print(f"   Error: {error_data}")
        except:
            print(f"   Raw response: {response.text}")
    
    return False

def test_enhanced_status(session):
    """Test enhanced AI status API"""
    print("\n📊 Testing Enhanced AI Status...")
    
    response = session.get(f"{BASE_URL}/api/enhanced-status/")
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            print("✅ Enhanced Status Success!")
            ai_status = data['ai_status']
            print(f"   Phase 2 Available: {ai_status['phase2_available']}")
            
            # Enhanced Search Status
            es = ai_status['enhanced_search']
            print(f"   Enhanced Search: Available={es['available']}, Initialized={es['initialized']}")
            
            # RAG Q&A Status  
            qa = ai_status['rag_qa']
            print(f"   RAG Q&A: Available={qa['available']}")
            
            return True
        else:
            print(f"❌ Enhanced Status Error: {data}")
    else:
        print(f"❌ Enhanced Status Failed: {response.status_code}")
    
    return False

def main():
    """Main test runner"""
    print("🚀 VaultIQ Phase 2 AI Testing Suite")
    print("=" * 50)
    
    # Test health check (no auth needed)
    if not test_health_check():
        print("❌ Basic health check failed, stopping tests")
        return
    
    # Create session for authenticated requests
    session = requests.Session()
    
    # For testing, we'll try without authentication first
    # since the APIs might not require it for testing
    
    test_results = []
    
    # Test enhanced status
    test_results.append(("Enhanced Status", test_enhanced_status(session)))
    
    # Test enhanced search
    test_results.append(("Enhanced Search", test_enhanced_search(session)))
    
    # Test RAG Q&A
    test_results.append(("RAG Q&A", test_rag_qa(session)))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    total = len(test_results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 AI tests passed!")
    else:
        print("⚠️  Some tests failed - check authentication or API endpoints")

if __name__ == "__main__":
    main() 