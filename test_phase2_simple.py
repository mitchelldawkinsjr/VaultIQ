#!/usr/bin/env python3
"""
Simple test for Phase 2 AI capabilities
Tests the core functionality with proper authentication
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:8003"
USERNAME = "admin"
PASSWORD = "admin123"

def test_with_session_auth():
    """Test Phase 2 APIs with Django session authentication"""
    session = requests.Session()
    
    print("🚀 VaultIQ Phase 2 AI Testing")
    print("=" * 40)
    
    # Step 1: Get CSRF token
    print("1. Getting CSRF token...")
    response = session.get(f"{BASE_URL}/accounts/login/")
    csrf_token = session.cookies.get('csrftoken')
    print(f"   CSRF token: {csrf_token[:10]}..." if csrf_token else "   No CSRF token")
    
    # Step 2: Login
    print("2. Logging in...")
    login_data = {
        'username': USERNAME,
        'password': PASSWORD,
        'csrfmiddlewaretoken': csrf_token
    }
    
    login_response = session.post(f"{BASE_URL}/accounts/login/", data=login_data)
    if login_response.status_code in [200, 302]:
        print("   ✅ Login successful")
    else:
        print(f"   ❌ Login failed: {login_response.status_code}")
        return
    
    # Step 3: Test Enhanced Status
    print("3. Testing Enhanced AI Status...")
    status_response = session.get(f"{BASE_URL}/api/enhanced-status/")
    if status_response.status_code == 200:
        try:
            status_data = status_response.json()
            print("   ✅ Status API works")
            ai_status = status_data['ai_status']
            print(f"      Phase 2: {ai_status['phase2_available']}")
            print(f"      Enhanced Search: {ai_status['enhanced_search']['available']}")
            print(f"      RAG Q&A: {ai_status['rag_qa']['available']}")
        except:
            print(f"   ❌ Status API response error: {status_response.text[:100]}")
    else:
        print(f"   ❌ Status API failed: {status_response.status_code}")
    
    # Step 4: Test Enhanced Search
    print("4. Testing Enhanced Search...")
    search_headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': session.cookies.get('csrftoken'),
        'Referer': BASE_URL
    }
    
    search_data = {
        "query": "personal growth and development",
        "k": 3
    }
    
    search_response = session.post(
        f"{BASE_URL}/api/enhanced-search/",
        json=search_data,
        headers=search_headers
    )
    
    if search_response.status_code == 200:
        try:
            search_result = search_response.json()
            if search_result['status'] == 'success':
                print("   ✅ Enhanced Search works!")
                print(f"      Query: '{search_result['query']}'")
                print(f"      Results: {search_result['total_results']}")
                print(f"      Model: {search_result['model_info']['model_name']}")
                
                # Show first result
                if search_result['results']:
                    first = search_result['results'][0]
                    print(f"      Sample: {first['video_title']}")
                    print(f"      Text: {first['text'][:80]}...")
            else:
                print(f"   ❌ Search Error: {search_result['message']}")
        except Exception as e:
            print(f"   ❌ Search response error: {e}")
    else:
        print(f"   ❌ Enhanced Search failed: {search_response.status_code}")
        try:
            print(f"      Error: {search_response.json()}")
        except:
            print(f"      Raw: {search_response.text[:200]}")
    
    # Step 5: Test RAG Q&A
    print("5. Testing RAG Q&A...")
    qa_data = {
        "question": "What are some tips for better communication?",
        "method": "auto"
    }
    
    qa_response = session.post(
        f"{BASE_URL}/api/rag-qa/",
        json=qa_data,
        headers=search_headers
    )
    
    if qa_response.status_code == 200:
        try:
            qa_result = qa_response.json()
            if qa_result['status'] == 'success':
                print("   ✅ RAG Q&A works!")
                print(f"      Question: '{qa_result['question']}'")
                print(f"      Answer: {qa_result['answer'][:100]}...")
                print(f"      Confidence: {qa_result['confidence']:.2f}")
                print(f"      Sources: {len(qa_result.get('sources', []))}")
            else:
                print(f"   ❌ Q&A Error: {qa_result['message']}")
        except Exception as e:
            print(f"   ❌ Q&A response error: {e}")
    else:
        print(f"   ❌ RAG Q&A failed: {qa_response.status_code}")
        try:
            print(f"      Error: {qa_response.json()}")
        except:
            print(f"      Raw: {qa_response.text[:200]}")
    
    # Step 6: Test Enhanced Search Interface
    print("6. Testing Enhanced Search Interface...")
    interface_response = session.get(f"{BASE_URL}/enhanced-search/")
    if interface_response.status_code == 200:
        if "VaultIQ Enhanced AI Search" in interface_response.text:
            print("   ✅ Enhanced Search UI loads")
        else:
            print("   ⚠️  Interface loads but content may be different")
    else:
        print(f"   ❌ Interface failed: {interface_response.status_code}")
    
    print("\n" + "=" * 40)
    print("🎉 Phase 2 Testing Complete!")

if __name__ == "__main__":
    test_with_session_auth() 