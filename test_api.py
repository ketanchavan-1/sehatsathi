#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test the API with different language inputs"""

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_api(text, lang):
    """Test the API with given text and language"""
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text}
        )
        
        print(f"\n{'='*60}")
        print(f"Language: {lang}")
        text_display = text[:30] + "..." if len(text) > 30 else text
        print(f"Input (truncated for display): [Text provided]")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Detected Language: {data.get('detected_language')}")
            print(f"Matched Symptoms: Found {len(data.get('matched_symptoms', []))} symptom(s)")
            predictions = data.get('predictions', [])
            print(f"Top Predictions: {len(predictions)} prediction(s)")
            for i, pred in enumerate(predictions[:3], 1):
                print(f"  {i}. {pred['disease']}")
            print("[SUCCESS] - API responded correctly")
        else:
            print(f"ERROR: {response.status_code}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

# Test cases
print("\n" + "="*60)
print("API TESTING SUITE - Language Support")
print("="*60)

# English tests
test_api("fever", "English (Single symptom)")
test_api("fever, headache, cough", "English (Multiple symptoms)")

# Hindi tests (Devanagari)
test_api("तेज बुखार", "Hindi (Single symptom - high fever)")
test_api("पेट दर्द, सिरदर्द", "Hindi (Multiple symptoms - abdominal pain, headache)")

# Marathi tests (Devanagari)
test_api("तीव्र ताप", "Marathi (Single symptom - high fever)")
test_api("पोटदुखी, डोकेदुखी", "Marathi (Multiple symptoms)")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60)
