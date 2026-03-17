"""
debug_response.py — prints full raw API response to diagnose issues
Run: python debug_response.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"

questions = [
    "How many customers do we have?",
    "Which customers placed the most orders?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"QUESTION: {q}")
    print('='*60)
    r = requests.post(
        f"{BASE_URL}/query",
        json={"question": q, "evaluate": False},
        timeout=60
    )
    data = r.json()
    print(json.dumps(data, indent=2, default=str))