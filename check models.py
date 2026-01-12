import google.generativeai as genai
import streamlit as st
import os

# Try to get key from secrets, or environment, or ask user
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ Error: Could not find API Key in .streamlit/secrets.toml")
else:
    genai.configure(api_key=api_key)
    print("\n🔍 Checking available models for your API key...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"   ✅ Available: {m.name}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")