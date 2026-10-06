import streamlit as st
import os
import sys
import requests
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils"))
from helpers import apply_custom_css, init_session_state, display_disclaimer

load_dotenv()

st.set_page_config(page_title="AI Health Assistant | Chatbot", page_icon="💬", layout="wide")
apply_custom_css()
init_session_state()

st.title("💬 Educational AI Healthcare Chatbot")
st.write("Ask educational questions about symptoms, health terminology, disease prevention, and wellness habits.")

display_disclaimer()

# Check for API keys
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")

# Offline Knowledge Base / FAQ Fallback Dictionary
OFFLINE_FAQ = {
    "diabetes": (
        "**Diabetes Mellitus** is a chronic condition where the body cannot properly process blood glucose (sugar). "
        "\n\n- **Type 1 Diabetes**: An autoimmune condition where the pancreas produces little to no insulin."
        "\n- **Type 2 Diabetes**: A metabolic condition where body cells become resistant to insulin."
        "\n- **Common Symptoms**: Excessive thirst, frequent urination, unexplained weight loss, and fatigue."
        "\n\n*Note: Diagnosis requires clinical laboratory blood tests (HbA1c or Fasting Glucose).* "
    ),
    "blood pressure": (
        "**Blood Pressure** is the force exerted by circulating blood against the walls of body arteries. "
        "\n\n- **Systolic (Top number)**: Pressure when the heart beats."
        "\n- **Diastolic (Bottom number)**: Pressure when the heart rests between beats."
        "\n- **Normal Range**: Less than 120/80 mm Hg."
    ),
    "cholesterol": (
        "**Cholesterol** is a waxy substance found in blood cells needed to build healthy cells. "
        "\n\n- **HDL (Good Cholesterol)**: Helps transport cholesterol back to the liver for removal."
        "\n- **LDL (Bad Cholesterol)**: High levels can cause plaque buildup in arteries."
    ),
    "bmi": (
        "**Body Mass Index (BMI)** is a numerical measurement derived from weight and height (weight in kg / height in m²). "
        "It provides a general classification of underweight (<18.5), normal weight (18.5-24.9), overweight (25-29.9), and obesity (30+)."
    ),
    "symptoms": (
        "Common health symptoms to monitor include unexplained fatigue, persistent cough, unusual weight changes, or high fever. "
        "\n\n🚨 **Emergency Warning**: If you or someone else experiences severe chest pain, shortness of breath, sudden numbness, "
        "or loss of consciousness, seek immediate emergency medical services!"
    ),
    "lifestyle": (
        "Healthy lifestyle practices include: "
        "\n- 🥗 Eating a balanced diet rich in fiber, vegetables, and lean proteins."
        "\n- 🏃 Getting at least 150 minutes of moderate aerobic exercise per week."
        "\n- 😴 Maintaining 7-9 hours of restful sleep daily."
        "\n- 💧 Staying hydrated and avoiding tobacco."
    ),
    "heart disease": (
        "**Heart Disease** (Cardiovascular Disease) refers to conditions affecting the heart structure and blood vessels, "
        "including coronary artery disease, heart valve disorders, and arrhythmias."
    )
}

def get_offline_response(prompt):
    """Generates rule-based offline response when API is unavailable."""
    prompt_lower = prompt.lower()
    
    # Check for emergency triggers
    if any(term in prompt_lower for term in ["emergency", "chest pain", "faint", "heart attack", "stroke", "cannot breathe"]):
        return (
            "🚨 **EMERGENCY WARNING**: If you are experiencing severe chest pain, sudden weakness/numbness, difficulty breathing, "
            "or a medical crisis, **PLEASE CALL EMERGENCY MEDICAL SERVICES (911 or local emergency number) IMMEDIATELY!** "
            "Do not rely on online software for emergency evaluation."
        )
        
    for key, text in OFFLINE_FAQ.items():
        if key in prompt_lower:
            return f"ℹ️ **Educational Overview - {key.title()}**:\n\n{text}"
            
    return (
        "I am an educational AI Health Assistant. I can answer questions about general health concepts such as "
        "**Diabetes**, **Blood Pressure**, **Cholesterol**, **BMI**, **Heart Disease**, and **Healthy Lifestyle Habits**.\n\n"
        "Please ask about one of these topics! *Note: I cannot diagnose conditions or prescribe medications.*"
    )

def query_gemini_api(prompt, api_key):
    """Calls Gemini REST API with strict healthcare guardrail system instructions."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    system_instruction = (
        "You are an educational AI Healthcare Assistant. "
        "Rules: 1. Answer general healthcare, wellness, and medical definition questions clearly. "
        "2. NEVER diagnose diseases or prescribe medications/dosages. "
        "3. For severe or emergency symptoms, explicitly advise seeking immediate emergency medical care. "
        "4. Keep responses concise and structured."
    )
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{system_instruction}\n\nUser Question: {prompt}"}]}
        ]
    }
    try:
        resp = requests.post(url, json=payload, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return text
        else:
            return get_offline_response(prompt)
    except Exception:
        return get_offline_response(prompt)

# Initialize chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "assistant", "content": "Hello! I am your AI Health Assistant. Ask me any educational question about blood pressure, diabetes, cholesterol, BMI, or healthy habits!"}
    ]

# Render chat messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_prompt := st.chat_input("Ask a health question (e.g., 'What is Type 2 Diabetes?')..."):
    st.session_state.chat_messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Processing healthcare knowledge..."):
            if API_KEY and API_KEY != "your_gemini_api_key_here":
                reply = query_gemini_api(user_prompt, API_KEY)
            else:
                reply = get_offline_response(user_prompt)
                
            st.markdown(reply)
            st.session_state.chat_messages.append({"role": "assistant", "content": reply})
