import streamlit as st
import requests
from typing import Any, Dict
import json
import logging
import os
import json
from datetime import datetime

st.set_page_config(page_title="Datasmith AI Chat", page_icon="💬")
st.title("Datasmith AI — Post-Discharge Assistant")

os.makedirs("logs", exist_ok=True)

def setup_session_logger():
    """Create and configure a dedicated logger for each session."""
    logger = logging.getLogger(datetime)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if already added
    if not logger.handlers:
        # Create a file handler for this session
        file_handler = logging.FileHandler(f"logs/{datetime}_streamlit.json", mode="a", encoding="utf-8")

        # JSON-style log format
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Backend configuration (adjust if your backend runs elsewhere)
if "backend_url" not in st.session_state:
    st.session_state.backend_url = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8000")

if "messages" not in st.session_state:
    # messages is a list of dicts: {"role": "user"|"assistant"|"system", "content": str}
    st.session_state.messages = []

if "session_id" not in st.session_state:
    # create a new session with the backend
    try:
        r = requests.post(f"{st.session_state.backend_url}/start", timeout=10)
        r.raise_for_status()
        st.session_state.session_id = r.json().get("session_id")
    except Exception as e:
        st.error(f"Could not create session with backend: {e}")
        st.stop()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.sidebar.markdown("Connects to the FastAPI backend. Change the URL and reload if needed.")

def send_to_backend(message: str) -> Dict[str, Any]:
    payload = {"session_id": st.session_state.get("session_id"), "message": message}
    try:
        r = requests.post(f"{st.session_state.backend_url}/chat", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# React to user input
if prompt := st.chat_input("Type a message and press Enter..."):
    # Display user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send to backend
    result = send_to_backend(prompt)

    if result.get("error"):
        with st.chat_message("assistant"):
            st.markdown(f"⚠️ Error contacting backend: {result['error']}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {result['error']}"})
    else:
        # Get the list of agent responses from backend
        responses = result.get("response") or result.get("responses") or []

        # Iterate through all responses and display
        for resp in responses:
            agent_name = resp.get("agent", "Assistant")
            message = resp.get("message", "")

            if not message.strip():
                # skip empty messages
                continue

            display_name = f"**{agent_name}:**"
            with st.chat_message("assistant"):
                st.markdown(f"{display_name} {message}")

            st.session_state.messages.append({"role": agent_name, "content": message})
