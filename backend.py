from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any
import uuid
import logging
import os
import json
from datetime import datetime

# --- import your conversation logic ---
from agent import *
# ---------------------------------------------------------------------
# ✅ Initialize FastAPI
# ---------------------------------------------------------------------
app = FastAPI(title="Post-Discharge Assistant API")

# Allow frontend (React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for security later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# ✅ Global in-memory session store
# ---------------------------------------------------------------------
sessions: Dict[str, ConversationState] = {}

# Compile the agent graph once
compiled_graph = super_graph()

# ---------------------------------------------------------------------
# ✅ Logging setup
# ---------------------------------------------------------------------
# Create a logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def setup_session_logger(session_id: str):
    """Create and configure a dedicated logger for each session."""
    logger = logging.getLogger(session_id)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if already added
    if not logger.handlers:
        # Create a file handler for this session
        file_handler = logging.FileHandler(f"logs/{session_id}_backend.json", mode="a", encoding="utf-8")

        # JSON-style log format
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", "message": %(message)s}',
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# ---------------------------------------------------------------------
# ✅ Request schema
# ---------------------------------------------------------------------
class MessageInput(BaseModel):
    session_id: str = None
    message: str

@app.get("/favicon.ico")
def favicon():
    # return empty 204 so browsers stop requesting and logging 404
    return Response(status_code=204)

# ---------------------------------------------------------------------
# ✅ Chat endpoint
# ---------------------------------------------------------------------
@app.post("/chat")
async def chat_endpoint(input_data: MessageInput):
    # Use an existing session or create a new one
    session_id = input_data.session_id or str(uuid.uuid4())
    state = sessions.get(session_id, ConversationState())
    logger = setup_session_logger(session_id)

    # Append user message
    state["messages"].append({"role": "user", "content": input_data.message})
    logger.info(json.dumps({
        "event": "user_message",
        "session_id": session_id,
        "agent": "user",
        "message": input_data.message
    }))

    # Determine current agent
    current_agent = state.get("active_agent", "user_input")
    collected_responses = []

    # --- Run the current agent ---
    if current_agent == "user_input":
        state = user_input(state)
        next_node = "receptionist"

        # 🚀 Immediately call receptionist for the first message
        print("🧾 First message → forwarding to receptionist")
        state["active_agent"] = "receptionist"
        state = receptionist_node(state)
        last_message = state["messages"][-1]["content"]
        collected_responses.append({"agent": "Receptionist", "message": last_message})
        logger.debug(json.dumps({
            "event": "agent_response",
            "agent": state["active_agent"],
            "response": last_message
        }))
        next_node = next_agent(state)

        # Handle clinical handoff if needed
        if next_node == "clinical":
            print("→ Handoff detected on first turn! Invoking clinical agent...")
            state["active_agent"] = "clinical"
            state = clinical_node(state)
            last_message = state["messages"][-1]["content"]
            collected_responses.append({"agent": "Clinical", "message": last_message})
            logger.debug(json.dumps({
                "event": "agent_response",
                "agent": state["active_agent"],
                "response": last_message
            }))
            next_node = next_agent(state)

    elif current_agent == "receptionist":
        print("🧾 Inside the receptionist agent")
        state = receptionist_node(state)
        if state["messages"]:
            collected_responses.append({
                "agent": "receptionist",
                "message": state["messages"][-1]["content"]
            })
        logger.debug(json.dumps({
                "event": "agent_response",
                "agent": state["active_agent"],
                "response": state["messages"][-1]["content"]
            }))
        next_node = next_agent(state)

        # 🚀 If receptionist decides to handoff to clinical agent
        if next_node == "clinical" or state.get("clinical_engaged"):
            print("→ Handoff detected! Invoking clinical agent...")
            state["active_agent"] = "clinical"
            state = clinical_node(state)
            if state["messages"]:
                collected_responses.append({
                    "agent": "Clinical AI Agent",
                    "message": state["messages"][-1]["content"]
                })
            logger.debug(json.dumps({
                "event": "agent_response",
                "agent": state["active_agent"],
                "response": state["messages"][-1]["content"]
            }))
            next_node = next_agent(state)

    elif current_agent == "clinical":
        print("🩺 Inside the clinical agent")
        state = clinical_node(state)
        if state["messages"]:
                collected_responses.append({
                    "agent": "Clinical AI Agent",
                    "message": state["messages"][-1]["content"]
                })
        logger.debug(json.dumps({
                "event": "agent_response",
                "agent": state["active_agent"],
                "response": state["messages"][-1]["content"]
            }))
        next_node = next_agent(state)

    else:
        next_node = "end"

    # --- Transition management ---
    state["last_agent_call"] = current_agent
    if next_node != "end":
        state["active_agent"] = next_node

    # --- Prepare response ---
    if next_node == "end" or state.get("end_conversation"):
        response_text = "Thank you! The session has ended."
        end = True
    else:
        last_message = state["messages"][-1]
        response_text = last_message["content"]
        end = False

    # Save session
    sessions[session_id] = state

    print(f"🧭 [Router] Active agent: {state.get('active_agent')}")
    print(f"    clinical_engaged: {state.get('clinical_engaged', False)}")
    print(f"    end_conversation: {state.get('end_conversation', False)}")

    return {
        "session_id": session_id,
        "agent": state.get("active_agent"),
        "response": collected_responses,
        "end_conversation": end,
    }

    state["last_agent_call"] = current_agent

    # ✅ Update which agent is active next
    if next_node != "end":
        state["active_agent"] = next_node

    # ✅ Prepare response
    if next_node == "end" or state.get("end_conversation"):
        response_text = "Thank you! The session has ended."
        end = True
    else:
        last_message = state["messages"][-1]
        response_text = last_message["content"]
        end = False


    # Save session state
    sessions[session_id] = state

    return {
        "session_id": session_id,
        "agent": state.get("active_agent"),
        "response": response_text,
        "end_conversation": end,
    }


# ---------------------------------------------------------------------
# ✅ Root test route
# ---------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Post-Discharge Medical Assistant API running"}


# ---------------------------------------------------------------------
# ✅ Run server
# ---------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, log_level="info",  reload=True)