from typing import TypedDict, List
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import json
from typing import Dict, Any
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY"),
    temperature = 0.2
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.load_local(
    "nephrology_db",
    embedding_model,
    allow_dangerous_deserialization=True  # required in LangChain >=0.2
)

logging.basicConfig(
    filename="(POC)system_logs.json",
    level=logging.DEBUG,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)

receptionist_prompt = """
You are the Receptionist Agent for a Post-Discharge Medical Assistant system.

Your tasks are strictly limited to:
1. Greeting the patient politely.
2. Asking for their full name if it is not provided.
3. Using the provided tool `retrieve_patient_discharge_report` to fetch their discharge report by name.
4. Once the report is retrieved:
   - Confirm that you found the report.
   - Do Not summarize the entire report.
   - Just politely ask one general follow-up question based on the discharge information.
5. If the patient says anything related to symptoms or medical concerns (e.g., swelling, pain, shortness of breath, dizziness, etc.), do NOT answer directly.
   - Instead say: "That sounds like a medical concern. Let me connect you with our Clinical AI Agent."
6. If no record is found, politely apologize and ask them to confirm their name.
7. Never provide medical advice yourself.

Maintain a friendly, calm, and caring tone.
Your role ends when you hand the conversation over to the Clinical Agent for medical advice.
"""

clinical_agent_prompt = """
You are the Clinical AI Agent in a Post-Discharge Medical Assistant system created by DataSmith AI.

Your responsibilities:
1. Handle all medical and clinical queries routed to you by the Receptionist Agent.
2. Use the provided tools to gather accurate medical information:
   - 'rag_search' → search the Nephrology reference materials (embeddings database).
   - 'web_search' → search the web for latest or missing information (fallback).
3. Always prefer the nephrology reference book as your first source. 
   Only use the web if the reference materials do not contain relevant information.
4. When you respond:
   - Clearly explain the answer in simple language the patient can understand.
   - Include brief, inline citations (e.g., “According to nephrology guidelines [Source: Comprehensive Clinical Nephrology or Web]”).
   - If the user asks for any latest information related to medical topics, always use the web search tool along with the RAG.
   - Always mention the Source of the information in your Answer after that sentence.
   - Never provide unverified or unsafe advice.
   - Remind patients that you are an AI and they should consult their doctor for medical emergencies.
5. Always log your interaction in detail, including:
   - Patient name (if known)
   - Query asked
   - Data sources used (RAG or Web)
   - Citations or reference identifiers
   - Summary of your answer
6. If the query seems unrelated to medicine or nephrology, politely inform the user that you can only assist with medical questions.

Tone and style:
- Be professional, calm, and caring.
- Avoid overly technical medical jargon.
- Always prioritize patient safety.

Disclaimers:
- Add this note at the end of every response:
  “⚠️ This information is for educational purposes only. Please consult a qualified healthcare provider for medical advice.”
"""

def retrieve_patient_discharge_report(patient_name: str) -> Dict[str, Any]:
    """
    Retrieve a patient's discharge report by their full name.
    Returns structured information about diagnosis, medications, and follow-up details.
    """
    with open("patient_reports.json", "r") as f:
        logging.info(f"Retrieving discharge report for patient: {patient_name}")
        reports = json.load(f)
    for patient in reports["patients"]:
        if patient["patient_name"] == patient_name:
            logging.info(f"Retrieved Discharge report for patient: {patient_name}")
            print(f"Discharge report for {patient_name}: {patient}")
            return patient
    logging.warning(f"No record found for patient: {patient_name}")
    return f"No record found for patient: {patient_name}"

def knowledge_search(query: str):
    """
    Run a similarity search on the Nephrology reference materials.
    Returns the top 3 most relevant document chunks.
    """
    logging.info(f"Retrieved Knowledge for query: {query}")
    results = vectordb.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in results])

receptionist_agent = create_agent(
    model = llm,
    tools=[retrieve_patient_discharge_report],
    system_prompt = receptionist_prompt
)

clinical_agent = create_agent(
    model = llm,
    tools=[
        DuckDuckGoSearchRun()
    ],
    system_prompt = clinical_agent_prompt
)

class ConversationState(dict):
    """A simple mock to hold conversation state."""
    def __init__(self):
        super().__init__()
        self["messages"] = []
        self["active_agent"] = "user_input"
        self["end_conversation"] = False
        self["clinical_engaged"] = False
        self["user_message"] = ""

# -----------------------------
# Node definitions
# -----------------------------

# User Node
def user_input(state):
    # print("\n💬 Waiting for user input...")
    # user_input = input("Patient: ").strip()
    # state["messages"].append({"role": "user", "content": user_input})
    # This node is driven by incoming messages appended to state["messages"] by the backend.
    # Avoid referencing undefined variables and instead pick the latest user content if present.
    latest = next((m["content"] for m in reversed(state.get("messages", [])) if isinstance(m, dict) and m.get("role") == "user"), "")
    state["user_message"] = latest
    # state["active_agent"] = "user_input"
    return state

# Receptionist Node
def receptionist_node(state):
    print("\n🧾 Inside the receptionist agent")
    latest_user_input = next((m["content"] for m in reversed(state["messages"]) if m.get("role") == "user"), "")
    if latest_user_input == "bye":
        state["end_conversation"] = True
        return state

    # Safely extract the last user message content (supports both dict messages and raw strings)
    # Protect against empty message lists to avoid IndexError when using [-1].
    last_msg = state.get("messages")[-1] if state.get("messages") else ""
    if isinstance(last_msg, dict):
        user_input = last_msg.get("content", "")
    else:
        user_input = str(last_msg)

    # Track which messages this agent has processed
    processed = state.setdefault("processed_messages", {})

    # Use the message text as the key (always a hashable string)
    key = user_input
    if processed.get(key) == "receptionist":
        return state  # skip if already handled

    if not user_input.strip():
        processed[key] = "receptionist"
        return state

    try:
        # Prepare messages for the LLM, extracting content when message entries are dicts
        agent_input = [{"role": "user", "content": (m.get("content") if isinstance(m, dict) else str(m))} for m in state["messages"]]
        response = receptionist_agent.invoke({"messages": agent_input})

        # Normalize the text from response
        if isinstance(response, dict) and "messages" in response:
            last_message = response["messages"][-1].content
            if isinstance(last_message, list):
                text = " ".join([m.get("text", "") for m in last_message])
            else:
                text = last_message
        else:
            text = str(response)

        print(f"Receptionist Agent: {text}")
        state["messages"].append({"role": "assistant", "content": text})
        processed[key] = "receptionist"
        logging.info(f"Receptionist Agent response: {text}")

    except Exception as e:
        print(f"⚠️ Gemini LLM Error: {e}")
        text = "I'm sorry, there was an issue retrieving that information."
        state["messages"].append({"role": "assistant", "content": text})

    # Conditional routing logic (for conditional edge behavior)
    if "That sounds like a medical concern. Let me connect you with our Clinical AI Agent." in text:
        print("→ Detected medical concern, forwarding to clinical agent.")
        state["active_agent"] = "receptionist"
        state["clinical_engaged"] = True
    elif any(word in user_input.lower() for word in ["bye", "goodbye", "thanks"]):
        print("→ Ending conversation.")
        state["end_conversation"] = True
        state["active_agent"] = "receptionist"
    else:
        print("→ Continuing with receptionist.")
        state["active_agent"] = "receptionist"

    return state

# Clinical AI Node
def clinical_node(state):
    print("\n🩺 Inside the clinical agent")
    latest_user_input = next((m["content"] for m in reversed(state["messages"]) if m.get("role") == "user"), "")
    if latest_user_input == "bye":
        state["end_conversation"] = True
        return state
    # Step 1: Get latest user input
    agent_input = [{"role": "user", "content": (m.get("content") if isinstance(m, dict) else str(m))} for m in state["messages"]]
    print(f"Clinical Agent Input Messages: {agent_input}")

    knowledge_search_query = llm.invoke([
        SystemMessage(content="Extract the main medical concern or symptom described by the patient in the following conversation. Return it as a clear medical search query suitable for retrieving relevant nephrology information either from web or vector database created using a nephrology book. Respond only with the extracted query."),
        HumanMessage(content=f"Patient Conversation: {agent_input}")
    ])
    logging.info("Created a query to search in vector database.")

    print(f"Knowledge Search Query: {knowledge_search_query}")

    # # Step 2: Perform Semantic Search on Nephrology DB
    knowledge = knowledge_search(knowledge_search_query.content)

        # Step 3: Call the agent (which can use web search tool if needed)
    response = clinical_agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"Reference Materials:\n{knowledge}\n\nPatient Query:\n{agent_input}"
                }
            ]
    })

    # Step 4: Safely extract model's reply text
    if isinstance(response, dict) and "messages" in response:
            last_message = response["messages"][-1].content
            if isinstance(last_message, list):
                text = " ".join([m.get("text", "") for m in last_message])
            else:
                text = last_message
    else:
        text = str(response)

    print(f"Receptionist Agent: {text}")
    state["messages"].append({"role": "assistant", "content": text})
    last_msg = state["messages"][-1]["content"].lower()
    logging.info(f"Clinical AI Agent response: {text}")

    if "thank" in last_msg or "bye" in last_msg:
        print("→ Clinical ends the session.")
        state["end_conversation"] = True
    else:
        print("→ Returning to user input.")
        state["clinical_engaged"] = True
        state["active_agent"] = "clinical"  # keep as clinical, router handles the rest

    return state


# Define state graph
def super_graph():
    graph = StateGraph(ConversationState)

    # Add nodes
    graph.add_node("receptionist", receptionist_node)
    graph.add_node("clinical", clinical_node)
    graph.add_node("user_input", user_input)

    # Start
    graph.add_edge(START, "user_input")
    # Conditional edges
    graph.add_conditional_edges(
        "user_input",
        lambda s: "clinical" if s.get("clinical_engaged") else "receptionist",
        {
            "receptionist": "receptionist",
            "clinical": "clinical"
        }
    )

    graph.add_conditional_edges(
        "receptionist",
        lambda s: (
            "clinical" if s.get("clinical_engaged")
            else "end" if s.get("end_conversation")
            else "user_input"
        ),
        {
            "clinical": "clinical",
            "end": END,
            "user_input": "user_input"
        }
    )

    graph.add_conditional_edges(
        "clinical",
        lambda s: "end" if s.get("end_conversation") else "user_input",
        {
            "user_input": "user_input",
            "end": END
        }
    )

    # Compile
    compiled_graph = graph.compile()
    return compiled_graph


def next_agent(state):
    active = state["active_agent"]
    print(f"\n🧭 [Router] Active agent: {active}")
    print(f"    clinical_engaged: {state.get('clinical_engaged', False)}")
    print(f"    end_conversation: {state.get('end_conversation', False)}")

    # 1️⃣ End conversation check
    if state.get("end_conversation", False):
        print("→ Router: Conversation ended by flag.")
        return "end"
    
    if active == "receptionist" and state.get("clinical_engaged"):
        return "clinical"

    # 2️⃣ Normal pre-clinical flow (before any handoff)
    if not state.get("clinical_engaged", False):
        if active == "user_input":
            print("→ Router: Normal mode → sending to receptionist")
            return "receptionist"
        elif active == "receptionist":
            # if the receptionist *just* triggered a clinical handoff
            if state.get("active_agent") == "clinical":
                print("→ Router: Receptionist triggered clinical handoff.")
                return "clinical"
            print("→ Router: Normal mode → back to user input")
            return "user_input"

    # 3️⃣ Clinical mode flow (after handoff)
    if state.get("clinical_engaged", False):
        if active == "user_input":
            print("→ Clinical mode: sending to clinical")
            return "clinical"
        elif active == "clinical":
            print("→ Clinical mode: back to user input")
            return "user_input"

    # 4️⃣ Default fallback
    print("⚠️ Router: Unhandled state, defaulting to END.")
    return "end"


def run_conversation(compiled_graph, state=None):
    print("=== Conversation Started ===")
    # Use provided state when available (the backend passes in the session state).
    if state is None:
        state = ConversationState()

    current_node = state.get("active_agent", "user_input")

    # Safety: limit iterations to avoid accidental infinite loops in production/dev.
    max_steps = 20
    steps = 0
    while steps < max_steps:
        steps += 1
        # Run the current node
        if current_node == "user_input":
            state = user_input(state)
        elif current_node == "receptionist":
            state = receptionist_node(state)
        elif current_node == "clinical":
            state = clinical_node(state)
        else:
            print("\n🚪 Conversation ended.")
            break

        # Decide next node using your compiled graph logic
        next_node = next_agent(state)
        state["last_agent_call"] = current_node
        print(f"\n➡️  Transition: {current_node} → {next_node}")

        # If the system wants to end
        if next_node == "end" or state.get("end_conversation"):
            print("\n✅ Conversation completed.")
            break

        # If node didn't change, stop to avoid loops
        if next_node == current_node:
            print("→ Node did not change; stopping conversation loop.")
            break

        current_node = next_node

    else:
        print("⚠️ Conversation loop reached max iterations; breaking.")

    # Return the mutated state to callers (backend may ignore return but state is mutated in-place).
    return state


if __name__ == "__main__":
    run_conversation(super_graph())