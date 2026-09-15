import streamlit as st
import os
import json
from openai import OpenAI
import psycopg2
from psycopg2.extras import Json
import requests

# --- Configuration & Setup ---
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3-27b")
DATABASE_URL = os.getenv("DATABASE_URL", "")
EXPECTED_USER = os.getenv("STREAMLIT_USERNAME", "admin")
EXPECTED_PASS = os.getenv("STREAMLIT_PASSWORD", "admin")

st.set_page_config(page_title="Chat - Qwen 27b ", page_icon="🤖", layout="wide")

# --- Authentication ---
def check_password():
    """Returns True if the user has entered the correct password."""
    def password_entered():
        if (st.session_state["username"] == EXPECTED_USER and 
            st.session_state["password"] == EXPECTED_PASS):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Username", key="username")
    st.text_input("Password", type="password", key="password")
    st.button("Login", on_click=password_entered)

    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()  # Stop execution until logged in

# --- Database Setup (Neon PostgreSQL) ---
@st.cache_resource
def init_db():
    if not DATABASE_URL:
        return None
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    messages JSONB NOT NULL
                );
            """)
        return conn
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")
        return None

db_conn = init_db()

def save_chat_history():
    if not db_conn or not st.session_state.messages:
        return
    try:
        with db_conn.cursor() as cur:
            # Simple approach: Upsert based on session_id (which we will define per chat)
            # For simplicity, we just insert a new record for this session if it doesn't exist, 
            # or update the existing one.
            session_id = st.session_state.get("session_id", "default_session")
            cur.execute("SELECT id FROM conversations WHERE session_id = %s", (session_id,))
            record = cur.fetchone()
            if record:
                cur.execute(
                    "UPDATE conversations SET messages = %s WHERE session_id = %s",
                    (Json(st.session_state.messages), session_id)
                )
            else:
                cur.execute(
                    "INSERT INTO conversations (session_id, mode, messages) VALUES (%s, %s, %s)",
                    (session_id, st.session_state.mode, Json(st.session_state.messages))
                )
    except Exception as e:
        st.sidebar.error(f"Save Error: {e}")

# --- Initialize OpenAI Client ---
client = OpenAI(
    api_key="EMPTY",  # Ollama doesn't require a real API key
    base_url=OLLAMA_API_BASE
)

# --- Sidebar & State ---
with st.sidebar:
    st.title("🤖 Qwen Swarm")
    
    # Status Check
    st.markdown("### System Status")
    try:
        res = requests.get(f"{OLLAMA_API_BASE}/models", timeout=2)
        if res.status_code == 200:
            models = res.json().get("data", [])
            served_model = models[0]["id"] if models else "Unknown"
            st.success(f"Ollama Online\nModel: {served_model}")
        else:
            st.warning("Ollama starting or downloading model...")
    except requests.exceptions.RequestException:
        st.error("Ollama Offline or Loading...")

    st.markdown("---")
    
    # Mode Selection
    mode = st.radio("Select Mode:", ("Standard Chat", "Agent Swarm"), key="mode_selector")
    
    # Session Management
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    
    if "mode" not in st.session_state or st.session_state.mode != mode:
        st.session_state.mode = mode
        st.session_state.messages = []
        
    if st.button("Clear Chat"):
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # DB History (Optional)
    if db_conn:
        st.markdown("---")
        with st.expander("📜 History"):
            with db_conn.cursor() as cur:
                cur.execute("SELECT session_id, created_at, mode FROM conversations ORDER BY created_at DESC LIMIT 10")
                history = cur.fetchall()
                for sid, created_at, h_mode in history:
                    if st.button(f"{h_mode} ({created_at.strftime('%m-%d %H:%M')})", key=sid):
                        cur.execute("SELECT messages FROM conversations WHERE session_id = %s", (sid,))
                        st.session_state.messages = cur.fetchone()[0]
                        st.session_state.session_id = sid
                        st.session_state.mode = h_mode
                        st.rerun()

# --- Main App ---
st.title(f"Qwen3.8-27B - {st.session_state.mode}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Logic ---
if prompt := st.chat_input("What would you like to ask?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if st.session_state.mode == "Standard Chat":
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                # Thinking disabled for speed in standard chat
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.messages,
                    max_tokens=4096,
                    temperature=0.7,
                    stream=True
                )
                for chunk in response:
                    delta = chunk.choices[0].delta
                    
                    # Extract reasoning if the server separated it
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning is None and hasattr(delta, "model_extra") and delta.model_extra:
                        reasoning = delta.model_extra.get("reasoning_content")
                    
                    # If there's reasoning content directly, add the tags manually so the UI formatting catches it
                    if reasoning:
                        if "<think>" not in full_response:
                            full_response += "<think>"
                        full_response += reasoning
                        
                    if delta.content is not None:
                        # If the server finished reasoning and is now giving content, close the think tag if we opened it
                        if "<think>" in full_response and "</think>" not in full_response and not reasoning:
                             full_response += "</think>\n\n"
                        full_response += delta.content

                    display_text = full_response.replace("<think>", "💭 **Thinking...**\n```text\n").replace("</think>", "\n```\n\n")
                    message_placeholder.markdown(display_text + "▌")
                
                final_display_text = full_response.replace("<think>", "💭 **Thinking...**\n```text\n").replace("</think>", "\n```\n\n")
                message_placeholder.markdown(final_display_text)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_chat_history()

        elif st.session_state.mode == "Agent Swarm":
            with st.chat_message("assistant"):
                st.markdown("**Swarm execution started...**")
                
                # --- Agent 1: Planner ---
                with st.expander("🧠 Planner Output (Agent 1)", expanded=True):
                    st.markdown("**Agent 1 (Planner) is generating a plan...**")
                    planner_placeholder = st.empty()
                    
                    planner_messages = [
                        {"role": "system", "content": "You are a planning agent. Decompose the user's request into a numbered list of clear, actionable steps for the synthesizer to execute. Do not answer the question directly, just provide the plan."},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Use xhigh reasoning effort
                    planner_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=planner_messages,
                        max_tokens=2048,
                        temperature=0.7,
                        stream=True
                    )
                    
                    plan = ""
                    for chunk in planner_response:
                        if chunk.choices[0].delta.content is not None:
                            plan += chunk.choices[0].delta.content
                            display_plan = plan.replace("<think>", "💭 **Thinking...**\n```text\n").replace("</think>", "\n```\n\n")
                            planner_placeholder.markdown(display_plan + "▌")
                    
                    final_display_plan = plan.replace("<think>", "💭 **Thinking...**\n```text\n").replace("</think>", "\n```\n\n")
                    planner_placeholder.markdown(final_display_plan)

                # --- Agent 2: Synthesizer ---
                with st.spinner("Agent 2 (Synthesizer) is generating the final response..."):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    synth_messages = [
                        {"role": "system", "content": "You are the synthesizer agent. Use the provided execution plan to address the user's original request. Provide a complete, polished, and direct answer."},
                        {"role": "user", "content": f"User Request: {prompt}\n\nExecution Plan to follow:\n{plan}"}
                    ]
                    
                    # Disable thinking for Synthesizer for lower latency
                    synth_response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=synth_messages,
                        max_tokens=4096,
                        temperature=0.5,
                        stream=True
                    )
                    
                    for chunk in synth_response:
                        delta = chunk.choices[0].delta
                        
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning is None and hasattr(delta, "model_extra") and delta.model_extra:
                            reasoning = delta.model_extra.get("reasoning_content")
                            
                        if reasoning:
                            if "<think>" not in full_response:
                                full_response += "<think>"
                            full_response += reasoning
                            
                        if delta.content is not None:
                            if "<think>" in full_response and "</think>" not in full_response and not reasoning:
                                full_response += "</think>\n\n"
                            full_response += delta.content
                            
                        display_text = full_response.replace("<think>", "💭 **Thinking...**\n```text\n").replace("</think>", "\n```\n\n")
                        message_placeholder.markdown(display_text + "▌")
                    
                    final_display_text = full_response.replace("<think>", "💭 **Thinking...**\n```text\n").replace("</think>", "\n```\n\n")
                    message_placeholder.markdown(final_display_text)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_chat_history()

    except Exception as e:
        error_msg = str(e)
        if "CUDA out of memory" in error_msg:
            st.error("🚨 **CUDA Out of Memory:** The context window is too large or the GPU is overloaded. Try clearing the chat.")
        elif "Connection error" in error_msg or "timeout" in error_msg.lower():
            st.error("🚨 **Connection Error:** Could not connect to vLLM. Make sure the server is fully booted.")
        else:
            st.error(f"🚨 **Error:** {error_msg}")
