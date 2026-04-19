import streamlit as st
import os
import tempfile
import shutil
from src.state import ClaimState
from src.orchestrator import run_agent_step


st.set_page_config(page_title="AI Claims Processing Agent", layout="wide")
st.title("Auto Claims Processing Portal")

if "claim_state" not in st.session_state:
    st.session_state.claim_state = ClaimState(claim_id="CLM-LIVE-001")
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

with st.sidebar:
    st.header("Upload Claim Documents")

    claim_id_input = st.text_input("Enter Claim ID (e.g., CLM-001)", value="CLM-001")

    if "claim_state" in st.session_state:
        st.session_state.claim_state.claim_id = claim_id_input

    if st.button("Start New Claim"):

        if os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
            
        del st.session_state.claim_state
        del st.session_state.chat_history
        del st.session_state.temp_dir
        st.rerun()

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs, Images, or Text files", 
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "txt"]
    )
    
    if st.button("Process Uploaded Files"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"Saved {len(uploaded_files)} files.")
            
            st.session_state.chat_history.append({
                "role": "system", 
                "content": "New files uploaded. Triggering agent review..."
            })
            
            with st.spinner("Agent is processing documents..."):
                agent_reply = run_agent_step(st.session_state.claim_state, st.session_state.temp_dir)
                if agent_reply:
                    st.session_state.chat_history.append({"role": "assistant", "content": agent_reply})
            st.rerun()

# 4. Main Layout: Chat and Live Debugger
col1, col2 = st.columns([2, 1])

# Column 1: The Chat Interface
with col1:
    st.subheader("Agent Conversation")
    
    # Render the chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            st.info(msg["content"])
        else:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if user_input := st.chat_input("Reply to the agent..."):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        st.session_state.claim_state.llm_history.append({
            "role": "user", 
            "content": f"CUSTOMER REPLIED: {user_input}"
        })
        
        with st.spinner("Agent is analyzing your response..."):
            agent_reply = run_agent_step(st.session_state.claim_state, st.session_state.temp_dir)
            if agent_reply:
                st.session_state.chat_history.append({"role": "assistant", "content": agent_reply})
        
        st.rerun()

with col2:
    st.subheader("Live Claim State")
    st.json(st.session_state.claim_state.model_dump(exclude={"llm_history"}))