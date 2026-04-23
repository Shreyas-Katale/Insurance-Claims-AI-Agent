import os
import json
from typing import Literal, Dict, Any, Optional
from pydantic import BaseModel, Field
from langsmith import traceable

from src.state import ClaimState, NextAction
from src.prompts.templates import SUPERVISOR_SYSTEM_PROMPT
from src.utils.llm_client import get_groq_client

from src.tools.document_parser import process_unidentified_files
from src.tools.validation import validate_claim_data
from src.tools.communication import draft_customer_message


class OrchestratorDecision(BaseModel):
    """The strict JSON format the Supervisor LLM must return on every turn."""
    type: Literal["tool_call", "message_customer", "final_decision"]
    tool_name: Optional[str] = Field(None, description="Name of the tool to call")
    arguments: Optional[Dict[str, Any]] = Field(None, description="Arguments for the tool")
    message: Optional[str] = Field(None, description="Message to the customer")
    status: Optional[Literal["complete", "incomplete", "needs_review"]] = Field(None, description="Final status")
    reasoning: str = Field(..., description="Brief explanation of why this decision was made")


@traceable(run_type="llm", name="Supervisor_70B_Decision")
def get_orchestrator_decision(state: ClaimState, folder_path: str) -> OrchestratorDecision: 
    client = get_groq_client()
    
    if os.path.exists(folder_path):
        all_files = os.listdir(folder_path)
    else:
        all_files = []
        
    identified_files = [doc.file for doc in state.documents.identified]
    unprocessed_files = [f for f in all_files if f not in identified_files]
    
    messages = [{"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT}]
    messages.extend(state.llm_history)
    
    current_state_json = state.model_dump_json(exclude={"llm_history"}, indent=2)

    messages.append({
        "role": "user",
        "content": f"CURRENT CLAIM STATE:\n{current_state_json}\n\nUNPROCESSED FILES WAITING IN FOLDER: {unprocessed_files}\n\nBased on this state and the waiting files, what is your next decision?"
    })

    # 2. Call Groq with JSON mode enforced
    response = client.chat.completions.create(
        # model="llama-3.3-70b-versatile",
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.1, 
        response_format={"type": "json_object"},
    )
    
    # 3. Parse the JSON response
    raw_response = response.choices[0].message.content
    decision = OrchestratorDecision.model_validate_json(raw_response)
    
    # Log the orchestrator's thought process so the Streamlit UI can show it if needed
    state.llm_history.append({"role": "assistant", "content": raw_response})
    
    return decision


@traceable(run_type="chain", name="ReAct_Loop")
def run_agent_step(state: ClaimState, folder_path: str) -> str:
    """
    Executes the ReAct loop, delegating to Python worker tools.
    Returns a string message to Streamlit when it needs to pause or finishes.
    """
    max_loops = 10 
    loop_count = 0
    
    while loop_count < max_loops:
        loop_count += 1
        
        decision = get_orchestrator_decision(state, folder_path)
        print(f"DEBUG: Agent decided to -> {decision.type} (Reason: {decision.reasoning})")
        
        # ROUTE 1: TOOL EXECUTION (Background tasks)
        if decision.type == "tool_call":
            
            if decision.tool_name == "process_unidentified_files":
                process_unidentified_files(state, folder_path)
                state.llm_history.append({"role": "user", "content": "Tool 'process_unidentified_files' executed successfully. Check the updated state."})
                
            elif decision.tool_name == "validate_claim_data":
                validate_claim_data(state)
                state.llm_history.append({"role": "user", "content": "Tool 'validate_claim_data' executed successfully. Check issues and fields in the state."})
                
            else:
                state.llm_history.append({"role": "user", "content": f"Error: Tool '{decision.tool_name}' does not exist. Use 'process_unidentified_files' or 'validate_claim_data'."})


        # ROUTE 2: PAUSE FOR CUSTOMER INTERACTION
        elif decision.type == "message_customer":
            
            if decision.status:
                state.status = decision.status
            else:
                state.status = "incomplete"
            
            # Call our specialized 8B communication tool to write a polite message
            drafted_message = draft_customer_message(state)
            
            state.next_action = NextAction(type="message_customer", message=drafted_message)
            
            state.llm_history.append({"role": "assistant", "content": f"SENT TO CUSTOMER: {drafted_message}"})
            os.makedirs("output", exist_ok=True)
            output_file = os.path.join("output", f"{state.claim_id}.json")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(exclude={"llm_history"}, indent=2))
            
            return drafted_message 


        # ROUTE 3: FINALIZE AND SAVE
        elif decision.type == "final_decision":
            if decision.status:
                state.status = decision.status
            else:
                state.status = "incomplete"
        
            final_msg = decision.message if decision.message else "Claim evaluation complete."
            
            state.next_action = NextAction(type="finalize", message=final_msg)
            
            os.makedirs("output", exist_ok=True)
            output_file = os.path.join("output", f"{state.claim_id}.json")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(state.model_dump_json(exclude={"llm_history"}, indent=2))
                
            # Return the natural message, plus a system confirmation that it saved successfully.
            return f"{final_msg}\n\n*System Note: Session closed. Claim saved to `{output_file}` as **{state.status.upper()}**.*"

    return "Agent paused: Reached maximum internal loops to prevent runaway API costs."