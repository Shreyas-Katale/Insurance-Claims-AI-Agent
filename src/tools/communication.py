from src.state import ClaimState
from src.utils.llm_client import get_groq_client
from src.prompts.templates import CUSTOMER_MESSAGE_PROMPT
from langsmith import traceable

@traceable(run_type="llm", name="Worker_8B_Draft_Message")
def draft_customer_message(state: ClaimState) -> str:
    client = get_groq_client()
    
    missing_docs = state.documents.missing
    active_issues = [issue if isinstance(issue, dict) else issue.model_dump() for issue in state.issues]
    
    # NEW: Grab the last 3-4 messages from the chat history to provide context
    recent_chat = "\n".join([f"{msg['role'].upper()}: {msg['content']}" 
                             for msg in state.llm_history[-4:] 
                             if msg['role'] in ['user', 'assistant']])
    
    context = f"Missing Documents: {', '.join(missing_docs)}\n\nActive Issues:\n"
    for issue in active_issues:
        context += f"- {issue['description']} (Details: {issue.get('details', 'N/A')})\n"
        
    # Inject the chat history into the context
    if recent_chat:
        context += f"\n\nRecent Conversation Context:\n{recent_chat}"
        
    messages = [
        {"role": "system", "content": CUSTOMER_MESSAGE_PROMPT},
        {"role": "user", "content": context}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.5 
    )
    
    message = response.choices[0].message.content
    state.log_tool_use("draft_customer_message", context, message)
    
    return message