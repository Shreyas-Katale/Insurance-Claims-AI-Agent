import json
from src.state import ClaimState, IssueRecord
from src.utils.llm_client import get_groq_client
from src.prompts.templates import VALIDATION_WORKER_PROMPT
from langsmith import traceable

@traceable(run_type="llm", name="Worker_8B_Validation")
def validate_claim_data(state: ClaimState) -> None:
    """
    Reviews all extracted fields and documents to find inconsistencies.
    """
    client = get_groq_client()

    if "vin" in state.extracted_fields:
        vin_val = str(state.extracted_fields["vin"].value).strip()
        
        # Skip the strict length check if it's a known conflict string
        if " VS " not in vin_val:
            if len(vin_val) != 17 or not vin_val.isalnum():
                state.issues.append(IssueRecord(
                    type="invalid",
                    description="VIN is not exactly 17 alphanumeric characters.",
                    details=f"Extracted VIN: {vin_val}"
                ))
                state.extracted_fields["vin"].confidence = "low"

    # LLM Consistency Checks (Comparing values across documents)
    state_subset = {
        "extracted_fields": {k: v.model_dump() for k, v in state.extracted_fields.items()},
        "documents": state.documents.model_dump()
    }
    
    messages = [
        {"role": "system", "content": VALIDATION_WORKER_PROMPT},
        {"role": "user", "content": f"Review this extracted claim data for conflicts:\n\n{json.dumps(state_subset, indent=2)}"}
    ]
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    validation_result = json.loads(response.choices[0].message.content)
    
    if not validation_result.get("is_consistent", True):
        for issue in validation_result.get("issues", []):
            state.issues.append(IssueRecord(**issue))
            
    state.log_tool_use("validate_claim_data", state_subset, validation_result)