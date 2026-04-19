import json
from dateutil import parser
from src.state import ClaimState, ExtractedField, DocumentRecord
from src.utils.llm_client import get_groq_client
from src.prompts.templates import EXTRACTION_WORKER_PROMPT
from langsmith import traceable
import re



def _normalize_date(date_str: str) -> str:
    """Attempts to parse any date string into standard YYYY-MM-DD format."""
    try:
        parsed_date = parser.parse(str(date_str), fuzzy=True)
        return parsed_date.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(date_str)

def _normalize_financial(value_str: str) -> str:
    """Attempts to strip currency symbols/commas and return a standard float string."""
    try:
        clean_str = re.sub(r'[^\d.-]', '', str(value_str))
        num = float(clean_str)
        return f"{num:.2f}"
    except ValueError:
        return str(value_str)

@traceable(run_type="llm", name="Worker_8B_Extraction")
def extract_data_from_text(state: ClaimState, filename: str, raw_text: str) -> None:
    client = get_groq_client()
    
    messages = [
        {"role": "system", "content": EXTRACTION_WORKER_PROMPT},
        {"role": "user", "content": f"Filename: {filename}\n\nDocument Text:\n{raw_text}"}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    extracted_json = json.loads(response.choices[0].message.content)
    
    # 1. Update identified documents
    doc_type = extracted_json.get("document_type", "unknown")
    state.documents.identified.append(DocumentRecord(file=filename, type=doc_type))
    
    if doc_type in state.documents.missing:
        state.documents.missing.remove(doc_type)

    # 2. Update extracted fields in the state
    found_data = extracted_json.get("extracted_data", {})
    
    for field_key, field_data in found_data.items():
        if field_data.get("value"):

            raw_value = str(field_data["value"])
            
            # Normalize Dates
            if field_key == "date_of_loss":
                raw_value = _normalize_date(raw_value)
                
            # Normalize Financials
            elif field_key in ["outstanding_loan_balance", "insurance_payout"]:
                raw_value = _normalize_financial(raw_value)
            
            new_field = ExtractedField(
                value=raw_value,
                confidence=field_data.get("confidence", "low"),
                source=filename,
                reason=field_data.get("reason")
            )
            
            if field_key in state.extracted_fields:
                existing = state.extracted_fields[field_key]
                
                if str(existing.value).lower() == str(new_field.value).lower():
                    if filename not in existing.source:
                        existing.source += f", {filename}"
                else:
                    existing.value = f"{existing.value} ({existing.source}) VS {new_field.value} ({filename})"
                    existing.source += f", {filename}"
                    existing.confidence = "low"
            else:
                state.extracted_fields[field_key] = new_field

    state.log_tool_use("extract_data_from_text", {"file": filename}, extracted_json)