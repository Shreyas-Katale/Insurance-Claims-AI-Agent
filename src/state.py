from typing import Any, Literal
from pydantic import BaseModel, Field

class ExtractedField(BaseModel):
    """Represents a single piece of extracted data with confidence scoring."""
    value: Any
    confidence: Literal["high", "medium", "low"]
    source: str
    reason: str | None = None

class DocumentRecord(BaseModel):
    """Tracks documents that have been successfully identified."""
    file: str
    type: str

class DocumentState(BaseModel):
    """Maintains the status of all documents in the claim folder."""
    identified: list[DocumentRecord] = Field(default_factory=list)
    missing: list[str] = Field(
        default_factory=lambda: ["police_report", "finance_agreement", "settlement_breakdown"]
    )
    duplicates: list[str] = Field(default_factory=list)

class IssueRecord(BaseModel):
    """Logs conflicts, missing data, or low-confidence extractions."""
    type: Literal["inconsistency", "missing_document", "invalid", "low_confidence", "policy_limit_exposure"]
    description: str
    details: str | None = None

class NextAction(BaseModel):
    """The immediate next step the Orchestrator intends to take."""
    type: Literal["finalize", "message_customer", "escalate", "continue"]
    message: str | None = None

class ClaimState(BaseModel):
    """
    The central memory object for the ReAct Agent. 
    This schema dictates exactly how the final JSON output will be structured.
    """
    claim_id: str
    status: Literal["processing", "complete", "incomplete", "needs_review"] = "processing"
    
    extracted_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    documents: DocumentState = Field(default_factory=DocumentState)
    issues: list[IssueRecord] = Field(default_factory=list)
    next_action: NextAction | None = None
    tools_used: list[dict[str, Any]] = Field(default_factory=list)
    
    # Internal agent memory - EXCLUDED from the final JSON output
    llm_history: list[dict[str, str]] = Field(default_factory=list, exclude=True)

    def log_tool_use(self, tool_name: str, input_data: Any, result: Any) -> None:
        """Helper method to keep the tool execution log clean."""
        self.tools_used.append({
            "tool": tool_name,
            "input": input_data,
            "result": result
        })