# Insurance-Claims-AI-Agent

This is a production-grade AI agent that automates vehicle total-loss insurance claims. Powered by Groq and a fully open-source LLM stack, it handles everything from document processing and data extraction to validation, and interactively prompts the user if a document is missing or if there are inconsistencies in the uploaded files.

---

## Repository Structure

```text
├── app.py                  # Main Streamlit user interface
├── prioritize.py           # Standalone script for prioritizing claims
├── requirements.txt        # Dependencies
├── .env           # Template for environment variables
├── src/
│   ├── orchestrator.py     # 70B Supervisor routing logic
│   ├── state.py            # Pydantic schemas (ClaimState)
│   ├── utils/
│   │   └── llm_client.py   # Groq and LangSmith setup
│   ├── tools/
│   │   ├── document_parser.py  # PDF and Vision OCR logic
│   │   ├── extraction.py       # 8B data extraction and conflict resolution
│   │   ├── validation.py       # 8B logic checking
│   │   └── communication.py    # 8B customer message drafting
```

## Approach and Architecture

The system is built on a **Stateful Supervisor-Worker Architecture**, similar to the LangGraph paradigm but implemented efficiently via Streamlit's session state and strict Pydantic schemas.

### The Open-Source Advantage

I deliberately designed this architecture to rely *exclusively* on open-source foundation models (the Meta Llama family). 

* **Data Privacy & Security:** Insurance documents are filled with Personally Identifiable Information (PII) and financial data. Open-source models provide a clear path to self-hosting and strict data governance, bypassing the privacy risks associated with closed-source APIs (like OpenAI or Anthropic).

* **Cost Efficiency:** Running bounded tasks (like extraction and routing) on highly efficient open-source models dramatically reduces inference costs without sacrificing reasoning quality.

* **No Vendor Lock-In:** Ensures the system remains flexible, customizable, and immune to unexpected API pricing changes or deprecations from proprietary providers.

### The Flow

1. **State Initialization:** A `ClaimState` object (defined via Pydantic) acts as the single source of truth, tracking identified documents, missing documents, extracted fields, issues, and conversation history.

2. **The Orchestrator (Supervisor):** A 70B open-source LLM evaluates the `ClaimState` and the physical files in the upload directory to route to the next logical action: reading a file, checking for conflicts, messaging the user, or finalizing.

3. **The Workers (Tools):** Specialized 8B and 17B open-source models act as functions. They receive a specific slice of data, perform their task (e.g., extract entities, validate logic), mutate the `ClaimState`, and return control to the Orchestrator.

### Tech Stack

* **Framework:** Streamlit (UI & State Management), Python 3.10+

* **Models (Fully Open-Source via Groq):** * `llama-3.3-70b-versatile` (Orchestrator)
  * `llama-3.1-8b-instant` (Extraction/Validation/Drafting)
  * `meta-llama/llama-4-scout-17b-16e-instruct` (Vision/OCR)

* **Data Validation:** Pydantic V2

* **Document Parsing:** `PyPDF` (Text extraction), `PyMuPDF / fitz` (Scan-to-image conversion)

* **Observability:** LangSmith (Tracing agent chains and worker LLM calls)

---

## Tool Design Rationale

I intentionally separated the agent's capabilities into distinct, modular tools rather than relying on a single massive prompt. 

* **Why Separate Extraction and Validation?** By separating extraction (pulling raw numbers) from validation (comparing those numbers), we dramatically reduce hallucinations. The Validator only sees a clean dictionary of extracted fields, not the noisy 10-page document, allowing it to strictly focus on logical conflicts.

* **Why Use Smaller 8B Models for Workers?** Extraction and message drafting are well-defined, bounded tasks. Using Groq's open-source `llama-3.1-8b-instant` ensures sub-second latency and minimal token costs, reserving the heavy, "expensive" reasoning for the 70B Orchestrator.

* **Why the Multi-Modal Vision Fallback?** Vision models are slower and costlier than standard text extraction. The Document Parser tool is designed to attempt standard `PyPDF` text extraction first. Only if the document yields fewer than 50 characters (indicating a flattened scan or image) does it trigger the powerful open-source `meta-llama/llama-4-scout-17b-16e-instruct` Vision model. This dynamic routing optimizes for both speed and accuracy while maintaining data privacy.

---


## Monitoring & Debugging

Because non-deterministic LLM pipelines can fail in unpredictable ways (e.g., hallucinating a date vs. failing to OCR a blurry image), robust monitoring is built directly into the architecture.

* **LangSmith Tracing:** Every execution is instrumented using LangSmith. The Orchestrator loop is tracked as a global "chain," while individual tool executions are tracked as sub-runs. This allows developers to audit the exact prompt, raw OCR text payload, and latency of every single LLM call to immediately pinpoint where a breakdown occurred.

* **State Logging:** The ClaimState Pydantic object acts as an auditable memory bank. It tracks a tools_used array that logs exactly which document was parsed, what JSON was extracted, and the LLM's confidence score.

* **Terminal Debugging:** Real-time console prints are implemented at the Orchestrator routing level, allowing the developer to see the exact reasoning the 70B model used to make its tool selection in real-time.

---


## Use Cases Handled & Real-World Scenarios

This agent is built to handle the "messy reality" of insurance processing, not just clean happy-paths.

* **Multi-Format Ingestion:** Real-world claims don't just contain digital PDFs. The agent seamlessly handles clean digital documents (e.g., official settlement breakdowns), scanned image files, and even handwritten field adjuster notes.

* **Fraud & Error Detection (Data Conflicts):** If a customer uploads a finance agreement with a typo in the VIN (e.g., ending in `182` instead of `127`), the agent catches the mismatch across documents, flags the severity, and sets the claim to `needs_review` to prevent a fraudulent or erroneous payout.

* **Asynchronous Customer Communication:** Real people don't always have their documents ready. If the agent asks for a police report and the customer replies, *"I don't have it, can I submit it next week?"*, the agent is conversationally intelligent enough to acknowledge the timeline, pause the claim, and gracefully close the session instead of getting stuck in an infinite loop.

* **Backlog Triage & Prioritization:** Using the `prioritize.py` engine, the system intelligently ranks a backlog of unresolved claims. It prioritizes high financial exposure (e.g., $30,000 payouts) and critical severity (VIN mismatches) over minor missing files, mimicking how a human manager assigns work to adjusters.

---


## Key Decisions and Tradeoffs

Building a resilient AI system requires knowing when *not* to trust the LLM. 

### 1. Mitigating "Data-Loss Pipelines" (The Collision Problem)

* **The Challenge:** Initially, if the 8B extractor found two different VINs across two documents, the dictionary silently overwrote the first one. The Validator never saw the conflict and hallucinated fake issues to compensate.

* **The Decision:** Refactored the extraction tool to detect key collisions. If values mismatch, it concatenates them (e.g., `123 (doc_A) VS 456 (doc_B)`) and forces the confidence score to `low`. This guaranteed the Validator had the exact context needed to catch the discrepancy.

### 2. State Grounding via OS File Injection

* **The Challenge:** The Orchestrator LLM acts like a "brain in a jar." Early on, it repeatedly asked the user for documents that were already sitting in the temp folder because it only read the `ClaimState` JSON (which listed them as missing).

* **The Decision:** Injected the literal output of `os.listdir()` directly into the Orchestrator's system prompt at runtime. This bridged the gap between the server's physical file system and the LLM's virtual context, completely eliminating redundant requests.

### 3. The "Amnesiac Worker" & Stateful Context

* **The Challenge:** The Communication Worker lacked chat history, leading to generic "Please upload the file" loops even when users asked for timeline extensions.

* **The Decision:** Injected a rolling window of the last 4 chat messages into the worker's drafting prompt and added a strict `DELAY RULE` to the Orchestrator. This allows the system to officially pause the claim and persist the incomplete state without deadlocking.

### 4. Deterministic vs. Probabilistic Logic

* **The Challenge:** The 8B extraction model pulled dates exactly as written (`03/05/2026` vs `2026-03-05`), causing the validation model to flag false-positive conflicts.

* **The Decision:** Used Python's deterministic `python-dateutil` library to intercept the LLM's output and force a strict `YYYY-MM-DD` normalization *before* saving to the state. **Rule of thumb:** Never trust an LLM to format data when standard Python libraries can do it flawlessly.

---

## What I'd Do With More Time

If I were to expand this into a production-ready enterprise application, I would implement the following:

1. **Human-in-the-Loop (HITL) Dashboard:** Build a dedicated Streamlit view for the Claims Adjuster. When a claim is marked `needs_review`, the UI would highlight the exact conflicting text snippets side-by-side (e.g., showing the cropped image of the VIN on the finance agreement next to the police report) with a button to manually override the extracted value.

2. **Parallel Tool Execution:** Currently, the system loops through unstructured files sequentially. I would use `asyncio` to parallelize the extraction worker, allowing the system to process a 10-document folder concurrently, dramatically reducing total processing time.

3. **Database Integration:** Replace the local JSON `output/` folder strategy with a robust PostgreSQL database using SQLAlchemy. `ClaimState` would map to relational tables, making the prioritization engine a fast SQL query rather than an LLM prompt.

4. **RAG for Complex Policy Rules:** Auto insurance rules vary wildly by state (e.g., GAP coverage laws). I would implement a Vector DB (like Pinecone) containing state-specific insurance regulations so the validation tool could check if a "Gap Amount" is actually covered by the user's specific policy rider.


## UI Screenshot

![Streamlit UI Screenshot](assets/UI%20Screenshot.png)

---