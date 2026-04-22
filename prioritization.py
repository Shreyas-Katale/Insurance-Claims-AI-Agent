import os
import json
from src.utils.llm_client import get_groq_client

def generate_processing_order():
    output_dir = "output"
    
    # 1. Read all processed claims from the output folder
    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' not found. Please process claims in the UI first.")
        return
        
    claim_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    
    if not claim_files:
        print(f"No processed claims found in the '{output_dir}' directory.")
        return

    claims_data = []
    for file in claim_files:
        try:
            with open(os.path.join(output_dir, file), "r", encoding="utf-8") as f:
                claims_data.append(json.load(f))
        except Exception as e:
            print(f"Warning: Could not read {file}. Error: {e}")

    prompt = """You are an AI Claims Manager. Review the following processed insurance claims.
    
    Your task is to determine the optimal processing order for the human claims team. Use the following logic to rank the backlog:
    
    1. Status Hierarchy: 'needs_review' claims take priority over 'incomplete' claims (human adjusters can actively resolve conflicts, whereas 'incomplete' claims are blocked waiting on the customer).
    2. Financial Exposure (High Priority): For claims with the same status, prioritize those with the highest extracted 'insurance_payout' or 'outstanding_loan_balance'. High-dollar liabilities must be triaged first.
    3. Issue Severity: Prioritize critical legal conflicts (e.g., VIN mismatches, which indicate potential fraud) over minor administrative conflicts (e.g., slight math discrepancies).
    4. Proximity to Finish: For 'incomplete' claims, prioritize claims missing only 1 document over claims missing multiple documents.
    
    You MUST respond ONLY with a valid JSON object matching this exact structure:
    {
      "processing_order": [
        {
          "claim_id": "CLM-001", 
          "status": "needs_review",
          "reason": "Highest financial exposure ($35k payout) and contains a critical VIN mismatch requiring immediate fraud review."
        }
      ]
    }
    
    Here is the raw data for the claims to evaluate:
    """
    prompt += json.dumps(claims_data, indent=2)

    # 3. Call Groq
    print(f"Analyzing {len(claims_data)} claims for prioritization...")
    try:
        client = get_groq_client()
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        # 4. Save and print the final output
        final_json = response.choices[0].message.content
        
        with open("prioritization_report.json", "w", encoding="utf-8") as f:
            f.write(final_json)
            
        print("\n Prioritization Complete! Saved to 'prioritization_report.json':")
        print(final_json)
        
    except Exception as e:
        print(f"An error occurred while calling the LLM: {e}")

if __name__ == "__main__":
    generate_processing_order()