# src/tools/document_parser.py
import os
import base64
import fitz
from pypdf import PdfReader
from src.state import ClaimState, IssueRecord
from src.utils.llm_client import get_groq_client
from src.tools.extraction import extract_data_from_text
from langsmith import traceable


def process_unidentified_files(state: ClaimState, folder_path: str) -> None:
    """
    Scans the folder for files that haven't been parsed yet.
    Reads them (via PyPDF, raw text, or Vision OCR) and passes to the extraction tool.
    """
    if not os.path.exists(folder_path):
        return

    identified_filenames = [doc.file for doc in state.documents.identified]
    
    for filename in os.listdir(folder_path):
        # Skip if we already processed this file
        if filename in identified_filenames:
            continue
            
        filepath = os.path.join(folder_path, filename)
        raw_text = ""
        
        try:
            # Route 1: Try Clean PDFs first
            if filename.lower().endswith(".pdf"):
                reader = PdfReader(filepath)
                raw_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

                if len(raw_text.strip()) < 50: 
                    print(f"DEBUG: {filename} appears to be a scanned PDF. Falling back to Vision OCR.")
                    raw_text = _extract_text_from_scanned_pdf(filepath)
                    
            # Route 2: Raw Text / Emails
            elif filename.lower().endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_text = f.read()
                    
            # Route 3: Images (Scans)
            elif filename.lower().endswith((".png", ".jpg", ".jpeg")):
                raw_text = _extract_text_from_image(filepath)
                
            else:
                continue # Unsupported file type
                
            # Pass the transcribed text to the extraction tool
            if raw_text.strip():
                extract_data_from_text(state, filename, raw_text)
                
        except Exception as e:
            state.issues.append(IssueRecord(
                type="invalid",
                description=f"Failed to read file {filename}",
                details=str(e)
            ))

def _extract_text_from_image(filepath: str) -> str:
    """Helper function to read an image file and pass it to the Vision OCR."""
    with open(filepath, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    return _call_groq_vision(base64_image)


def _extract_text_from_scanned_pdf(filepath: str) -> str:
    """Converts a scanned PDF into images and runs them through Groq Vision."""
    doc = fitz.open(filepath)
    full_extracted_text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        
        img_bytes = pix.tobytes("jpeg")
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        page_text = _call_groq_vision(base64_image)
        full_extracted_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
    return full_extracted_text


@traceable(run_type="llm", name="Vision_Model_llama-4-scout-17b")
def _call_groq_vision(base64_image: str) -> str:
    """Single source of truth for the Groq Vision API call."""
    client = get_groq_client()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """You are an expert data transcription system. Your task is to extract all text from this image exactly as written.
                    CRITICAL INSTRUCTIONS:
                    1. Pay strict attention to alphanumeric codes, specifically Vehicle Identification Numbers (VIN). 
                    2. VINs are often 17 characters long and contain a mix of numbers and capitalized letters.
                    3. DO NOT autocorrect or guess. Be extremely careful not to confuse the number '0' with the letter 'O', or the number '1' with the letter 'I'.
                    4. Output the raw text exactly as it appears on the page. Do not format it as JSON, just provide the transcribed text."""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.0
    )
    
    return response.choices[0].message.content