import pymupdf4llm
import os
from app.config import settings

def convert_pdf_to_markdown(pdf_path: str, output_path: str):
    """
    Reads a PDF and converts it to Markdown format.
    """
    print(f"Reading {pdf_path}...")
    
    # 1. The Magic: Convert the PDF directly to a Markdown string
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # 2. Save that string into a new text file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)
        
    print(f"Success! Saved to {output_path}")

# 3. A quick way to test this specific file from the terminal
if __name__ == "__main__":
    
    
    # Calculate the exact root of the project
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    
    # Build the exact paths using our config!
    input_pdf = os.path.join(PROJECT_ROOT, settings.RAW_PDF_DIR, "sample.pdf")
    output_md = os.path.join(PROJECT_ROOT, settings.MARKDOWN_DIR, "sample.md")
    
    if os.path.exists(input_pdf):
        convert_pdf_to_markdown(input_pdf, output_md)
    else:
        print(f"Error: Could not find {input_pdf}. Please make sure sample.pdf is in data/raw/")