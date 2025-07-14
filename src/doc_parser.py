from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv()

client = genai.Client()

def generate_html_from_pdf(document_path: str):
    """
    Convert PDF to HTML using Gemini AI.
    
    Args:
        document_path: Path to the PDF file
    """
    prompt = "Parse this PDF document into clean HTML format. Preserve the structure, headings, paragraphs, lists, and tables. Use semantic HTML tags like <h1>, <h2>, <p>, <ul>, <ol>, <table>, etc. Do not include any styling or CSS."
    
    # Upload the file to the client
    file_path = Path(document_path)
    file = client.files.upload(file=file_path)
    
    # Generate content with the PDF file
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[file, prompt]
    )

    try:
        # Extract filename without extension
        file_name = Path(document_path).stem
    except Exception as e:
        print(f"Error extracting filename: {e}")
        # Use the document path as the file name
    file_name = document_path.split("/")[-1]
    
    # Ensure the HTML output directory exists
    html_dir = Path("documents/processed/HTML")
    html_dir.mkdir(parents=True, exist_ok=True)

    # Save the HTML output
    html_path = html_dir / f"{file_name}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print(f"Generated HTML from {document_path} -> {html_path}")