# from fastapi import FastAPI, File, UploadFile
# from typing import List
import fitz  # PyMuPDF

# app = FastAPI()


# def extract_skills_from_pdf(pdf_content) -> List[str]:
#     pdf_path = pdf_content
#
#     # Open the PDF
#     doc = fitz.open(pdf_path)
#
#     # Initialize an empty string to hold text
#     text = ""
#
#     # Extract text from each page
#     for page in doc:
#         text += page.get_text()
#
#     # Close the document
#     doc.close()
#
#     # Output the extracted text for analysis
#
#     # Dummy function to extract skills from PDF content.
#     # Replace this with your actual logic or integration with an external API.
#     return ["Python", "FastAPI", "SQL"]  # Example skills
#
#
# @app.post("/upload-resume/")
# async def upload_resume(resume: UploadFile = File(...)):
#     if resume.content_type != "application/pdf":
#         return {"error": "File must be a PDF."}
#     try:
#         pdf_content = await resume.read()
#         skills = extract_skills_from_pdf(pdf_content)
#         return {"skills": skills}
#     except Exception as e:
#         return {"error": str(e)}

import fitz  # PyMuPDF
import re

# Open the PDF
pdf_path = 'your_pdf_file.pdf'
doc = fitz.open(pdf_path)

# Extract text from each page
text = ""
for page in doc:
    text += page.get_text()

# Close the document
doc.close()

# Define a regular expression pattern for GitHub URLs
github_url_pattern = r'https://github\.com/[a-zA-Z0-9_-]+'

# Search for all occurrences of the pattern
github_urls = re.findall(github_url_pattern, text)

print("Found GitHub URLs:", github_urls)

