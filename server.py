from io import StringIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os

from pydantic import BaseModel
from llm_logic import create_resume_latex, create_cover_letter, get_jobs_from_api, get_top_4_jobs  
from fastapi.responses import FileResponse
import tempfile
from pylatex import Document, Package
import subprocess
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
from unstructured.partition.auto import partition

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# @app.get("/upload-pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         # Create a temporary file to store the uploaded PDF
#         with open(f"temp_{file.filename}", "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Process the PDF
#         result = process_pdf(f"temp_{file.filename}")
#         print(result)
        
#         # Remove the temporary file
#         os.remove(f"temp_{file.filename}")
        
#         return JSONResponse(content={"message": "PDF processed successfully", "result": result}, status_code=200)
#     except Exception as e:
#         return JSONResponse(content={"message": f"Error processing PDF: {str(e)}"}, status_code=500)
class ResumeProcessingResponse(BaseModel):
    resume_text: str
    job_description: Optional[str] = None

@app.post("/process-resume/")
async def process_resume(resume_data: ResumeProcessingResponse):
    try:
        resume_text = resume_data.resume_text
        job_description = resume_data.job_description
        # Call the get_similarity function with the resume text
        if job_description is None:
            jobs_titles, experience = get_top_4_jobs(resume_text=resume_text)  # Assuming 'jobs' is available
        else:   
            optimized_resume, latex = create_resume_latex(resume_text=resume_text, job_description=job_description)  # Assuming 'jobs' is available
            cover_letter = create_cover_letter(resume_text=optimized_resume, job_description=job_description)
            jobs_titles, experience = get_top_4_jobs(resume_text=optimized_resume)



        return JSONResponse(content={
            "message": "Resume processed successfully",
            "optimized_resume": optimized_resume if job_description else None, 
            "latex": latex if job_description else None, 
            "cover_letter": cover_letter if job_description else None, 
            "jobs_titles": jobs_titles, 
            "experience": experience}, 
        status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")



@app.get("/download-resume-pdf/")
async def download_resume_pdf(latex_content: str, name: str):
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a temporary .tex file
            tex_file_path = os.path.join(tmpdir, f"resume_{name}.tex")
            with open(tex_file_path, "w") as tex_file:
                tex_file.write(latex_content)
            
            # Compile the LaTeX file to PDF
            subprocess.run(["pdflatex", "-output-directory", tmpdir, tex_file_path], check=True)
            
            # Path to the generated PDF
            pdf_file_path = os.path.join(tmpdir, f"resume_{name}.pdf")
                
            # Check if the PDF was created successfully
            if not os.path.exists(pdf_file_path):
                raise HTTPException(status_code=500, detail="Failed to generate PDF")
            
            # Return the PDF file
            return FileResponse(pdf_file_path, media_type="application/pdf", filename=f"resume_{name}.pdf")
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error compiling LaTeX: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

def handle_nan_values(obj):
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, float) and (np.isinf(obj) or obj == float('inf') or obj == float('-inf')):
        return str(obj)  # Convert infinity to string
    return obj

def custom_json_serializer(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj) if not np.isnan(obj) and not np.isinf(obj) else None
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def clean_dataframe(df):
    def clean_value(val):
        if isinstance(val, (date, datetime)):
            return val.isoformat()
        elif pd.isna(val) or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return None
        elif isinstance(val, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(val)
        elif isinstance(val, (np.float_, np.float16, np.float32, np.float64)):
            return float(val)
        return val

    return df.applymap(clean_value)

@app.get("/get_jobs/")
def get_jobs(job_title: str, city: str, country: str, total_jobs: int):
    try:
        # Get the DataFrame from the API
        jobs_df = get_jobs_from_api(job_titles=job_title, city=city, country=country, total_jobs=total_jobs)
        res = jobs_df.to_json(orient="records")
        parsed = json.loads(res)
    

        # Return the JSON response
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting jobs: {str(e)}")

@app.post("/upload-and-process-resume/")
async def upload_and_process_resume(
    file: UploadFile = File(...),
    job_description: str = None
):
    try:
        # Verify file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

        # Create a temporary file to store the uploaded PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Write uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # Parse PDF using unstructured
            elements = partition(temp_path)
            resume_text = "\n".join([str(element) for element in elements])

            # Process the resume using existing endpoint logic
            resume_data = ResumeProcessingResponse(
                resume_text=resume_text,
                job_description=job_description
            )
            
            # Call the existing process_resume function
            result = await process_resume(resume_data)
            
            return result

        finally:
            # Clean up: remove temporary file
            os.unlink(temp_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")