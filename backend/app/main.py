from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.openai_service import OpenAIService
import PyPDF2
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="CV Analysis API")

# Initialize OpenAI service
openai_service = OpenAIService()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to CV Analysis API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze-cv")
async def analyze_cv(file: UploadFile = File(...)):
    # Check if file is PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read the PDF file
        pdf_content = await file.read()
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        cv_text = ""
        for page in pdf_reader.pages:
            cv_text += page.extract_text()

        print(cv_text)
        
        # Analyze CV using OpenAI
        analysis_result = await openai_service.analyze_cv(cv_text)
        
        return {"analysis": analysis_result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
