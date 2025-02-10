from app.services.openai_service.service import OpenAIService
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="CV Analysis API")

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
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        extracted_text = await openai_service.process_cv(file)
        return {"extracted_text": extracted_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
