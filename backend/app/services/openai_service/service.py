import openai
import os
from typing import Optional
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

TEMP_DIR = os.path.join(os.path.dirname(__file__), "tmp/")
os.makedirs(TEMP_DIR, exist_ok=True)

class OpenAIService:
    """
    A service to interact with OpenAI's API, including Assistant and File Processing.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in OPENAI_API_KEY environment variable"
            )

        self.client = openai.OpenAI(api_key=self.api_key)

    def create_pdf_assistant(self):
        """
        Creates an OpenAI Assistant for PDF content extraction.
        """
        return self.client.beta.assistants.create(
            model="gpt-4o",
            description="An assistant to extract the contents of PDF files.",
            tools=[{"type": "file_search"}],
            name="PDF assistant",
        )

    def upload_file(self, file_path: str):
        """
        Uploads a PDF file to OpenAI.
        """
        return self.client.files.create(
            file=open(file_path, "rb"), purpose="assistants"
        )

    async def process_cv(self, file):
        """
        Handles CV processing by extracting text using OpenAI's Assistant.
        """

        temp_file_path = os.path.join(TEMP_DIR, file.filename)

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())
        print(f"File temporary saved in {TEMP_DIR}")
        
        # Create PDF Assistant
        print("Creating PDF assistant...")
        pdf_assistant = self.create_pdf_assistant()
        print(f"Created. Assistant ID: {pdf_assistant.id}")

        # Create OpenAI Thread
        thread = self.client.beta.threads.create()

        # Upload file to OpenAI
        uploaded_file = self.upload_file(temp_file_path)

        # Define prompt
        prompt = """You are an expert in resume analysis, resume optimization, and recruitment technology. Your task is to analyze the provided CV file, ensuring it meets industry best practices for automated screening and ranking.

            <Instructions>
            OUTPUT FORMAT:
            Return ONLY a valid JSON object in this format:
            {
                "features" : [
                    {
                        "issue": "actual found issue here",
                        "description": "issue explanation here",
                        "action": "specific recommendation for improvement",
                        "result": "fix the issue"
                    }
                ]
            }

            IMPORTANT:
            Do NOT return any additional text or explanations outside the JSON structure.

            </Instructions>
            """

        # Send message to OpenAI Assistant
        self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            attachments=[
                Attachment(
                    file_id=uploaded_file.id,
                    tools=[AttachmentToolFileSearch(type="file_search")],
                )
            ],
            content=prompt,
        )

        # Run assistant thread
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=pdf_assistant.id, timeout=1000
        )

        if run.status != "completed":
            raise Exception(f"Run failed: {run.status}")

        # Fetch response messages
        messages_cursor = self.client.beta.threads.messages.list(thread_id=thread.id)
        messages = [message for message in messages_cursor]

        # Extract the text response
        extracted_text = messages[0].content[0].text.value

        # Cleanup temp file
        # os.remove(temp_file_path)

        print("OUTPUT:" + extracted_text)

        return extracted_text
