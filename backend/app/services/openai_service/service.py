import openai
import os
from typing import Optional
from openai.types.beta.threads.message_create_params import (
    Attachment,
    AttachmentToolFileSearch,
)
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()

base_dir = Path(__file__).resolve().parent.parent.parent # backend/app
temporary_dir_path = base_dir / "cv-processing"
temporary_dir_path.mkdir(parents=True, exist_ok=True)

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
            instructions="You are a helpful PDF assistant and you scan the file provided to you.",
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
    
    def load_prompt(self, prompt_file: str) -> str:
        """
        Loads the prompt text from a given .txt file.
        """
        prompt_file_path = base_dir / "prompts" / prompt_file

        if not prompt_file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

        with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read()

    async def process_cv(self, file):
        """
        Handles CV processing by extracting text using OpenAI's Assistant.
        """

        temp_file_path = temporary_dir_path / file.filename

        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())
        print(f"File temporary saved in {temp_file_path}")
        
        # Create PDF Assistant
        print("Creating PDF assistant...")
        pdf_assistant = self.create_pdf_assistant()
        print(f"Created. Assistant ID: {pdf_assistant.id}")

        # Create OpenAI Thread
        thread = self.client.beta.threads.create()

        # Upload file to OpenAI
        uploaded_file = self.upload_file(temp_file_path)

        # Define prompt
        prompt = self.load_prompt("cv-scan.txt")

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
        json_output_str = messages[0].content[0].text.value

        cleaned_json = json_output_str.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json.split("\n", 1)[1]  # Remove first line
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json.rsplit("\n", 1)[0]  # Remove last line

        output = json.loads(cleaned_json)

        # Cleanup temp file
        # os.remove(temp_file_path)

        return output
