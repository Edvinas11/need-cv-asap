from typing import Dict
import os
from openai import AsyncOpenAI

class OpenAIService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def analyze_cv(self, cv_text: str) -> Dict:
        try:
            system_prompt = """You are an expert CV/Resume analyzer. Analyze the provided CV and return:
            1. Key skills
            2. Years of experience
            3. Education summary
            4. Career highlights
            5. Areas of improvement
            6. Overall assessment
            Provide the response in a structured JSON format."""

            response = await self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for a more cost-effective option
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": cv_text}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error analyzing CV: {str(e)}") 