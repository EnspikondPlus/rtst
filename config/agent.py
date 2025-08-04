from google import genai
from google.genai import types

class Agent:
    def __init__(self, system_instruction: str, api_key: str, temperature: float, response_schema: str, model: str):
        self.system_instruction = system_instruction
        self.api_key = api_key
        self.temperature = temperature
        self.response_schema = response_schema
        self.model = model
    
    def __call__(self, query):
        client = genai.Client(api_key=self.api_key)

        response = client.models.generate_content(
            model=self.model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=self.temperature,
                response_mime_type="application/json",
                response_schema=self.response_schema),
            contents=query
        )

        return response