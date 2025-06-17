# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import tempfile
import uvicorn
import base64
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class TextInput(BaseModel):
    text: str

@app.post("/process_text/")
async def process_text(input: TextInput):
    return await get_response(input.text)

async def get_response(prompt: str):
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "input": prompt,
        "reply": reply,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
