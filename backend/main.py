# backend/main.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
import redis
import uvicorn

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "openai-rag-index"
REDIS_URL = os.getenv("REDIS_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

redis_client = redis.from_url(REDIS_URL)

# Initialize Pinecone and LangChain components
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
)

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index = pc.Index(PINECONE_INDEX_NAME)

text_field = "text"
vectorstore = PineconeVectorStore(index, embeddings, text_field)


# Custom system prompt to make sure model only answers from context
PROMPT = PromptTemplate(
    template="""You are a helpful assistant. Use ONLY the following context to answer questions.
If you do not know the answer, say "I do not have information on that.".
---
{context}
---
Question: {question}
""",
    input_variables=["context", "question"],
)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT},
)


class TextInput(BaseModel):
    text: str


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    ip = request.client.host
    count = redis_client.incr(ip)
    if count == 1:
        redis_client.expire(ip, 60)
    if count > 10:
        raise HTTPException(status_code=429, detail="Too many requests")
    return await call_next(request)


@app.post("/process_text/")
async def process_text(input: TextInput):
    response = qa_chain.invoke({"query": input.text})
    return {
        "input": input.text,
        "reply": response["result"],
    }


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
