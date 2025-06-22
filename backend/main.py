# backend/main.py

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from pinecone import Pinecone
import redis
import uvicorn

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "openai-rag-index"
PINECONE_INDEX_NAME = "study-buddhism-index"
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
vectorstore = PineconeVectorStore(index, embeddings, text_key=text_field)


# Custom system prompt to make sure model only answers from context

prompt_template = """Use the following pieces of context to answer the question at the end.  Try to answer in a structured way. Write your answer in HTML format but do not include ```html ```. Put words in bold that directly answer your question.
If you don't know the answer, just say 'I am sorry I dont know the answer to this question or you dont have access to the files needed to answer the question.' Don't try to make up an answer.

{summaries}


Question: {question}.
"""

PROMPT_WITH_SOURCE = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)


llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0
)


qa_chain_source = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": PROMPT_WITH_SOURCE},
    return_source_documents=True,
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

    response = qa_chain_source.invoke({"question": input.text})
    # Extract sources as list of strings
    sources = []
    if "I dont know the answer" not in response["answer"]:
        source_docs = response["source_documents"]
        sources = [
            doc.metadata.get("source", "")
            for doc in source_docs
            if doc.metadata.get("source")
        ]
    return {
        "input": input.text,
        "reply": response["answer"],
        "sources": list(set(sources)),  # List of URLs
    }

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
