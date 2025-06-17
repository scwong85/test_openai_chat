from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import redis
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
CHROMA_PATH = "./chroma_db"

# Initialize services
app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

redis_client = redis.from_url(REDIS_URL)

# Setup vector DB
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
)
vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Setup LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_API_KEY)

# Custom RAG chain
prompt_template = """
You are a helpful assistant. Use ONLY the following context to answer.
If the answer cannot be found, say "I don't know."

Context:
{context}

Question:
{question}
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
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

    if not os.path.exists(CHROMA_PATH):
        print(f"ERROR: Chroma path {CHROMA_PATH} does not exist.")
    else:
        print(f"Chroma path {CHROMA_PATH} exists.")
        print(os.listdir(CHROMA_PATH))
        for files in os.listdir(CHROMA_PATH):
            print(files)

    collection = vectorstore.get()
    print("Number of documents loaded from vector store:", len(collection["documents"]))

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(input.text)
    print(f"Retrieved {len(docs)} documents")
    for doc in docs:
        print(doc.page_content[:200])

    if not docs:
        return {"reply": "Sorry, I don't have any relevant information to answer that."}

    if not os.path.exists(CHROMA_PATH):
        print(f"ERROR: Chroma path {CHROMA_PATH} does not exist.")
    else:
        print(f"Chroma path {CHROMA_PATH} exists.")

    response = qa_chain.invoke({"query": input.text})
    return {"input": input.text, "reply": response["result"]}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
