import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "./chroma_db"

# Load documents from your CRM export (assume plain text for now)
loader = DirectoryLoader("./documents", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# Split documents into chunks for embeddings
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# Generate embeddings and persist to chroma db
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
)
vectorstore = Chroma.from_documents(
    documents=splits, embedding=embeddings, persist_directory=CHROMA_PATH
)

print("Ingestion complete.")
