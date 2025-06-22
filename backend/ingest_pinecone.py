import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = "openai-rag-index"
MODEL = "text-embedding-3-small"


def process_txt_file(file_path):
    # create a loader
    loader = DirectoryLoader(file_path, glob="**/*.txt", loader_cls=TextLoader)
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0)
    documents = text_splitter.split_documents(data)

    return documents


def main():

    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        # You can remove this parameterfor your own projects
    )

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            # dimension of the vector embeddings produced by
            # OpenAI's text-embedding-3-small
            dimension=1536,
            metric="euclidean",
            # parameters for the free tier index
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    file_path = "../documents"
    # Replace with your actual file path,
    # there should be .txt files in that folder for ingestion
    text_chunks = process_txt_file(file_path)
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
    )

    PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    PineconeVectorStore.from_documents(
        text_chunks, index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    index = pc.Index(name=PINECONE_INDEX_NAME)
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
