import os
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


load_dotenv()

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_INDEX_NAME = "study-buddhism-index"
MODEL = "text-embedding-3-small"


def process_txt_file(file_path, source_url):
    # create a loader
    loader = TextLoader(file_path)
    data = loader.load()
    for doc in data:
        doc.metadata["source"] = source_url  # this is usually filename
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0)
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

    file_path_1 = "../documents/8-buddhist-tips-for-dealing-with-anger.txt"
    file_path_2 = "../documents/compassion-as-a-source-of-happiness.txt"
    # Replace with your actual file path,
    # there should be .txt files in that folder for ingestion
    text_chunks_1 = process_txt_file(
        file_path_1,
        "https://studybuddhism.com/en/essentials/how-to/8-buddhist-tips-for-dealing-with-anger",
    )

    text_chunks_2 = process_txt_file(
        file_path_2,
        "https://studybuddhism.com/en/essentials/universal-values/compassion-as-a-source-of-happiness",
    )
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small"
    )

    PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=embeddings)

    PineconeVectorStore.from_documents(
        text_chunks_1, index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    PineconeVectorStore.from_documents(
        text_chunks_2, index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )
    index = pc.Index(name=PINECONE_INDEX_NAME)
    print(index.describe_index_stats())


if __name__ == "__main__":
    main()
