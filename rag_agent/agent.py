import os
import streamlit as st
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize LLM + embeddings
embeddings =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

def process_url(url: str):
    """Loads webpage, splits text, stores in vector DB, and returns vectorstore."""

    loader = WebBaseLoader(url)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)

    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(documents=splits)

    return vectorstore


def create_rag_agent(vectorstore):
    """Returns a LangChain agent using the tool + Gemini."""

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information from vectorstore."""
        docs = vectorstore.similarity_search(query, k=3)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        )
        return serialized, docs

    tools = [retrieve_context]

    prompt = (
        "You have a tool that retrieves context from the webpage. "
        "Use it to answer questions."
    )

    agent = create_agent(llm, tools, system_prompt=prompt)
    return agent
