import streamlit as st
from rag_agent.agent import process_url, create_rag_agent
from langchain_google_genai import ChatGoogleGenerativeAI

import os
   # load GOOGLE_API_KEY + LANGSMITH_API_KEY

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Web RAG Agent", layout="wide")
st.title("🌐 Chat With Any Webpage (LangChain + Gemini)")


# UI Input
url = st.text_input("Enter a webpage URL")
query = st.text_area("Ask a question about this webpage")


if st.button("Ask"):
    if not url or not query:
        st.warning("Enter a URL and a question.")
    else:
        with st.spinner("Processing webpage..."):
            vectorstore = process_url(url)
            agent = create_rag_agent(vectorstore)

            response = agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )

        st.subheader("Answer:")
        st.write(response["messages"][-1].content)
