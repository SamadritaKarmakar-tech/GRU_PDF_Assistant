import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found.")
    st.stop()


# Load embeddings + FAISS
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

vectorstore = FAISS.load_local(
    "gru_index",
    embeddings,
    allow_dangerous_deserialization=True
)



# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
    temperature=0
)


# Streamlit UI
st.title("GRU PDF Assistant")

query = st.text_input("Ask a question about GRU:")


if query:

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        query,
        k=3
    )

    # Create context
    context = "\n\n".join(
        r.page_content for r in results
    )

    # Prompt
    prompt = f"""
Answer the question using ONLY the information
provided in the context.

Context:
{context}

Question:
{query}

Give a clear and concise answer within 5-10 lines.
"""

    try:

        # Generate answer
        answer = llm.invoke(prompt)

        # Extract only text
        if isinstance(answer.content, str):
            final_answer = answer.content

        elif isinstance(answer.content, list):
            final_answer = "\n".join(
                item.get("text", "")
                for item in answer.content
                if isinstance(item, dict)
                and item.get("type") == "text"
            )

        else:
            final_answer = str(answer.content)

        # ONLY display answer
        st.subheader("AI Assistant Answer")
        st.markdown(final_answer)

    except Exception as e:

        st.error(f"Error while generating answer: {e}")