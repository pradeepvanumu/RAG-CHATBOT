import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()

# Cache document loading and embedding
@st.cache_resource
def load_data_and_create_vectorstore():
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = load_data_and_create_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Sidebar model info
st.sidebar.title("Model Settings")
model_name = "llama-3.3-70b-versatile"
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
st.sidebar.write(f"Model: {model_name}")

# Main interface
st.title("RAG Sales Assistant")
user_question = st.text_input("Ask a sales-related question:")

# System prompt
SYSTEM_PROMPT = """
You are NovaEdge AI Sales Assistant, an experienced Senior Sales Consultant with 3 years of experience.

Your role is to mentor junior sales executives and assist them in their day-to-day sales activities.

Your responsibilities include:
- Explaining CRM concepts in simple language.
- Helping users understand company products and services.
- Guiding sales executives through the sales process.
- Helping prepare for customer meetings.
- Assisting with proposal preparation.
- Helping draft professional customer emails.
- Helping handle customer objections.
- Recommending best practices from the company documents.
- Teaching instead of simply repeating document content.

Rules:
- Always use the retrieved context as your primary source.
- Never invent company policies, pricing, discounts, product features, or commercial terms.
- If the answer is not available in the retrieved context, reply:
"I couldn't find that information in the company documents."

Keep your responses professional, practical, concise, and easy to understand.
"""

if user_question:
    retrieved_docs = retriever.invoke(user_question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""{SYSTEM_PROMPT}

Retrieved Context:
{context}

User Question:
{user_question}

Respond according to the user's request.

Instructions:

1. If the user asks for an explanation, explain the concept clearly and provide practical sales advice.

2. If the user asks to write an email, draft a professional business email using the retrieved company information.

3. If the user asks how to respond to a customer, provide a professional customer response.

4. If the user asks about proposals, pricing, products, sales processes, CRM features, customer objections, meetings, or company policies, answer only using the retrieved context.

5. If the user asks for steps or guidance, provide clear step-by-step instructions.

6. If the information is unavailable in the retrieved context, respond exactly:
"I couldn't find that information in the company documents."

Always respond as an experienced Senior Sales Consultant mentoring junior sales executives.
"""

    # Initialize LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=model_name,
        temperature=temperature
    )

    # Generate Response
    with st.spinner("🤖 Thinking..."):
        response = llm.invoke(prompt)

    # Display Answer
    st.markdown("## 💬 AI Response")
    st.success(response.content)

    # Divider
    st.divider()

    # Retrieved Source Documents
    st.subheader("📄 Source Documents")

    sources = list(
        set(
            doc.metadata.get("source", "Unknown")
            for doc in retrieved_docs
        )
    )

    for source in sources:
        st.write(f"✅ {os.path.basename(source)}")

    # Expand Retrieved Chunks
    with st.expander("📚 View Retrieved Context"):

        for i, doc in enumerate(retrieved_docs, start=1):

            st.markdown(f"### Chunk {i}")

            st.write(doc.page_content)

            st.markdown("---")
