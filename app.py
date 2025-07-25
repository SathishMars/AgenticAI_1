import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# === Streamlit App Config ===
st.set_page_config(page_title="Agentic PDF Assistant", page_icon="🤖")
st.title("🤖 Agentic PDF Assistant")
st.write("Upload a PDF and ask your AI agent anything about its content.")

# === Validate API Key ===
if not groq_api_key:
    st.error("🚫 GROQ_API_KEY is missing in the .env file. Please add it before proceeding.")
    st.stop()

# === Initialize LLM ===
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.7,
    max_tokens=1024
)

# === PDF Upload UI ===
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

# === Session state ===
if "agent" not in st.session_state:
    st.session_state.agent = None

# === PDF Processing and Agent Setup ===
def setup_agent(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = vectorstore.as_retriever()

        tool = Tool(
            name="PDF Retriever",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
            description="Useful for answering questions from the uploaded PDF"
        )

        agent = initialize_agent(
            tools=[tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False
        )

        return agent
    except Exception as e:
        st.error(f"❌ Failed to set up agent: {e}")
        return None

# === Create agent after upload ===
if uploaded_file and st.session_state.agent is None:
    with st.spinner("⚙️ Processing PDF and setting up the agent..."):
        agent = setup_agent(uploaded_file)
        if agent:
            st.session_state.agent = agent
            st.success("🎯 Agent is ready! Ask your first question below.")

# === Ask a Question ===
if st.session_state.agent:
    st.markdown("### 💬 Ask something about your PDF")
    user_input = st.text_input("Type your question and press Enter:")

    if user_input:
        with st.spinner("🤔 Thinking..."):
            try:
                answer = st.session_state.agent.run(user_input)
                st.markdown(f"**🧠 Answer:** {answer}")
            except Exception as e:
                st.error(f"❌ Error generating answer: {e}")
