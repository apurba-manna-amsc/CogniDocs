import os
import streamlit as st
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader, Docx2txtLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
FAISS_INDEX_PATH = "faiss.index"
TEXT_CHUNKS_PATH = "text_chunks.npy"

#st.session_state.chat_history = []

# Streamlit UI
st.set_page_config(page_title="CogniDocs", layout="wide")
st.title("📚✨ CogniDocs: Your Smart Document Assistant")

# Sidebar for document upload
st.sidebar.header("📂 Upload & Index Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF, TXT, DOCX files", accept_multiple_files=True)


class DocumentProcessor:
    """Handles loading and splitting documents."""
    
    @staticmethod
    def load_document(file_path):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        return loader.load()

    @staticmethod
    def split_documents(docs, chunk_size=512, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(docs)


class VectorStoreManager:
    """Manages FAISS vector store."""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.text_chunks = []
        self.load_faiss_index()

    def generate_embeddings(self, texts):
        return np.array([self.embedding_model.encode(text, convert_to_numpy=True) for text in texts]).astype("float32")

    def build_faiss_index(self, texts):
        if not texts:
            raise ValueError("No text chunks to index.")
        embeddings = self.generate_embeddings(texts)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.text_chunks = texts
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        np.save(TEXT_CHUNKS_PATH, self.text_chunks)
        print("✅ FAISS index built and saved.")

    def load_faiss_index(self):
        """Loads FAISS index if available."""
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_CHUNKS_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            self.text_chunks = np.load(TEXT_CHUNKS_PATH, allow_pickle=True).tolist()
            print("✅ FAISS index and text chunks loaded successfully.")
        else:
            print("⚠ FAISS index not found. Please index documents first.")

    def retrieve_most_relevant(self, query, top_k=1):
        """Retrieve the most relevant chunk from FAISS index."""
        if self.index is None:
            return "General knowledge"
        
        query_embedding = self.generate_embeddings([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return " ".join([self.text_chunks[i] for i in indices[0] if i < len(self.text_chunks)]) or "No relevant information found."


class GroqAPIHandler:
    """Handles Groq API requests for AI responses with chat memory."""
    
    @staticmethod
    def generate_response(query, context, chat_history):
        """Sends query along with chat history for better contextual answers."""
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

        # Build full conversation context
        conversation_history = "\n".join([f"{role}: {msg}" for role, msg in chat_history[-5:]])  # Keep last 5 exchanges
        
        prompt = f"""You are an AI assistant named CogniDocs that can answer general questions and retrieve document-based knowledge. If the user asks a common question (like greetings or general queries), provide a friendly and helpful response.

        Chat History:
        {conversation_history}

        Context:
        {context}

        Query: {query}

        If the context is empty or unrelated, generate a response based on general knowledge.
        """

        
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "system", "content": "You are a helpful AI assistant."}, {"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        try:
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"


# Handle document indexing
if st.sidebar.button("Index Documents"):
    if uploaded_files:
        os.makedirs("uploaded_docs", exist_ok=True)
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join("uploaded_docs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        documents = []
        for file_path in file_paths:
            docs = DocumentProcessor.load_document(file_path)
            documents.extend(DocumentProcessor.split_documents(docs))
        
        text_chunks = [doc.page_content for doc in documents]
        vector_store = VectorStoreManager()
        vector_store.build_faiss_index(text_chunks)
        st.sidebar.success("✅ Indexing completed successfully!")
    else:
        st.sidebar.warning("⚠ Please upload documents before indexing.")

# Initialize vector store
vector_store = VectorStoreManager()

# Chat UI
st.subheader("🤖💬 Welcome to CogniDocs Chat Assistant!")

# Initialize chat memory in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # Store (role, message) tuples

query = st.chat_input("Ask a question...")

if query:
    # Store user query
    st.session_state.chat_history.append(("You", query))

    # Retrieve relevant document context
    retrieved_context = vector_store.retrieve_most_relevant(query)
    # print(retrieved_context)

    if retrieved_context.startswith("⚠"):
        response = retrieved_context
    else:
        # Pass chat history for contextual responses
        response = GroqAPIHandler.generate_response(query, retrieved_context, st.session_state.chat_history)
    
    # Store AI response
    st.session_state.chat_history.append(("🤖 AI", response))

    # Display chat history
    for sender, message in st.session_state.chat_history:
        with st.chat_message("user" if sender == "You" else "assistant"):
            st.write(f"{message}")