# =======================
# FILE: rag_chain.py
# =======================
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
def build_rag_chain(embedding_path="embeddings/faiss_index", model_name: str | None = None):
    """
    Build a RAG chain with the specified embedding path and model.
    Args:
        embedding_path (str): Path to the FAISS index
        model_name (str): Name of the OpenAI model to use
    Returns:
        ConversationalRetrievalChain: Configured RAG chain
    """
    try:
        # Check if the embedding path exists
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding path {embedding_path} does not exist. Please run data_loader.py first.")
        print(f"[INFO] Loading embeddings from {embedding_path}")
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        # Load vector store
        vectorstore = FAISS.load_local(
            folder_path=embedding_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        # Configure retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        # Initialize LLM (using Google Gemini)
        # Prefer explicit arg, otherwise env, otherwise safe default
        desired_model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")
        print(f"[INFO] Initializing Gemini LLM: {desired_model}")
        try:
            llm = ChatGoogleGenerativeAI(
                model=desired_model,
                temperature=0.7,
                convert_system_message_to_human=True,
            )
        except Exception as e:
            msg = str(e)
            print(f"[RAG] Error initializing Gemini model '{desired_model}': {msg}")
            # Try a couple of known-good fallbacks
            fallbacks = [m for m in [
                "gemini-1.5-flash-002",
                "gemini-1.5-pro-002",
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro-latest",
            ] if m != desired_model]
            llm = None
            for alt in fallbacks:
                try:
                    print(f"[RAG] Retrying with fallback model: {alt}")
                    llm = ChatGoogleGenerativeAI(
                        model=alt,
                        temperature=0.7,
                        convert_system_message_to_human=True,
                    )
                    desired_model = alt
                    break
                except Exception:
                    continue
            if llm is None:
                raise
        # Build chain with memory and source documents
        print("[INFO] Building RAG chain...")
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        print("[SUCCESS] RAG chain built successfully")
        return chain
    except Exception as e:
        print(f"[ERROR] Failed to build RAG chain: {str(e)}")
        raise
def get_answer(query, chat_history=None, embedding_path="embeddings/faiss_index"):
    """
    Get an answer for a query using the RAG chain.
    Args:
        query (str): User's question
        chat_history (list): List of previous conversation turns
        embedding_path (str): Path to the FAISS index
    Returns:
        tuple: (answer, source_documents)
    """
    if chat_history is None:
        chat_history = []
    try:
        chain = build_rag_chain(embedding_path)
        # Format chat history for the chain
        formatted_history = [(h[0], h[1]) for h in chat_history]
        # Get response from the chain
        result = chain({
            "question": query,
            "chat_history": formatted_history
        })
        return result["answer"], result["source_documents"]
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg, []
# Example usage
if __name__ == "__main__":
    question = "What is this document about?"
    answer, sources = get_answer(question)
    print("\nQuestion:", question)
    print("\nAnswer:", answer)
    print("\nSources:")
    for i, doc in enumerate(sources, 1):
        print(f"\nSource {i} (Page {doc.metadata.get('page', 'N/A')}):")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
