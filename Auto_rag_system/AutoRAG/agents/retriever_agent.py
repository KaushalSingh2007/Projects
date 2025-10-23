# ==============================
# FILE: agents/retriever_agent.py
# ==============================
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
class RetrieverAgent:
    """
    Agent responsible for retrieving relevant document chunks based on user queries.
    Uses FAISS for efficient similarity search over document embeddings.
    """
    def __init__(self, db_path: str = "embeddings/faiss_index"):
        """
        Initialize the retriever agent with a FAISS vector store.
        Args:
            db_path (str): Path to the directory containing the FAISS index
        """
        # Resolve index path relative to project root (folder containing this file's parent)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        abs_path = db_path if os.path.isabs(db_path) else os.path.abspath(os.path.join(project_root, db_path))
        self.db_path = abs_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = None
        # Load the vector store if it exists
        if os.path.exists(self.db_path):
            self._load_vector_store()
    def _load_vector_store(self) -> bool:
        """Load the FAISS vector store from disk."""
        try:
            self.vectorstore = FAISS.load_local(
                folder_path=self.db_path,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[Retriever] Loaded vector store from {self.db_path}")
            return True
        except Exception as e:
            print(f"[Retriever] Error loading vector store: {str(e)}")
            self.vectorstore = None
            return False
    def is_ready(self) -> bool:
        """Check if the retriever is ready to use."""
        return self.vectorstore is not None
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        score_threshold: float = 0.5,
        **search_kwargs
    ) -> List[Document]:
        """
        Retrieve relevant document chunks for a given query.
        Args:
            query (str): The user's query
            top_k (int): Maximum number of documents to retrieve
            score_threshold (float): Minimum similarity score for retrieved documents
            **search_kwargs: Additional arguments for the similarity search
        Returns:
            List[Document]: List of relevant document chunks
        """
        if not self.is_ready():
            print("[Retriever] Vector store not loaded. Cannot perform retrieval.")
            return []
        try:
            # Configure search parameters
            search_params = {
                "k": top_k,
                "score_threshold": score_threshold,
                **search_kwargs
            }
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                **search_params
            )
            # Fallback if no results using score API
            if not results:
                try:
                    basic_docs = self.vectorstore.similarity_search(query=query, k=top_k)
                except TypeError:
                    # some versions require positional args
                    basic_docs = self.vectorstore.similarity_search(query, k=top_k)
                results = [(d, 0.0) for d in basic_docs]
            # Process results
            documents = []
            for doc, score in results:
                # Add score to document metadata
                try:
                    doc.metadata["relevance_score"] = float(score)
                except Exception:
                    doc.metadata["relevance_score"] = 0.0
                documents.append(doc)
            print(f"[Retriever] Retrieved {len(documents)} documents for query")
            return documents
        except Exception as e:
            print(f"[Retriever] Error during retrieval: {str(e)}")
            return []
    def get_similar_documents(
        self, 
        doc_id: str, 
        top_k: int = 3
    ) -> List[Document]:
        """
        Find documents similar to a specific document.
        Args:
            doc_id (str): ID of the reference document
            top_k (int): Number of similar documents to retrieve
        Returns:
            List[Document]: List of similar documents
        """
        if not self.is_ready():
            return []
        try:
            # Get the document's embedding
            doc_index = self.vectorstore.index_to_docstore_id.get(doc_id)
            if not doc_index:
                print(f"[Retriever] Document with ID {doc_id} not found")
                return []
            # Get similar documents
            similar_docs = self.vectorstore.similarity_search(
                query=None,
                k=top_k,
                filter={"doc_id": doc_id}
            )
            return similar_docs
        except Exception as e:
            print(f"[Retriever] Error finding similar documents: {str(e)}")
            return []
# Example usage
if __name__ == "__main__":
    # Initialize the retriever
    retriever = RetrieverAgent()
    if retriever.is_ready():
        # Example query
        query = "What is the main topic of this document?"
        results = retriever.retrieve(query, top_k=3)
        print(f"\nResults for query: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\nDocument {i} (Score: {doc.metadata.get('relevance_score', 'N/A'):.3f}):")
            print("-" * 50)
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
    else:
        print("Retriever is not ready. Please ensure the FAISS index exists at the specified path.")
