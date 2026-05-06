# ==============================
# FILE: agents/summarizer_agent.py
# ==============================
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline
class SummarizerAgent:
    """
    Agent responsible for generating summaries of documents or text.
    Can operate in different modes: 'stuff', 'map_reduce', 'refine', 'map_rerank'.
    """
    def __init__(
        self, 
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        chain_type: str = "map_reduce"
    ):
        """
        Initialize the summarizer agent.
        Args:
            model_name (str): Name of the language model to use
            temperature (float): Controls randomness in generation (0-1)
            chain_type (str): Type of summarization chain to use
        """
        # Model selection: prefer explicit arg, else env, else safe default
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")
        self.temperature = temperature
        self.chain_type = chain_type.lower()
        # Supported chain types
        self.supported_chains = ["stuff", "map_reduce", "refine", "map_rerank"]
        if self.chain_type not in self.supported_chains:
            print(f"[Summarizer] Warning: Unsupported chain type '{chain_type}'. Defaulting to 'map_reduce'.")
            self.chain_type = "map_reduce"
        # Load environment variables
        load_dotenv()
        # Initialize the language model (Gemini)
        self.llm = self._initialize_llm()
        # Local fallback summarizer config
        self.local_summarizer = None
        self.local_model_name = os.getenv("LOCAL_SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
        self.enable_local_fallback = os.getenv("ENABLE_LOCAL_SUMMARIZER", "true").lower() in ("1", "true", "yes")
    def _initialize_llm(self):
        """Initialize the language model for summarization."""
        try:
            # Uses GOOGLE_API_KEY from environment (loaded via load_dotenv())
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                convert_system_message_to_human=True,
            )
        except Exception as e:
            msg = str(e)
            print(f"[Summarizer] Error initializing Gemini model '{self.model_name}': {msg}")
            # Try fallback model variants on 404/not found
            if "404" in msg or "not found" in msg.lower():
                fallbacks = [m for m in [
                    "gemini-1.5-flash-002",
                    "gemini-1.5-pro-002",
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-pro-latest",
                ] if m != self.model_name]
                for alt in fallbacks:
                    try:
                        print(f"[Summarizer] Retrying with fallback model: {alt}")
                        self.model_name = alt
                        return ChatGoogleGenerativeAI(
                            model=self.model_name,
                            temperature=self.temperature,
                            convert_system_message_to_human=True,
                        )
                    except Exception as _:
                        continue
            raise
    def summarize_text(
        self, 
        text: str,
        summary_length: str = "brief",
        **kwargs
    ) -> str:
        """
        Generate a summary of the input text.
        Args:
            text (str): The text to summarize
            summary_length (str): Desired length of the summary ('brief', 'medium', 'detailed')
            **kwargs: Additional arguments for the summarization chain
        Returns:
            str: The generated summary
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        try:
            # Topic-like detection to bias towards theorems/definitions/examples
            topic_like = (len(text) < 120 and len(text.split()) <= 12)
            if topic_like:
                templates = {
                    "brief": (
                        "Summarize the topic focusing on 3-6 bullet points: key definitions, core theorems/laws (with names),"
                        " typical examples/applications, and common pitfalls. Keep it crisp.\n\nTOPIC:\n{text}\n\nSUMMARY:"
                    ),
                    "medium": (
                        "Write a focused paragraph explaining the topic with definitions and 2-3 named theorems/laws,"
                        " plus typical use cases and a short example. Under 180 words.\n\nTOPIC:\n{text}\n\nSUMMARY:"
                    ),
                    "detailed": (
                        "Explain the topic with definitions, major theorems/laws (name + 1-line statement), examples,"
                        " and practical considerations. 200-300 words.\n\nTOPIC:\n{text}\n\nSUMMARY:"
                    ),
                }
            else:
                templates = {
                    "brief": (
                        "Write a concise summary of the text in 3-5 bullet points."
                        " Be specific and avoid filler.\n\nTEXT:\n{text}\n\nSUMMARY:"
                    ),
                    "medium": (
                        "Write a detailed single-paragraph summary capturing key points, context, and conclusions."
                        " Keep it under 180 words.\n\nTEXT:\n{text}\n\nSUMMARY:"
                    ),
                    "detailed": (
                        "Write a comprehensive multi-paragraph summary including context, examples, and key takeaways."
                        " Aim for 200-300 words.\n\nTEXT:\n{text}\n\nSUMMARY:"
                    ),
                }
            prompt = templates.get(summary_length.lower(), templates["medium"]).replace("{text}", text)
            response = self.llm.invoke(prompt)
            result_text = getattr(response, "content", None) or str(response)
            return result_text.strip()
        except Exception as e:
            msg = str(e)
            print(f"[Summarizer] Error during summarization: {msg}")
            if self.enable_local_fallback:
                try:
                    print("[Summarizer] Falling back to local Transformers summarizer...")
                    return self._local_summarize(text, summary_length)
                except Exception as le:
                    print(f"[Summarizer] Local fallback failed: {le}")
            return f"Error during summarization: {msg}"

    def summarize_documents(
        self, 
        documents: List[Document],
        summary_length: str = "brief",
        **kwargs
    ) -> str:
        """Summarize a list of documents using the direct prompt path (no LC summarize chain)."""
        if not documents:
            return "No documents provided for summarization."
        merged_text = "\n\n".join([getattr(d, "page_content", "") for d in documents if getattr(d, "page_content", "")])
        if not merged_text.strip():
            return "No document content to summarize."
        return self.summarize_text(merged_text, summary_length=summary_length, **kwargs)

    def _init_local(self):
        if self.local_summarizer is None:
            self.local_summarizer = hf_pipeline("summarization", model=self.local_model_name)

    def _local_summarize(self, text: str, summary_length: str = "medium") -> str:
        self._init_local()
        # Heuristic chunking to respect model max length
        # Most BART-like models accept ~1024 tokens; we chunk by characters conservatively.
        max_chunk_chars = 3000
        chunks = []
        t = text.strip()
        while t:
            chunks.append(t[:max_chunk_chars])
            t = t[max_chunk_chars:]
        # Length settings
        if summary_length == "brief":
            max_len, min_len = 100, 30
        elif summary_length == "detailed":
            max_len, min_len = 220, 80
        else:
            max_len, min_len = 150, 60
        partial_summaries = []
        for ch in chunks:
            out = self.local_summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)
            partial_summaries.append(out[0]["summary_text"].strip())
        # If multiple chunks, summarize the summaries for coherence
        merged = "\n\n".join(partial_summaries)
        if len(partial_summaries) > 1:
            out2 = self.local_summarizer(merged, max_length=max_len, min_length=min_len, do_sample=False)
            return out2[0]["summary_text"].strip()
        return merged
    def summarize_with_chain_of_thought(
        self, 
        text: str,
        question: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a summary with chain-of-thought reasoning.
        Args:
            text (str): The text to summarize
            question (str, optional): A specific question the summary should answer
            **kwargs: Additional arguments for the summarization
        Returns:
            Dict[str, Any]: Dictionary containing the summary and reasoning steps
        """
        try:
            # Create a more sophisticated prompt for chain-of-thought summarization
            prompt = """
            You are an expert summarizer with strong reasoning abilities. 
            Follow these steps to create a high-quality summary:
            1. Analyze the text to understand the main topics and key points.
            2. Identify the most important information and supporting details.
            3. Consider the context and any implicit information.
            4. Synthesize the information into a coherent summary.
            {question}
            Text to summarize:
            {text}
            Please provide:
            1. A concise summary of the key points.
            2. The reasoning behind why these points are important.
            3. Any additional context or implications.
            """
            # Format the prompt with the input text and question
            formatted_prompt = prompt.format(
                text=text,
                question=f"Question to answer: {question}" if question else ""
            )
            # Generate the summary with reasoning
            response = self.llm.generate([formatted_prompt])
            summary = response.generations[0][0].text.strip()
            return {
                "summary": summary,
                "reasoning": "Chain-of-thought reasoning was used to generate this summary.",
                "model": self.model_name
            }
        except Exception as e:
            error_msg = f"Error in chain-of-thought summarization: {str(e)}"
            print(f"[Summarizer] {error_msg}")
            return {"error": error_msg}

    def simplify(self, text: str, audience: str = "beginner", max_words: int = 160) -> str:
        """Rewrite text in simple terms for a target audience using the LLM; fallback to local on error."""
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
        try:
            prompt = (
                f"Rewrite the following content in simple, clear language for a {audience}. "
                f"Avoid jargon, explain key terms briefly, and use short sentences. Limit to about {max_words} words.\n\n"
                f"CONTENT:\n{text}\n\n"
                "SIMPLE VERSION:"
            )
            resp = self.llm.invoke(prompt)
            out = getattr(resp, "content", None) or str(resp)
            return out.strip()
        except Exception as e:
            msg = str(e)
            print(f"[Summarizer] Error simplifying text: {msg}")
            # Fallback: crude shortening via local summarizer if available
            if self.enable_local_fallback:
                try:
                    base = self._local_summarize(text, summary_length="brief")
                    return base
                except Exception:
                    pass
            return f"Error simplifying text: {msg}"
# Example usage
if __name__ == "__main__":
    # Initialize the summarizer
    summarizer = SummarizerAgent(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
        chain_type="map_reduce"
    )
    # Example text to summarize
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence 
    displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions that maximize its chance of achieving 
    its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and display 
    "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving".
    """
    # Generate a summary
    print("Generating summary...")
    summary = summarizer.summarize_text(
        sample_text,
        summary_length="medium"
    )
    print("\nOriginal Text:")
    print("-" * 50)
    print(sample_text)
    print("\nSummary:")
    print("-" * 50)
    print(summary)
