# ==============================
# FILE: agent_pipeline.py
# ==============================
import os
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Import agents
from agents.retriever_agent import RetrieverAgent
from agents.summarizer_agent import SummarizerAgent
from agents.generator_agent import GeneratorAgent, GenerationConfig
from agents.evaluator_agent import EvaluatorAgent, BiasScore
class PipelineMode(str, Enum):
    """Available pipeline modes."""
    ASK = "ask"
    SUMMARIZE = "summarize"
    GENERATE = "generate"
    EVALUATE = "evaluate"
    FULL = "full"
@dataclass
class PipelineConfig:
    """Configuration for the agent pipeline."""
    mode: PipelineMode = PipelineMode.FULL
    max_retrieved_docs: int = 5
    summary_length: str = "medium"
    generation_config: Optional[Dict[str, Any]] = None
    enable_evaluation: bool = True
    save_outputs: bool = True
    output_dir: str = "../outputs"
class AgentPipeline:
    """
    Main pipeline that orchestrates all agents to process user queries.
    Handles the entire workflow from retrieval to generation and evaluation.
    """
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the agent pipeline with configuration.
        Args:
            config (PipelineConfig, optional): Configuration for the pipeline
        """
        self.config = config or PipelineConfig()
        self.retriever = None
        self.summarizer = None
        self.generator = None
        self.evaluator = None
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Initialize agents
        self._initialize_agents()
    def _initialize_agents(self):
        """Initialize all agents with appropriate configuration."""
        try:
            logger.info("Initializing agents...")
            # Initialize Retriever Agent
            self.retriever = RetrieverAgent()
            # Initialize Summarizer Agent
            self.summarizer = SummarizerAgent()
            # Initialize Generator Agent (lazy load when needed)
            self.generator = None
            # Initialize Evaluator Agent
            self.evaluator = EvaluatorAgent()
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {str(e)}")
            raise
    def _initialize_generator(self):
        """Lazy initialization of the generator agent."""
        if self.generator is None:
            try:
                from agents.generator_agent import GeneratorAgent
                self.generator = GeneratorAgent()
                logger.info("Generator agent initialized")
            except ImportError:
                logger.warning("Failed to initialize generator agent. Image generation will not be available.")
                self.generator = None
    def process_query(
        self, 
        query: str,
        mode: Optional[Union[str, PipelineMode]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query through the pipeline.
        Args:
            query (str): The user's query
            mode (str or PipelineMode, optional): Override the pipeline mode
            **kwargs: Additional parameters for the pipeline
        Returns:
            Dict containing the pipeline's response and metadata
        """
        # Determine the mode
        mode = PipelineMode(mode) if mode is not None else self.config.mode
        # Prepare the result dictionary
        result = {
            "query": query,
            "mode": mode.value,
            "timestamp": datetime.utcnow().isoformat(),
            "results": {}
        }
        try:
            # Process based on the selected mode
            if mode == PipelineMode.ASK:
                result["results"] = self._process_ask_mode(query, **kwargs)
            elif mode == PipelineMode.SUMMARIZE:
                result["results"] = self._process_summarize_mode(query, **kwargs)
            elif mode == PipelineMode.GENERATE:
                result["results"] = self._process_generate_mode(query, **kwargs)
            elif mode == PipelineMode.EVALUATE:
                result["results"] = self._process_evaluate_mode(query, **kwargs)
            elif mode == PipelineMode.FULL:
                result["results"] = self._process_full_mode(query, **kwargs)
            else:
                raise ValueError(f"Unsupported pipeline mode: {mode}")
            # Mark as successful
            result["status"] = "success"
            # Save the result if enabled
            if self.config.save_outputs:
                self._save_pipeline_result(result)
            return result
        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.update({
                "status": "error",
                "error": error_msg,
                "traceback": str(e)
            })
            return result
    def _process_ask_mode(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query in ask mode (retrieval + summarization)."""
        result = {}
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query,
            top_k=self.config.max_retrieved_docs,
            **kwargs.get("retriever_kwargs", {})
        )
        result["retrieved_documents"] = [
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata)
            }
            for doc in retrieved_docs
        ]
        # Step 2: Generate a summary/answer
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        if not context.strip():
            # If nothing was retrieved, summarize the user's query instead of an empty string
            context = query
        summary = self.summarizer.summarize_text(
            context,
            summary_length=self.config.summary_length,
            **kwargs.get("summarizer_kwargs", {})
        )
        result["answer"] = summary
        # Also provide a simplified version for easier consumption
        try:
            result["answer_simplified"] = self.summarizer.simplify(summary, audience="beginner", max_words=160)
        except Exception as _:
            result["answer_simplified"] = summary
        return result
    def _process_summarize_mode(self, text: str, **kwargs) -> Dict[str, Any]:
        """Process a text summarization request."""
        result = {}
        # Generate the summary
        summary = self.summarizer.summarize_text(
            text,
            summary_length=self.config.summary_length,
            **kwargs.get("summarizer_kwargs", {})
        )
        result["summary"] = summary
        return result
    def _process_generate_mode(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Process an image generation request."""
        # Lazy load the generator
        self._initialize_generator()
        if self.generator is None:
            raise RuntimeError("Image generation is not available. Required dependencies may be missing.")
        # Prepare generation config
        base_cfg = self.config.generation_config or {}
        override_cfg = kwargs.get("generation_config", {})
        gen_config = GenerationConfig(**{**base_cfg, **override_cfg})
        # Generate the image
        gen_kwargs = dict(kwargs.get("generator_kwargs", {}))
        # Allow callers to pass context_docs; default to None in generate-only
        gen_kwargs.setdefault("context_docs", None)
        generation_result = self.generator.generate(
            prompt=prompt,
            config=gen_config,
            **gen_kwargs
        )
        return {"generation_result": generation_result}
    def _process_evaluate_mode(self, text: str, **kwargs) -> Dict[str, Any]:
        """Process a text evaluation request."""
        result = {}
        # Evaluate for bias
        bias_eval = self.evaluator.evaluate_bias(text)
        result["bias_evaluation"] = bias_eval
        # Generate explanation
        explanation = self.evaluator.explain_decision(
            text,
            **kwargs.get("explanation_kwargs", {})
        )
        result["explanation"] = explanation
        return result
    def _process_full_mode(self, query: str, **kwargs) -> Dict[str, Any]:
        """Process a query through the full pipeline."""
        result = {}
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query,
            top_k=self.config.max_retrieved_docs,
            **kwargs.get("retriever_kwargs", {})
        )
        result["retrieved_documents"] = [
            {
                "content": doc.page_content,
                "metadata": dict(doc.metadata)
            }
            for doc in retrieved_docs
        ]
        # Step 2: Generate a summary/answer
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        # Fallback: if nothing was retrieved, summarize the user's query instead of an empty string
        if not context.strip():
            context = query
        summary = self.summarizer.summarize_text(
            context,
            summary_length=self.config.summary_length,
            **kwargs.get("summarizer_kwargs", {})
        )
        result["answer"] = summary
        # Step 3: Evaluate the answer if enabled
        if self.config.enable_evaluation:
            evaluation = self.evaluator.evaluate_rag_output(
                query=query,
                context=retrieved_docs,
                generated_answer=summary,
                **kwargs.get("evaluator_kwargs", {})
            )
            result["evaluation"] = evaluation
        # Step 4: Generate an image if generator is available
        # IMPORTANT: Support choosing the image prompt source for clarity
        if kwargs.get("generate_image", False):
            try:
                self._initialize_generator()
                if self.generator is not None:
                    # Merge generation config overrides
                    base_cfg = self.config.generation_config or {}
                    override_cfg = kwargs.get("generation_config", {})
                    gen_cfg = GenerationConfig(**{**base_cfg, **override_cfg})
                    # Determine prompt source: query | summary | custom
                    prompt_source = kwargs.get("image_prompt_source", "query")
                    custom_prompt = kwargs.get("custom_image_prompt", "")
                    style = kwargs.get("image_style", "realistic")
                    if prompt_source == "summary" and isinstance(result.get("answer"), str) and result["answer"].strip():
                        # Use enhanced prompting from summary
                        image_result = self.generator.generate_from_summary(
                            text_summary=result["answer"],
                            additional_prompt="",
                            style=style,
                            config=gen_cfg,
                            **kwargs.get("generator_kwargs", {})
                        )
                    else:
                        gen_cfg = GenerationConfig(**(self.config.generation_config or {}))
                        gen_kwargs = dict(kwargs.get("generator_kwargs", {}))
                        # Provide retrieved documents to generator for flowchart paper nodes
                        gen_kwargs.setdefault("context_docs", result.get("retrieved_documents", []))
                        image_result = self.generator.generate(
                            prompt=(custom_prompt or query).strip(),
                            config=gen_cfg,
                            **gen_kwargs
                        )
                    result["generated_image"] = image_result
            except Exception as e:
                logger.warning(f"Image generation failed: {str(e)}")
                result["generated_image"] = {"error": str(e)}
        # Optional: attach a Graphviz DOT flowchart for this run
        if kwargs.get("generate_diagram", False):
            try:
                num_docs = len(result.get("retrieved_documents", []))
                had_answer = bool(result.get("answer"))
                gen_img = bool(result.get("generated_image"))
                result["diagram_dot"] = self._build_flowchart_dot(query, num_docs, had_answer, gen_img)
            except Exception:
                pass
        return result
    def _json_safe(self, obj):
        """Recursively convert pipeline result into JSON-serializable form."""
        try:
            import PIL.Image  # type: ignore
            pil_image = PIL.Image.Image
        except Exception:
            pil_image = tuple()  # never matches
        if isinstance(obj, dict):
            return {k: self._json_safe(v) for k, v in obj.items() if k != "images"}
        if isinstance(obj, (list, tuple)):
            return [self._json_safe(x) for x in obj]
        if isinstance(obj, pil_image):
            return "<PIL.Image>"
        try:
            json.dumps(obj)  # type: ignore
            return obj
        except Exception:
            return str(obj)

    def _save_pipeline_result(self, result: Dict[str, Any]) -> str:
        """
        Save the pipeline result to a JSON file.
        Args:
            result (Dict): The result to save
        Returns:
            str: Path to the saved file
        """
        try:
            # Create a filename based on timestamp and query
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_query = "".join(c if c.isalnum() else "_" for c in result["query"][:50]).strip("_")
            filename = f"pipeline_result_{timestamp}_{safe_query}.json"
            filepath = os.path.join(self.config.output_dir, filename)
            # Save the result as JSON (strip non-serializable objects like PIL images)
            safe_result = self._json_safe(result)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(safe_result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved pipeline result to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save pipeline result: {str(e)}")
            return ""

    def _build_flowchart_dot(self, query: str, num_docs: int, had_answer: bool, generated_image: bool) -> str:
        """Create a Graphviz DOT describing the pipeline flow for this run."""
        # Basic flow: Query -> Retrieval -> Summarizer -> [Evaluator] -> [ImageGen]
        label_query = query[:60].replace('"', '\"') + ("..." if len(query) > 60 else "")
        dot = [
            "digraph AutoRAG {",
            "  rankdir=LR;",
            "  node [shape=box, style=rounded, color=gray40, fontname=Helvetica];",
            f"  query [label=\"Query: {label_query}\"];",
            f"  retrieve [label=\"Retrieval\\nDocs: {num_docs}\"];",
            f"  summarize [label=\"Summarizer\\nAnswer: {'yes' if had_answer else 'no'}\"];",
            "  evaluate [label=\"Evaluator (optional)\"];",
            f"  image [label=\"Image Gen: {'yes' if generated_image else 'no'}\"];",
            "  query -> retrieve -> summarize -> evaluate -> image;",
            "}"
        ]
        return "\n".join(dot)
def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration."""
    return PipelineConfig(
        mode=PipelineMode.FULL,
        max_retrieved_docs=5,
        summary_length="medium",
        generation_config={
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "width": 512,
            "height": 512
        },
        enable_evaluation=True,
        save_outputs=True,
        output_dir="../outputs"
    )
# Example usage
if __name__ == "__main__":
    # Create a pipeline with default configuration
    pipeline = AgentPipeline()
    # Example query
    query = "Explain the concept of artificial intelligence and its applications."
    # Process the query
    print(f"Processing query: {query}")
    result = pipeline.process_query(
        query,
        mode="full",
        generate_image=True
    )
    # Print the result
    print("\nPipeline Result:")
    print("-" * 50)
    print(f"Status: {result.get('status', 'unknown')}")
    if "results" in result:
        results = result["results"]
        if "answer" in results:
            print("\nAnswer:")
            print("-" * 50)
            print(results["answer"])
        if "evaluation" in results:
            eval_data = results["evaluation"]
            print("\nEvaluation:")
            print("-" * 50)
            print(f"Relevance Score: {eval_data.get('metrics', {}).get('relevance_score', 'N/A'):.2f}")
            print(f"Faithfulness Score: {eval_data.get('metrics', {}).get('faithfulness_score', 'N/A'):.2f}")
        if "generated_image" in results:
            img_result = results["generated_image"]
            if "saved_paths" in img_result and img_result["saved_paths"]:
                print(f"\nGenerated image saved to: {img_result['saved_paths'][0]}")
