# ==============================
# FILE: app.py
# ==============================
import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import base64
from dotenv import load_dotenv
# Import pipeline components
from agent_pipeline import AgentPipeline, PipelineConfig, PipelineMode
# Load environment variables from .env
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AutoRAG - Agentic AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        margin-bottom: 1rem;
    }
    .header {
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)
def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = AgentPipeline()
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = PipelineMode.FULL.value
def display_header():
    """Display the application header."""
    st.markdown("""
    <div class="header">
        <h1>AutoRAG</h1>
        <p>Agentic AI System for Domain-Specific Knowledge Creation and Generation</p>
    </div>
    """, unsafe_allow_html=True)
def display_sidebar():
    """Display the sidebar with mode selection and settings."""
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60.png?text=AutoRAG", width=200)
        st.markdown("### Mode Selection")
        mode = st.radio(
            "Select Operation Mode:",
            options=[mode.value for mode in PipelineMode],
            format_func=lambda x: x.capitalize(),
            index=list(PipelineMode).index(PipelineMode.FULL)
        )
        st.session_state.current_mode = mode
        st.markdown("### Settings")
        st.session_state.max_docs = st.slider(
            "Maximum Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="Maximum number of relevant documents to retrieve from the knowledge base."
        )
        st.session_state.summary_length = st.selectbox(
            "Summary Length",
            options=["brief", "medium", "detailed"],
            index=1,
            help="Length of the generated summary/answer."
        )
        # Image generation settings (shown for Generate and Full modes)
        st.markdown("### Image Generation")
        st.session_state.image_prompt_source = st.selectbox(
            "Prompt Source",
            options=["query", "summary", "custom"],
            index=0,
            help="Use the original query, the generated summary, or a custom prompt for image generation."
        )
        st.session_state.image_style = st.selectbox(
            "Image Style (for summary)",
            options=["realistic", "artistic", "painting", "sketch", "anime", "cyberpunk"],
            index=0,
            help="Applied when using the summary as the prompt to enhance the result."
        )
        st.session_state.custom_image_prompt = st.text_input(
            "Custom Image Prompt",
            value="",
            help="Only used if Prompt Source is set to 'custom'."
        )
        st.session_state.steps = st.slider("Steps", min_value=10, max_value=75, value=35)
        st.session_state.guidance = st.slider("Guidance Scale", min_value=1.0, max_value=12.0, value=7.0, step=0.5)
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.width = st.number_input("Width", min_value=256, max_value=1024, value=768, step=64)
        with c2:
            st.session_state.height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=64)
        st.session_state.seed = st.text_input("Seed (optional)", value="")
        st.session_state.show_diagram = st.checkbox("Show Pipeline Flowchart", value=False, help="Render a flowchart of the pipeline steps for this run.")
        st.markdown("### About")
        st.markdown("""
        AutoRAG is an intelligent retrieval-augmented generation system that combines 
        document retrieval with large language models to provide accurate, 
        context-aware responses.
        """)
def get_icon_for_mode(mode: str) -> str:
    """Return empty icon to keep UI clean and robust across encodings."""
    return ""
def display_query_history():
    """Display the query history in the sidebar."""
    if st.session_state.query_history:
        with st.sidebar:
            st.markdown("### Recent Queries")
            for i, query in enumerate(reversed(st.session_state.query_history[-5:])):
                mode_icon = get_icon_for_mode(query.get('mode', 'ask'))
                if st.sidebar.button(
                    f"{mode_icon} {query['query'][:30]}...",
                    key=f"history_{i}",
                    help=query['query']
                ):
                    st.session_state.current_query = query['query']
                    st.session_state.current_mode = query.get('mode', 'ask')
                    st.rerun()
def process_query(query: str, mode: str) -> Dict[str, Any]:
    """Process a query using the agent pipeline."""
    try:
        # Update pipeline configuration
        config = PipelineConfig(
            mode=PipelineMode(mode),
            max_retrieved_docs=st.session_state.get('max_docs', 5),
            summary_length=st.session_state.get('summary_length', 'medium'),
            enable_evaluation=True,
            save_outputs=True
        )
        # Apply the updated config to the pipeline before processing
        st.session_state.pipeline.config = config
        # Process the query
        with st.spinner(f"Processing {mode} query..."):
            result = st.session_state.pipeline.process_query(
                query=query,
                mode=mode,
                generate_image=(mode in ['generate', 'full']),
                image_prompt_source=st.session_state.get('image_prompt_source', 'query'),
                custom_image_prompt=st.session_state.get('custom_image_prompt', ''),
                image_style=st.session_state.get('image_style', 'realistic'),
                generate_diagram=st.session_state.get('show_diagram', False),
                generation_config={
                    "num_inference_steps": st.session_state.get('steps', 35),
                    "guidance_scale": st.session_state.get('guidance', 7.0),
                    "width": int(st.session_state.get('width', 768)),
                    "height": int(st.session_state.get('height', 512)),
                    **({"seed": int(st.session_state.seed)} if str(st.session_state.get('seed', '')).isdigit() else {})
                }
            )
            # Add to query history
            if query not in [q['query'] for q in st.session_state.query_history]:
                st.session_state.query_history.append({
                    'query': query,
                    'mode': mode,
                    'timestamp': datetime.now().isoformat()
                })
            return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return {"status": "error", "error": str(e)}
def display_result(result: Dict[str, Any]):
    """Display the processing result."""
    if result.get("status") == "error":
        st.error(f"Error: {result.get('error', 'Unknown error occurred')}")
        return
    results = result.get("results", {})
    mode = result.get("mode", "ask")
    # Display answer/summary if available
    if "answer" in results:
        with st.expander("Generated Answer", expanded=True):
            st.markdown(results["answer"])
    # Display evaluation results if available
    if "evaluation" in results and results["evaluation"]:
        with st.expander("Evaluation Results", expanded=False):
            eval_data = results["evaluation"]
            # Display metrics
            if "metrics" in eval_data:
                cols = st.columns(3)
                with cols[0]:
                    st.metric("Relevance", f"{eval_data['metrics'].get('relevance_score', 0) * 100:.1f}%")
                with cols[1]:
                    st.metric("Faithfulness", f"{eval_data['metrics'].get('faithfulness_score', 0) * 100:.1f}%")
                with cols[2]:
                    st.metric("Length", f"{len(eval_data.get('answer', '').split())} words")
            # Display suggestions
            if "suggestions" in eval_data and eval_data["suggestions"]:
                st.markdown("#### Suggestions for Improvement")
                for suggestion in eval_data["suggestions"]:
                    st.markdown(f"- {suggestion}")
    # Display retrieved documents if available
    if "retrieved_documents" in results and results["retrieved_documents"]:
        with st.expander(f"Retrieved Documents ({len(results['retrieved_documents'])}) ", expanded=False):
            for i, doc in enumerate(results["retrieved_documents"], 1):
                with st.container():
                    st.markdown(f"**Document {i}**")
                    st.caption(f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}")
                    st.markdown(doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"])
                    st.markdown("---")
    # Display generated image if available
    if "generated_image" in results and results["generated_image"]:
        img_result = results["generated_image"]
        if "saved_paths" in img_result and img_result["saved_paths"]:
            with st.expander("Generated Image", expanded=True):
                st.image(img_result["saved_paths"][0], use_container_width=True)
                # Show prompt and config used
                st.caption("Prompt and settings used")
                if "prompt" in img_result:
                    st.code(img_result["prompt"], language="text")
                if "config" in img_result:
                    st.json(img_result["config"])
                # Add download button
                with open(img_result["saved_paths"][0], "rb") as f:
                    img_data = f.read()
                    b64 = base64.b64encode(img_data).decode()
                    href = f'<a href="data:image/png;base64,{b64}" download="generated_image.png">Download Image</a>'
                    st.markdown(href, unsafe_allow_html=True)
    # Display pipeline flowchart if present
    if results.get("diagram_dot"):
        with st.expander("Pipeline Flowchart", expanded=False):
            st.graphviz_chart(results["diagram_dot"], use_container_width=True)
def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    # Display header
    display_header()
    # Display sidebar
    display_sidebar()
    # Display query history
    display_query_history()
    # Main content area
    st.markdown("### Enter Your Query")
    # Query input
    query = st.text_area(
        "Ask a question or provide instructions:",
        value=st.session_state.get("current_query", ""),
        placeholder="e.g., Explain the concept of artificial intelligence...",
        height=150,
        key="query_input"
    )
    # Process button
    col1, col2 = st.columns([1, 3])
    with col1:
        process_clicked = st.button(
            "Process",
            use_container_width=True,
            type="primary"
        )
    # Display mode info
    with col2:
        mode_info = {
            "ask": "Answer questions using retrieved documents",
            "summarize": "Generate a concise summary of the input text",
            "generate": "Create an image based on the input prompt",
            "evaluate": "Analyze text for bias and quality",
            "full": "Complete processing with retrieval, generation, and evaluation"
        }
        st.info(f"Mode: {st.session_state.current_mode.capitalize()} - {mode_info.get(st.session_state.current_mode, '')}")
    # Process the query when the button is clicked
    if process_clicked and query.strip():
        result = process_query(query, st.session_state.current_mode)
        display_result(result)
    # Display example queries
    st.markdown("### Example Queries")
    examples = {
        "ask": "What are the key principles of machine learning?",
        "summarize": "Summarize the latest research on large language models.",
        "generate": "A futuristic city with flying cars and neon lights",
        "evaluate": "Analyze this text for bias and clarity: 'The CEO announced that all employees must return to the office full-time.'"
    }
    cols = st.columns(len(examples))
    for i, (mode, example) in enumerate(examples.items()):
        with cols[i]:
            if st.button(
                f"{mode.capitalize()}",
                help=example,
                use_container_width=True
            ):
                st.session_state.current_mode = mode
                st.session_state.current_query = example
                st.rerun()
if __name__ == "__main__":
    main()
