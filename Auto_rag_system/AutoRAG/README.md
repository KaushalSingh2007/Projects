# AutoRAG - Agentic AI System for Domain-Specific Knowledge

![AutoRAG Logo](https://via.placeholder.com/800x200.png?text=AutoRAG+Agentic+AI+System)

AutoRAG is an advanced Retrieval-Augmented Generation (RAG) system that combines document retrieval with large language models to provide accurate, context-aware responses. It features a modular architecture with specialized agents for retrieval, summarization, generation, and evaluation.

## 🚀 Features

- **Document Retrieval**: Efficiently find relevant information from a document collection
- **Intelligent Summarization**: Generate concise and accurate summaries of retrieved content
- **Image Generation**: Create visual representations of text content using Stable Diffusion
- **Bias & Quality Evaluation**: Analyze generated content for potential biases and quality issues
- **Streamlit Web Interface**: User-friendly interface for interacting with the system
- **Modular Architecture**: Easily extensible with custom agents and components

## 🛠️ Installation

1. Get the source code:
   - Option A: Clone your repository
     ```bash
     git clone <your-repo-url>
     cd AutoRAG
     ```
   - Option B: Download the ZIP from your source and extract it, then open the extracted `AutoRAG/` directory in your editor.

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## 🏃‍♂️ Quick Start

1. **Prepare your documents**:
   - Place your PDF documents in the `data/pdfs/` directory
   - Or add text files to `data/text/`

2. **Process your documents and create embeddings**:
   ```bash
   python data_loader.py
   ```

3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501` and start interacting with AutoRAG!

## 🧩 System Architecture

```
AutoRAG/
├── agents/                        # AI agent implementations
│   ├── retriever_agent.py         # Document retrieval agent (FAISS/embeddings)
│   ├── summarizer_agent.py        # Summarization via Google Gemini (langchain_google_genai)
│   ├── generator_agent.py         # Stable Diffusion image generation (diffusers)
│   └── evaluator_agent.py         # Bias/quality evaluation utilities
├── data/                          # Domain data storage
│   └── pdfs/                      # Source PDF documents (put your PDFs here)
├── embeddings/                    # Vector DB storage (generated)
│   └── faiss_index/               # Saved FAISS index (created by data_loader.py)
├── outputs/                       # Generated artifacts
│   └── images/                    # Images saved by generator agent
├── app.py                         # Streamlit UI entrypoint
├── agent_pipeline.py              # Orchestrates all agents and modes
├── data_loader.py                 # Builds FAISS index from PDFs and optional JSONL
├── rag_chain.py                   # RAG helpers (if used)
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (e.g., GOOGLE_API_KEY)
└── README.md                      # This file
```

## 🤖 Available Agents

1. **Retriever Agent**: Finds relevant documents based on user queries
2. **Summarizer Agent**: Generates concise summaries of retrieved content (Gemini via `langchain_google_genai` with local Transformers fallback on quota errors)
3. **Generator Agent**: Creates images from text descriptions
4. **Evaluator Agent**: Analyzes content for bias, quality, and relevance

## 🌐 Web Interface

The Streamlit-based web interface provides an intuitive way to interact with AutoRAG:

- **Ask Questions**: Get answers based on your document collection
- **Summarize Content**: Generate concise summaries of text or documents
- **Generate Images**: Create visual representations of concepts
- **Evaluate Content**: Analyze text for potential issues and biases

![Web Interface](https://via.placeholder.com/800x500.png?text=AutoRAG+Web+Interface)

---

## 🔁 How the Pipeline Works

The central orchestrator is `agent_pipeline.py` and its `AgentPipeline` class. It wires up agents and executes one of several modes. High-level flow in `FULL` mode (`AgentPipeline._process_full_mode()`):

1. **Retrieve documents** using `RetrieverAgent.retrieve()` from `agents/retriever_agent.py`.
   - Uses FAISS index at `embeddings/faiss_index/` with `HuggingFaceEmbeddings` (`sentence-transformers/all-MiniLM-L6-v2`).
   - If `similarity_search_with_score()` yields no results, it falls back to `similarity_search()` to avoid zero-return edge cases.

2. **Summarize context** with `SummarizerAgent.summarize_text()` in `agents/summarizer_agent.py`.
   - The retrieved chunks are concatenated; if retrieval returns nothing, the pipeline now gracefully summarizes the user’s query instead of an empty string.
   - Primary LLM: Google Gemini via `langchain_google_genai.ChatGoogleGenerativeAI` (model defaults to `gemini-1.5-flash-002`).
   - Optional local fallback summarizer (`transformers` pipeline) activates on quota/429 errors if enabled.

3. **Evaluate answer** with `EvaluatorAgent` (optional; enabled by default in UI’s sidebar-configurable `PipelineConfig`).

4. **Generate image** with `GeneratorAgent` (optional; in FULL mode this uses your original query as the prompt so visuals match intent). Images are saved under `outputs/images/`.

5. **Persist outputs**: A JSON snapshot is saved under `outputs/` by `_save_pipeline_result()`, excluding non-serializable objects.

Streamlit UI (`app.py`) provides mode selection and settings. Sidebar selections are applied to the running pipeline’s `PipelineConfig` before processing.

---

## 🧭 Operation Modes

- **ask**: Retrieve documents and produce an answer/summary based on them (falls back to summarizing the query if nothing is retrieved).
- **summarize**: Summarize arbitrary input text you provide.
- **generate**: Create an image from the prompt using Stable Diffusion.
- **evaluate**: Analyze text for bias/quality.
- **full**: Run retrieval → summarization → (optional) evaluation → (optional) image generation.

You can select the mode in the Streamlit sidebar. The app also tracks recent queries.

---

## ⚙️ Configuration (.env)

Create a `.env` file in the project root with relevant keys:

```
# Google Gemini (required for cloud summarization)
GOOGLE_API_KEY=your_google_api_key

# Optional: specify Gemini model
GEMINI_MODEL=gemini-1.5-flash-002

# Local fallback summarizer (optional; used on quota/429 errors)
ENABLE_LOCAL_SUMMARIZER=true
LOCAL_SUMMARIZER_MODEL=sshleifer/distilbart-cnn-12-6
```

Notes:

- The retriever/summarizer embeddings run on CPU by default for compatibility; you can tune devices in code if you have GPU.
- The image generator may enable attention slicing to reduce VRAM usage. Safety checker can be configured in `generator_agent.py`.

---

## 🧱 Building the Knowledge Base (FAISS Index)

1. Place your PDFs under `data/pdfs/`.
2. Optionally, ingest arXiv JSONL metadata by enabling the flag and setting a path.
3. Build the index:

```bash
python data_loader.py \
  --data-path data/pdfs/ \
  --save-path embeddings/faiss_index \
  --include-arxiv \
  --arxiv-json "archive (3)/arxiv-metadata-oai-snapshot.json" \
  --arxiv-max 500   # optional cap
```

On success, the FAISS index is written to `embeddings/faiss_index/`. The retriever auto-loads it on app start.

---

## 📤 Outputs

- Pipeline JSON snapshots: `outputs/pipeline_result_<timestamp>_<query>.json`
- Generated images: `outputs/images/*.png`
- The UI shows retrieved snippets, the generated answer, evaluation metrics (if enabled), and the image with a download link.

---

## 🛠️ Troubleshooting

- **No documents retrieved**: Ensure you built the FAISS index and it contains data. The pipeline will now summarize your query when no docs are found, but retrieval is key for grounded answers.
- **Gemini quota/429**: The summarizer can fall back to a local Transformers model if `ENABLE_LOCAL_SUMMARIZER=true`.
- **Streamlit deprecation warning**: The app uses `use_container_width=True` for images to avoid deprecation issues.
- **LangChain deprecations**:
  - `HuggingFaceEmbeddings` is moving to `langchain-huggingface`. Consider migrating imports and `requirements.txt` accordingly.
  - `Chain.run` is deprecated; future refactors may use `chain.invoke()`.
- **Diffusers safety checker**: If disabled, ensure you don’t expose unfiltered results in public apps. Review `generator_agent.py` comments.

---

## 📚 Documentation

For detailed documentation, including API references and development guides, please visit our [documentation site](https://your-docs-url.com).

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest new features.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or feedback, please contact [your-email@example.com](mailto:your-email@example.com) or open an issue on GitHub.

---

<div align="center">
  Made with ❤️ by Your Name | [Twitter](https://twitter.com/yourhandle) | [GitHub](https://github.com/yourusername)
</div>
