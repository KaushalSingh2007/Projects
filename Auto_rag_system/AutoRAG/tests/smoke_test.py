"""
CLI smoke test for AutoRAG.

Runs the AgentPipeline across ASK, SUMMARIZE, GENERATE, and FULL modes
and prints concise results to stdout. Exits with non-zero code if any
critical step fails. Designed to be run from the project root (AutoRAG/):

    python -m tests.smoke_test

Environment:
- Requires .env with GOOGLE_API_KEY for Gemini (or local fallback enabled in SummarizerAgent).
- Assumes embeddings/faiss_index exists for retrieval (ASK/FULL). If missing, the
  pipeline will still summarize the user query directly.
"""
from __future__ import annotations
import os
import sys
import json
import traceback
from typing import Dict, Any
from dotenv import load_dotenv

# Ensure we can import project modules when executed as a module
# (python -m tests.smoke_test) from project root.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent_pipeline import AgentPipeline, PipelineConfig, PipelineMode  # noqa: E402

load_dotenv()

def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_mode(pipeline: AgentPipeline, mode: str, payload: str) -> Dict[str, Any]:
    """Run a single mode and capture a concise JSON-like response."""
    try:
        if mode == "ask":
            res = pipeline.process_query(payload, mode=PipelineMode.ASK)
        elif mode == "summarize":
            res = pipeline.process_query(payload, mode=PipelineMode.SUMMARIZE)
        elif mode == "generate":
            res = pipeline.process_query(
                payload,
                mode=PipelineMode.GENERATE,
                generation_config={"num_inference_steps": 15, "guidance_scale": 6.5, "width": 640, "height": 384},
                generator_kwargs={"save_to_disk": True, "return_pil": False},
            )
        elif mode == "full":
            res = pipeline.process_query(
                payload,
                mode=PipelineMode.FULL,
                generate_image=True,
                generator_kwargs={"save_to_disk": True, "return_pil": False},
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        return res
    except Exception:
        return {"status": "error", "error": traceback.format_exc()}


def main() -> int:
    # Configure a lightweight default
    config = PipelineConfig(
        mode=PipelineMode.FULL,
        max_retrieved_docs=3,
        summary_length="brief",
        generation_config={"num_inference_steps": 15, "guidance_scale": 6.5, "width": 640, "height": 384},
        enable_evaluation=False,
        save_outputs=False,
        output_dir="outputs",
    )

    pipeline = AgentPipeline(config=config)

    # Test payloads
    ask_q = "What are embeddings in machine learning?"
    sum_text = (
        "Large language models (LLMs) are neural networks trained on massive corpora to predict the next token.\n"
        "They can be adapted for tasks like question answering, summarization, and code generation."
    )
    gen_prompt = "A clean, photorealistic background with subtle abstract shapes, professional, studio lighting"
    full_q = "Explain RAG (Retrieval Augmented Generation) in simple terms"

    failures = 0

    # ASK
    print_section("ASK MODE")
    res = run_mode(pipeline, "ask", ask_q)
    print(json.dumps({k: res.get(k) for k in ("status", "mode")}, indent=2))
    if res.get("status") != "success":
        failures += 1
    else:
        ans = res.get("results", {}).get("answer", "")
        print("Answer:", (ans[:200] + "...") if len(ans) > 200 else ans)

    # SUMMARIZE
    print_section("SUMMARIZE MODE")
    res = run_mode(pipeline, "summarize", sum_text)
    print(json.dumps({k: res.get(k) for k in ("status", "mode")}, indent=2))
    if res.get("status") != "success":
        failures += 1
    else:
        summ = res.get("results", {}).get("summary", "")
        print("Summary:", (summ[:200] + "...") if len(summ) > 200 else summ)

    # GENERATE
    print_section("GENERATE MODE")
    res = run_mode(pipeline, "generate", gen_prompt)
    print(json.dumps({k: res.get(k) for k in ("status", "mode")}, indent=2))
    if res.get("status") != "success":
        failures += 1
    else:
        gen_res = res.get("results", {}).get("generation_result", {})
        print("Image saved:", gen_res.get("saved_paths", ["<none>"])[0])

    # FULL
    print_section("FULL MODE")
    res = run_mode(pipeline, "full", full_q)
    print(json.dumps({k: res.get(k) for k in ("status", "mode")}, indent=2))
    if res.get("status") != "success":
        failures += 1
    else:
        ans = res.get("results", {}).get("answer", "")
        print("Answer:", (ans[:200] + "...") if len(ans) > 200 else ans)
        img = res.get("results", {}).get("generated_image", {})
        print("Image saved:", img.get("saved_paths", ["<none>"])[0] if isinstance(img, dict) else "<none>")

    print_section("SUMMARY")
    if failures:
        print(f"Smoke test completed with {failures} failure(s)")
        return 1
    print("All smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
