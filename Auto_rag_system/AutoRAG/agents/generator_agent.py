# ==============================
# FILE: agents/generator_agent.py
# ==============================
import os
import torch
from typing import Optional, Union, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import re
from PIL import Image, ImageDraw, ImageFont
import json
from langchain_google_genai import ChatGoogleGenerativeAI
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Check for GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    # Fewer but higher-quality steps by default; can be raised as needed
    num_inference_steps: int = 35
    guidance_scale: float = 7.0
    # Strong default negative prompt to suppress artifacts, random text, watermarks
    negative_prompt: Optional[str] = (
        "text, watermark, logo, signature, blurry, lowres, jpeg artifacts,"
        " worst quality, low quality, deformed, distorted, extra fingers, malformed,"
        " nsfw, cropped, frame, border"
    )
    width: int = 768
    height: int = 512
    num_images: int = 1
    seed: Optional[int] = None
class GeneratorAgent:
    """
    Agent responsible for generating images from text prompts using Stable Diffusion.
    Handles model loading, image generation, and result management.
    """
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "outputs/images",
        use_safetensors: bool = True,
        torch_dtype: torch.dtype = torch.float16 if DEVICE == "cuda" else torch.float32,
        **kwargs
    ):
        """
        Initialize the generator agent.
        Args:
            model_id (str): Hugging Face model ID or path to local model
            output_dir (str): Directory to save generated images
            use_safetensors (bool): Whether to use safetensors format if available
            torch_dtype (torch.dtype): Data type for model weights
            **kwargs: Additional model loading arguments
        """
        self.model_id = model_id
        self.output_dir = output_dir
        self.torch_dtype = torch_dtype
        self.pipeline = None
        self.generator = None
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        # Do NOT initialize Stable Diffusion pipeline here; we only render mindmaps now.
        self.llm = None
    def _init_llm(self):
        if self.llm is not None:
            return self.llm
        try:
            model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-002")
            self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.2, convert_system_message_to_human=True)
        except Exception:
            self.llm = None
        return self.llm
    def _initialize_pipeline(self, **kwargs):
        """Initialize the Stable Diffusion pipeline."""
        try:
            from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
            logger.info(f"Loading model: {self.model_id}")
            # Load the pipeline with specified configuration
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                safety_checker=None,  # Disable safety checker for more control
                use_safetensors=kwargs.pop("use_safetensors", True),
                **kwargs
            )
            # Move to appropriate device
            self.pipeline = self.pipeline.to(DEVICE)
            # Use a high-quality scheduler (Karras) to reduce mushy/text-like artifacts
            try:
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config,
                    use_karras=True,
                    algorithm_type="sde-dpmsolver++",
                )
            except Exception:
                pass
            # Enable attention slicing if using a GPU with limited VRAM
            if DEVICE == "cuda" and torch.cuda.get_device_properties(0).total_memory < 16e9:  # < 16GB VRAM
                self.pipeline.enable_attention_slicing()
                logger.info("Enabled attention slicing for lower VRAM usage")
            logger.info("Model loaded successfully")
        except ImportError:
            error_msg = "Failed to import required diffusers library. Please install it with: pip install diffusers transformers"
            logger.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    def set_seed(self, seed: Optional[int] = None):
        """Set the random seed for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            if DEVICE == "cuda":
                torch.cuda.manual_seed_all(seed)
            self.generator = torch.Generator(device=DEVICE).manual_seed(seed)
            logger.info(f"Set random seed to {seed}")
        return self
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        save_to_disk: bool = True,
        return_pil: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        Args:
            prompt (str): Text prompt for image generation
            config (GenerationConfig, optional): Configuration for generation
            save_to_disk (bool): Whether to save the generated image to disk
            return_pil (bool): Whether to return the PIL image
            **kwargs: Additional generation parameters
        Returns:
            Dict containing the generated image(s) and metadata
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        # Always render a mindmap diagram (no Stable Diffusion). Optionally enrich with research papers via context_docs.
        context_docs = kwargs.get("context_docs")
        return self._generate_mindmap(prompt, save_to_disk=save_to_disk, return_pil=return_pil, context_docs=context_docs)
    def generate_from_summary(
        self,
        text_summary: str,
        additional_prompt: str = "",
        style: str = "realistic",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an image from a text summary with enhanced prompt engineering.
        Args:
            text_summary (str): Text summary to generate image from
            additional_prompt (str): Additional instructions for the generator
            style (str): Style of the generated image (realistic, artistic, etc.)
            **kwargs: Additional generation parameters
        Returns:
            Dict containing the generated image(s) and metadata
        """
        if not text_summary:
            raise ValueError("Text summary cannot be empty")
        # Create an enhanced prompt based on the summary and style
        style_prompts = {
            "realistic": "highly detailed, photorealistic, 8k, professional photography",
            "artistic": "digital art, concept art, trending on artstation, highly detailed",
            "painting": "oil painting, brush strokes, artistic, masterpiece",
            "sketch": "pencil sketch, black and white, detailed line art",
            "anime": "anime style, vibrant colors, highly detailed, studio ghibli"
        }
        style_prompt = style_prompts.get(style.lower(), style_prompts["realistic"])
        # Construct the final prompt (sanitize to avoid text overlays)
        clean_summary = self._sanitize_prompt(text_summary)
        clean_additional = self._sanitize_prompt(additional_prompt)
        final_prompt = f"{clean_summary}. {clean_additional} {style_prompt}. no text, no watermark, no logo"
        # Ensure negative prompt explicitly discourages text/watermarks
        cfg = kwargs.pop("config", None) or GenerationConfig()
        if cfg.negative_prompt:
            cfg.negative_prompt += ", text, watermark, logo"
        else:
            cfg.negative_prompt = "text, watermark, logo"
        # Generate the image
        return self.generate(final_prompt, config=cfg, **kwargs)

    def _sanitize_prompt(self, text: str) -> str:
        """Clean bullets/headings and excessive punctuation to reduce text artifacts in SD output."""
        if not text:
            return ""
        # Normalize newlines and remove markdown-like bullets/headings
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cleaned = []
        for ln in lines:
            # Remove common bullet markers and numbering at line start
            ln = re.sub(r"^([\-\*•\u2022\u2023\u25E6\u2043\u2219]+)\s+", "", ln)
            ln = re.sub(r"^(\d+\.|\d+\))\s+", "", ln)
            ln = re.sub(r"^[A-Za-z]\)\s+", "", ln)
            cleaned.append(ln)
        text = ", ".join(cleaned)
        # Collapse multiple spaces and remove stray pipe/backticks
        text = re.sub(r"\s+", " ", text)
        text = text.replace("|", " ").replace("`", " ")
        # Discourage explicit phrases that cause overlaid text
        text = re.sub(r"\b(title|header|bullet|list|slide|caption|subtitle|text)\b", "", text, flags=re.I)
        return text.strip()
    def _is_mindmap_prompt(self, prompt: str) -> bool:
        if not prompt:
            return False
        p = prompt.strip().lower()
        return p.startswith("mindmap:") or p.startswith("mind map:")
    def _parse_edges(self, body: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Parse simple edge definitions like 'A->B->C, D->E' into nodes and edges."""
        edges: List[Tuple[str, str]] = []
        nodes_set = set()
        # split by commas or newlines
        parts = re.split(r"[\n,]", body)
        for part in parts:
            seq = [s.strip() for s in re.split(r"->|→|⇒|➜", part) if s.strip()]
            if len(seq) >= 2:
                for a, b in zip(seq, seq[1:]):
                    nodes_set.add(a)
                    nodes_set.add(b)
                    edges.append((a, b))
        nodes = sorted(nodes_set)
        return nodes, edges
    def _generate_mindmap(self, prompt: str, save_to_disk: bool, return_pil: bool, context_docs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Render a mindmap. If USE_GEMINI_SVG_ONLY=true and LLM available, request a full SVG from Gemini and save it directly. Otherwise use local renderer."""
        body = re.sub(r"^(mind\s*map:|mindmap:|flow\s*chart:|flowchart:|diagram:)\s*", "", prompt.strip(), flags=re.I)
        # Gemini SVG-only path
        if os.getenv("USE_GEMINI_SVG_ONLY", "true").lower() in ("1", "true", "yes") and self._init_llm() is not None:
            try:
                logger.info("[Mindmap] Requesting direct SVG from Gemini")
                svg = self._svg_mindmap_from_gemini(body or prompt, context_docs)
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.output_dir, f"{timestamp}_mindmap.svg")
                if save_to_disk:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(svg)
                    logger.info(f"Saved generated SVG to: {filepath}")
                return {
                    "prompt": prompt,
                    "model_id": "mindmap-gemini-svg",
                    "device": DEVICE,
                    "generated_at": datetime.utcnow().isoformat(),
                    "saved_paths": [filepath] if save_to_disk else [],
                }
            except Exception as e:
                logger.warning(f"[Mindmap] Gemini SVG failed, falling back to local diagram: {e}")
        center, branches = None, None
        use_gemini = os.getenv("USE_GEMINI_MINDMAP", "true").lower() in ("1", "true", "yes")
        if use_gemini and self._init_llm() is not None:
            try:
                logger.info("[Mindmap] Using Gemini to derive structure")
                center, branches = self._mindmap_from_gemini(body or prompt, context_docs)
            except Exception:
                logger.warning("[Mindmap] Gemini failed, falling back to heuristic parsing")
                center, branches = None, None
        if center is None or branches is None:
            center, branches = self._extract_mindmap(body or prompt)
        # Augment with research papers as a separate branch
        paper_nodes: List[str] = []
        if context_docs:
            for d in context_docs[:6]:
                meta = d.get("metadata", {}) if isinstance(d, dict) else {}
                title = meta.get("title") or (d.get("content", "").splitlines()[0] if isinstance(d, dict) else None)
                if title:
                    paper_nodes.append(title.strip())
        if paper_nodes:
            branches.append({"label": "Research Papers", "children": paper_nodes})
        # Canvas settings (adaptive to content)
        base_w, base_h = 1600, 1100
        extra_w = 200 * max(0, len(branches) - 8)
        W, H = base_w + extra_w, base_h
        cx, cy = W // 2, H // 2
        img = Image.new("RGB", (W, H), color=(248, 249, 250))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            font_center = ImageFont.truetype("arial.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
            font_center = font
        # Draw center
        center_r = 110
        center_box = [cx - center_r, cy - center_r, cx + center_r, cy + center_r]
        draw.ellipse(center_box, fill=(255, 255, 255), outline=(41, 128, 185), width=4)
        self._draw_text_centered(draw, center, (cx, cy), font_center, max_w=center_r * 2 - 16)
        # Place first-level branches around one or two rings for spacing
        import math
        n = max(1, len(branches))
        inner_radius = 360
        outer_radius = 520
        node_w, node_h = 260, 80
        child_w, child_h = 220, 60
        # Two-ring distribution if many branches
        if n <= 8:
            rings = [(n, inner_radius)]
        else:
            n_inner = 8
            n_outer = n - n_inner
            rings = [(n_inner, inner_radius), (n_outer, outer_radius)]
        idx = 0
        branch_positions: List[Tuple[int, int, float]] = []  # x,y,angle
        for count, r in rings:
            if count <= 0:
                continue
            angle_step = 2 * math.pi / count
            # offset outer ring angles for better separation
            offset = 0 if r == inner_radius else angle_step / 2
            for k in range(count):
                angle = -math.pi / 2 + offset + k * angle_step
                bx = int(cx + r * math.cos(angle))
                by = int(cy + r * math.sin(angle))
                branch_positions.append((bx, by, angle))
        # Draw branches
        for (bx, by, angle), br in zip(branch_positions, branches):
            # Curved connector: polyline via control point halfway
            mx = int((cx + bx) / 2 + 60 * math.cos(angle + math.pi / 2))
            my = int((cy + by) / 2 + 60 * math.sin(angle + math.pi / 2))
            draw.line([(cx, cy), (mx, my), (bx, by)], fill=(41, 128, 185), width=3)
            # Draw branch box
            rect = [bx - node_w // 2, by - node_h // 2, bx + node_w // 2, by + node_h // 2]
            draw.rounded_rectangle(rect, radius=12, outline=(39, 174, 96), width=3, fill=(255, 255, 255))
            label = br["label"] if isinstance(br, dict) else str(br)
            self._draw_text_centered(draw, label, (bx, by), font, max_w=node_w - 20)
            # Sub-branches (children) along tangent for clarity
            children = br.get("children", []) if isinstance(br, dict) else []
            if children:
                # Tangent unit vector at angle
                tx = math.cos(angle + math.pi / 2)
                ty = math.sin(angle + math.pi / 2)
                # Place children spaced along tangent, offset outward from branch
                spacing = child_w + 40
                start_offset = -((len(children) - 1) / 2) * spacing
                base_out = 170  # outward from branch center
                for j, child in enumerate(children):
                    off = start_offset + j * spacing
                    cx2 = int(bx + base_out * math.cos(angle) + off * tx)
                    cy2 = int(by + base_out * math.sin(angle) + off * ty)
                    # connector (curved polyline)
                    cmx = int((bx + cx2) / 2 + 40 * tx)
                    cmy = int((by + cy2) / 2 + 40 * ty)
                    draw.line([(bx, by), (cmx, cmy), (cx2, cy2)], fill=(127, 140, 141), width=2)
                    rect2 = [cx2 - child_w // 2, cy2 - child_h // 2, cx2 + child_w // 2, cy2 + child_h // 2]
                    draw.rounded_rectangle(rect2, radius=10, outline=(127, 140, 141), width=2, fill=(255, 255, 255))
                    self._draw_text_centered(draw, str(child), (cx2, cy2), font, max_w=child_w - 16)
        # Save and return
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_mindmap_0.png"
        filepath = os.path.join(self.output_dir, filename)
        if save_to_disk:
            img.save(filepath)
            logger.info(f"Saved generated image to: {filepath}")
        result = {
            "prompt": prompt,
            "model_id": "mindmap-renderer",
            "device": DEVICE,
            "generated_at": datetime.utcnow().isoformat(),
            "saved_paths": [filepath] if save_to_disk else [],
        }
        if return_pil:
            result["images"] = [img]
        return result
    def _svg_mindmap_from_gemini(self, text: str, context_docs: Optional[List[Dict[str, Any]]]) -> str:
        doc_titles = []
        if context_docs:
            for d in context_docs[:8]:
                meta = d.get("metadata", {}) if isinstance(d, dict) else {}
                title = meta.get("title") or (d.get("content", "").splitlines()[0] if isinstance(d, dict) else None)
                if title:
                    doc_titles.append(title.strip())
        sys_instr = (
            "You are an SVG diagram generator. Create a clean, spacious radial mindmap as pure SVG markup. "
            "Requirements: central circle for the topic; 6-10 first-level branches arranged around it; each branch 2-5 child nodes connected by curved lines; readable sans-serif font; no external assets; white background; generous spacing. "
            "Return ONLY raw SVG (starting with <svg ...> and ending with </svg>), no markdown, no explanations."
        )
        user = {
            "topic": text,
            "papers": doc_titles,
            "include_papers_branch": True if doc_titles else False,
        }
        prompt = f"{sys_instr}\nINPUT:\n{json.dumps(user)}\nOUTPUT SVG:"
        resp = self.llm.invoke(prompt)
        content = getattr(resp, "content", None) or str(resp)
        # extract SVG block if wrapped
        m = re.search(r"<svg[\s\S]*?</svg>", content, flags=re.I)
        svg = m.group(0) if m else content.strip()
        if not svg.lower().lstrip().startswith("<svg"):
            raise ValueError("Gemini did not return SVG")
        return svg
    def _mindmap_from_gemini(self, text: str, context_docs: Optional[List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
        doc_titles = []
        if context_docs:
            for d in context_docs[:8]:
                meta = d.get("metadata", {}) if isinstance(d, dict) else {}
                title = meta.get("title") or (d.get("content", "").splitlines()[0] if isinstance(d, dict) else None)
                if title:
                    doc_titles.append(title.strip())
        sys_instr = (
            "You are a mindmap planner. Output ONLY valid JSON with keys: center (str), branches (list). "
            "Each branch: {\"label\": str, \"children\": [str]}. Focus on precise keyword-level nodes: definitions, named theorems/laws, key concepts, standard methods/algorithms, core distributions/categories, typical applications, pitfalls. "
            "Constraints: 6-10 first-level branches; each 2-5 children; avoid duplicates; keep items short (2-5 words); prefer canonical names (e.g., 'Central Limit Theorem'). No prose, no explanations, no markdown."
        )
        user_msg = {
            "topic": text,
            "papers": doc_titles,
        }
        prompt = f"{sys_instr}\nINPUT:\n{json.dumps(user_msg)}\nOUTPUT:"
        resp = self.llm.invoke(prompt)
        content = getattr(resp, "content", None) or str(resp)
        try:
            data = json.loads(content)
        except Exception:
            # try to extract JSON block
            m = re.search(r"\{[\s\S]*\}$", content.strip())
            if m:
                data = json.loads(m.group(0))
            else:
                raise
        center = str(data.get("center", "Mindmap")).strip() or "Mindmap"
        branches_raw = data.get("branches", []) or []
        branches: List[Dict[str, Any]] = []
        for b in branches_raw:
            if isinstance(b, dict) and b.get("label"):
                children = b.get("children", []) or []
                if isinstance(children, list):
                    branches.append({"label": str(b["label"]), "children": [str(c) for c in children[:6]]})
        if not branches:
            raise ValueError("empty branches from LLM")
        return center, branches

    def _draw_text_centered(self, draw: ImageDraw.ImageDraw, text: str, center_xy: Tuple[int, int], font: ImageFont.ImageFont, max_w: int) -> None:
        # Break into lines to fit width
        words = text.strip().split()
        lines: List[str] = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            tw, _ = draw.textbbox((0, 0), test, font=font)[2:]
            if tw <= max_w or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        total_h = sum(draw.textbbox((0, 0), ln, font=font)[3] - draw.textbbox((0, 0), ln, font=font)[1] for ln in lines) + (len(lines) - 1) * 4
        y = center_xy[1] - total_h // 2
        for ln in lines:
            tw, th = draw.textbbox((0, 0), ln, font=font)[2:]
            x = center_xy[0] - tw // 2
            draw.text((x, y), ln, fill=(44, 62, 80), font=font)
            y += th + 4

    def _extract_mindmap(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Heuristically extract a center label and branches from free text or simple 'A->B->C' sequences.
        Returns: center, list of {label: str, children: List[str]}
        """
        # If arrows present, use first token as center, chain as nested children
        arrow_parts = [s.strip() for s in re.split(r"->|→|⇒|➜", text) if s.strip()]
        if len(arrow_parts) >= 2:
            center = arrow_parts[0]
            rest = arrow_parts[1:]
            branches: List[Dict[str, Any]] = []
            if rest:
                branches.append({"label": rest[0], "children": rest[1:]})
            return center, branches
        # Else parse "Topic: a, b, c | Other: x, y" style if present
        if ":" in text:
            parts = re.split(r"\|", text)
            # Center: first segment before ':' of first part
            first = parts[0]
            center = first.split(":")[0].strip() or "Mindmap"
            branches: List[Dict[str, Any]] = []
            for seg in parts:
                if ":" in seg:
                    key, vals = seg.split(":", 1)
                    items = [v.strip() for v in re.split(r",|;|/|\n", vals) if v.strip()]
                    if key.strip() and items:
                        branches.append({"label": key.strip(), "children": items})
            if branches:
                return center, branches
        # Fallback: keyword extraction from text
        center = (text.strip()[:60] or "Mindmap").strip()
        keywords = self._extract_keywords(text)
        branches = [{"label": kw, "children": []} for kw in keywords[:8]]
        return center, branches

    def _extract_keywords(self, text: str) -> List[str]:
        stop = set("""
            the a an of to in for on with and or as by from is are be that this these those it its into over under
            about using use via at between within without than then which who where when how what why can could may might
            we you they I he she them his her their our your not also more most less least many much very such including etc
        """.split())
        # split on non-alphanum, dedupe, keep length>=3
        tokens = re.split(r"[^A-Za-z0-9_]+", text.lower())
        uniq = []
        for t in tokens:
            if len(t) >= 3 and t not in stop and t not in uniq:
                uniq.append(t)
        # Prefer capitalized words from original text as well
        caps = re.findall(r"\b([A-Z][a-zA-Z0-9_]{2,})\b", text)
        ordered = list(dict.fromkeys([*caps, *uniq]))
        return ordered
    def _render_text_image(self, text: str, save_to_disk: bool, return_pil: bool) -> Dict[str, Any]:
        img = Image.new("RGB", (1024, 256), color=(248, 249, 250))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((20, 100), text[:500], fill=(44, 62, 80), font=font)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_text_0.png"
        filepath = os.path.join(self.output_dir, filename)
        if save_to_disk:
            img.save(filepath)
            logger.info(f"Saved generated image to: {filepath}")
        result = {
            "prompt": text,
            "model_id": "text-renderer",
            "device": DEVICE,
            "generated_at": datetime.utcnow().isoformat(),
            "saved_paths": [filepath] if save_to_disk else [],
        }
        if return_pil:
            result["images"] = [img]
        return result
    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple images from a list of prompts.
        Args:
            prompts (List[str]): List of text prompts
            config (GenerationConfig, optional): Configuration for generation
            **kwargs: Additional generation parameters
        Returns:
            List of results for each prompt
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, config=config, **kwargs)
            results.append(result)
        return results
# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = GeneratorAgent(
        model_id="runwayml/stable-diffusion-v1-5",
        output_dir="../outputs/images"
    )
    # Example prompt
    prompt = "A beautiful landscape with mountains and a lake at sunset, highly detailed, 8k"
    # Configure generation
    config = GenerationConfig(
        num_inference_steps=30,
        guidance_scale=7.5,
        width=768,
        height=512,
        seed=42
    )
    # Generate the image
    print(f"Generating image for prompt: {prompt}")
    result = generator.generate(prompt, config=config)
    if "saved_paths" in result:
        print(f"\nImage saved to: {result['saved_paths'][0]}")
    # Example of generating from a summary
    summary = "A futuristic city with flying cars and neon lights"
    print(f"\nGenerating image from summary: {summary}")
    result = generator.generate_from_summary(
        summary,
        style="cyberpunk",
        config=GenerationConfig(
            num_inference_steps=50,
            guidance_scale=8.0,
            width=768,
            height=512
        )
    )
    if "saved_paths" in result:
        print(f"Summary image saved to: {result['saved_paths'][0]}")
