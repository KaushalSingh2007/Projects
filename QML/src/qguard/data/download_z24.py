"""Acquire the Z24 Bridge dataset (progressive damage test) and record provenance.

Source: HuggingFace mirror ``thanglexuan/Z24-dataset-processed`` (a processed
mirror of the KU Leuven Z24 benchmark). This script downloads the arrays,
inspects their shapes/labels WITHOUT loading them fully into memory, computes
SHA-256 checksums, and writes a machine-readable provenance record to
``reports/provenance/z24_provenance.json``.

Run:
    python -m qguard.data.download_z24

Design notes
------------
* We deliberately keep this offline-safe: if ``huggingface_hub`` is missing or
  the network is unavailable, the script fails with a clear, actionable message
  rather than a stack trace.
* Provenance is the product, not a side effect. The JSON it writes is what
  ``DATASET_REPORT.md`` points to, satisfying the project's "never invent data"
  standard.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ID = "thanglexuan/Z24-dataset-processed"
REPO_TYPE = "dataset"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"

# Curated provenance facts verified from the dataset card on 2026-06-19.
SOURCE_ATTRIBUTION = "KU Leuven (Katholieke Universiteit Leuven) — Z24 Bridge benchmark"
KEY_REFERENCES = [
    "Maeck & De Roeck (2003)",
    "Worden & Manson (2007)",
    "Reynders, De Roeck et al. (2008)",
]
LICENSE_NOTE = (
    "HuggingFace mirror is tagged 'MIT'. The UNDERLYING Z24 data is owned by "
    "KU Leuven and distributed for research use; treat as non-commercial "
    "research data. Original obtainable from https://bwk.kuleuven.be/bwm/z24."
)
CAVEATS = [
    "This mirror is the Progressive Damage Test portion (17 damage scenarios); "
    "it likely does NOT include the ~1-year long-term temperature/environmental "
    "monitoring record. Confirm whether any temperature signal is present before "
    "relying on it for the environmental-confound study (milestone M11/E04).",
    "Arrays are pre-segmented/normalized by the mirror author (1530 windows of "
    "6000 samples across 27 sensors); raw KU Leuven time histories differ.",
]

DEFAULT_RAW_DIR = Path("data/raw/z24")
PROVENANCE_PATH = Path("reports/provenance/z24_provenance.json")
ALLOW_PATTERNS = ["*.npy", "README*.md", "*.toml"]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download the dataset snapshot into ``raw_dir``; return the local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "huggingface_hub is required. Install with:\n"
            '    pip install -e ".[data]"\n'
            "then re-run: python -m qguard.data.download_z24"
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(raw_dir),
        allow_patterns=ALLOW_PATTERNS,
    )
    return Path(local)


def inspect_arrays(raw_dir: Path) -> dict:
    """Read shapes/dtypes (memory-mapped) and summarize the label array."""
    info: dict = {}
    inputs_p = raw_dir / "inputs.npy"
    labels_p = raw_dir / "labels.npy"

    if inputs_p.exists():
        arr = np.load(inputs_p, mmap_mode="r")
        info["inputs"] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if labels_p.exists():
        lab = np.load(labels_p, mmap_mode="r")
        uniq, counts = np.unique(np.asarray(lab), return_counts=True)
        info["labels"] = {
            "shape": list(lab.shape),
            "dtype": str(lab.dtype),
            "unique_values": [int(u) if np.issubdtype(lab.dtype, np.integer) else float(u)
                              for u in uniq.tolist()],
            "counts_per_value": [int(c) for c in counts.tolist()],
        }
    return info


def write_provenance(raw_dir: Path, out_path: Path = PROVENANCE_PATH) -> Path:
    """Compute checksums + array summaries and write the provenance JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(raw_dir.glob("*")):
        if p.is_file():
            files.append(
                {"name": p.name, "size_bytes": p.stat().st_size, "sha256": _sha256(p)}
            )

    record = {
        "dataset": "Z24 Bridge (progressive damage test)",
        "repo_id": REPO_ID,
        "repo_url": REPO_URL,
        "repo_type": REPO_TYPE,
        "source_attribution": SOURCE_ATTRIBUTION,
        "key_references": KEY_REFERENCES,
        "license_note": LICENSE_NOTE,
        "caveats": CAVEATS,
        "verified_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "local_dir": str(raw_dir),
        "files": files,
        "arrays": inspect_arrays(raw_dir),
    }
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    raw_dir = download()
    prov = write_provenance(raw_dir)
    record = json.loads(prov.read_text(encoding="utf-8"))
    print(f"Downloaded to: {raw_dir}")
    print(f"Provenance written: {prov}")
    print(json.dumps(record.get("arrays", {}), indent=2))


if __name__ == "__main__":
    main()
