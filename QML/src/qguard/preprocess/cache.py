"""Cache a preprocessed `DatasetSplit` to disk, reproducibly and auditably.

Layout (under ``out_dir``, default ``data/processed/z24``):
    train_signals.npy     # healthy-only, preprocessed (label-free)
    eval_signals.npy      # preprocessed
    eval_labels.npy       # sealed on load into a LabelVault (evaluator-only)
    split_meta.json       # reconstructs SplitMetadata (indices/groups) -> no-leakage
    manifest.json         # pipeline config + fitted-state summary + provenance + shapes

The cache stores TRANSFORMED arrays, so using it never requires re-fitting the
pipeline. `manifest.json` records exactly how they were produced (transform
configs, seed, source-data checksum reference), keeping processed data auditable.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..data.schema import DatasetSplit, LabelVault, Recording, SplitMetadata
from .pipeline import Pipeline

DEFAULT_OUT_DIR = Path("data/processed/z24")
PROVENANCE_REF = "reports/provenance/z24_provenance.json"


def _split_to_dict(s: SplitMetadata) -> dict:
    return {
        "strategy": s.strategy,
        "healthy_labels": list(s.healthy_labels),
        "segments_per_group": s.segments_per_group,
        "seed": s.seed,
        "train_indices": s.train_indices.tolist(),
        "eval_indices": s.eval_indices.tolist(),
        "train_groups": list(s.train_groups),
        "eval_groups": list(s.eval_groups),
        "notes": s.notes,
    }


def _dict_to_split(d: dict) -> SplitMetadata:
    return SplitMetadata(
        strategy=d["strategy"],
        healthy_labels=tuple(d["healthy_labels"]),
        segments_per_group=d["segments_per_group"],
        seed=d["seed"],
        train_indices=np.array(d["train_indices"], dtype=int),
        eval_indices=np.array(d["eval_indices"], dtype=int),
        train_groups=tuple(d["train_groups"]),
        eval_groups=tuple(d["eval_groups"]),
        notes=d.get("notes", ""),
    )


def cache_split(
    split: DatasetSplit,
    pipeline: Pipeline,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    provenance_ref: str = PROVENANCE_REF,
) -> Path:
    """Write the preprocessed split + manifest to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_signals.npy", np.asarray(split.train.signals))
    np.save(out_dir / "eval_signals.npy", np.asarray(split.eval.signals))
    # Eval labels: persisted but only ever loaded back into a sealed LabelVault.
    np.save(out_dir / "eval_labels.npy", split.eval_labels.reveal(consumer="cache_writer"))

    (out_dir / "split_meta.json").write_text(
        json.dumps(_split_to_dict(split.split), indent=2), encoding="utf-8"
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "Z24",
        "fs": split.train.fs,
        "source_provenance_ref": provenance_ref,
        "shapes": {
            "train_signals": list(np.asarray(split.train.signals).shape),
            "eval_signals": list(np.asarray(split.eval.signals).shape),
        },
        "n_train": split.split.n_train,
        "n_eval": split.split.n_eval,
        "pipeline": pipeline.manifest(),
        "unsupervised_contract": "train is healthy-only & label-free; eval labels sealed",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def load_cached(out_dir: str | Path = DEFAULT_OUT_DIR) -> DatasetSplit:
    """Reconstruct a `DatasetSplit` from the cache (eval labels re-sealed)."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    fs = manifest.get("fs", 100.0)

    train_sig = np.load(out_dir / "train_signals.npy")
    eval_sig = np.load(out_dir / "eval_signals.npy")
    eval_lab = np.load(out_dir / "eval_labels.npy")
    split = _dict_to_split(json.loads((out_dir / "split_meta.json").read_text(encoding="utf-8")))

    train = Recording(train_sig, fs=fs, sensor_ids=tuple(range(train_sig.shape[1])),
                      name="z24-train-healthy+pp", meta={"preprocessed": True, "role": "train"})
    evalr = Recording(eval_sig, fs=fs, sensor_ids=tuple(range(eval_sig.shape[1])),
                     name="z24-eval+pp", meta={"preprocessed": True, "role": "eval"})
    return DatasetSplit(
        train=train, eval=evalr,
        eval_labels=LabelVault(eval_lab, purpose="evaluation"),
        split=split,
    )
