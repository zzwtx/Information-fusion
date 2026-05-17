#!/usr/bin/env python3
"""Export Stanford Cars images from Hugging Face parquet shards.

The mteb/StanfordCars parquet files keep the original Stanford Cars ordering in
the ``id`` column. The FedMGP split file refers to images as
``cars_train/00001.jpg`` and ``cars_test/00001.jpg``, so we reconstruct those
filenames using ``id + 1``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def export_split(parquet_dir: Path, out_dir: Path, split: str) -> int:
    files = sorted(parquet_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet files found in {parquet_dir}")

    image_dir = out_dir / f"cars_{split}"
    image_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for parquet_file in files:
        table = pq.read_table(parquet_file, columns=["image", "id"])
        for row in table.to_pylist():
            image = row["image"]
            image_bytes = image["bytes"] if isinstance(image, dict) else None
            if not image_bytes:
                raise ValueError(f"Missing image bytes in {parquet_file}")
            image_id = int(row["id"]) + 1
            out_file = image_dir / f"{image_id:05d}.jpg"
            if not out_file.exists() or out_file.stat().st_size == 0:
                out_file.write_bytes(image_bytes)
            written += 1

    return written


def verify_split_paths(data_dir: Path) -> None:
    split_path = data_dir / "split_zhou_StanfordCars.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing {split_path}")

    split = json.loads(split_path.read_text(encoding="utf-8"))
    missing = []
    for section in ("train", "val", "test"):
        for rel_path, _label, _classname in split[section][:20]:
            if not (data_dir / rel_path).is_file():
                missing.append(rel_path)
    if missing:
        raise FileNotFoundError("Missing split-referenced files: " + ", ".join(missing[:10]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/stanford_cars"))
    parser.add_argument("--parquet-dir", type=Path, default=Path("DATA/stanford_cars/hf_parquet"))
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    parquet_dir = args.parquet_dir.resolve()

    train_count = export_split(parquet_dir, data_dir, "train")
    test_count = export_split(parquet_dir, data_dir, "test")
    verify_split_paths(data_dir)

    print(f"Exported {train_count} train images and {test_count} test images to {data_dir}")


if __name__ == "__main__":
    main()
