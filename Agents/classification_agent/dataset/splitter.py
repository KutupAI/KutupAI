"""
splitter.py
-------------
Stratified train/val/test split (task doc section 6: "Stratified
train/validation/test split kullan").

Two rules from section 6 enforced here, not left to the caller to remember:
  1. Synthetic documents (LabeledDocument.is_synthetic=True) are excluded
     from the test split -- "final test seti mumkun oldugunca gercek ve
     modelin gormedigi orneklerden olusmali". They can still go to train.
  2. The split is stratified per class so every split keeps roughly the
     same class proportions as the full labeled set (protects minority
     classes from landing entirely in one split by chance).
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from Agents.classification_agent.dataset.loader import load_manifest, write_manifest_template
from Agents.classification_agent.dataset.schema import LabeledDocument


def stratified_split(
    records: list[LabeledDocument],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[LabeledDocument]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    labeled = [r for r in records if r.label]
    unlabeled_count = len(records) - len(labeled)
    if unlabeled_count:
        print(f"[splitter] skipping {unlabeled_count} unlabeled document(s) -- label them first.")

    by_class: dict[str, list[LabeledDocument]] = defaultdict(list)
    for r in labeled:
        by_class[r.label].append(r)

    rng = random.Random(seed)
    train: list[LabeledDocument] = []
    val: list[LabeledDocument] = []
    test: list[LabeledDocument] = []

    for label, group in by_class.items():
        real = [r for r in group if not r.is_synthetic]
        synthetic = [r for r in group if r.is_synthetic]
        rng.shuffle(real)
        rng.shuffle(synthetic)

        n_real = len(real)
        n_test = round(n_real * test_ratio)
        n_val = round(n_real * val_ratio)

        class_test = real[:n_test]
        class_val = real[n_test : n_test + n_val]
        class_train = real[n_test + n_val :] + synthetic  # synthetic only ever goes to train

        for r in class_test:
            r.split = "test"
        for r in class_val:
            r.split = "val"
        for r in class_train:
            r.split = "train"

        test.extend(class_test)
        val.extend(class_val)
        train.extend(class_train)

        if n_real > 0 and n_test == 0:
            print(f"[splitter] WARNING: class '{label}' has only {n_real} real example(s) -- no test examples assigned.")

    return {"train": train, "val": val, "test": test}


def write_split_manifests(splits: dict[str, list[LabeledDocument]], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, records in splits.items():
        path = output_dir / f"{name}.csv"
        write_manifest_template(records, path)
        written[name] = path
    return written


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stratified train/val/test split of a labeled manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="Agents/classification_agent/dataset/splits")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_manifest(args.manifest, require_labels=False)
    splits = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    written = write_split_manifests(splits, args.output_dir)

    for name, path in written.items():
        print(f"{name}: {len(splits[name])} document(s) -> {path}")


if __name__ == "__main__":
    _cli()
