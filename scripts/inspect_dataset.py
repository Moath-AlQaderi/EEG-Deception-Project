"""Recursively inspect the EEG dataset and freeze the requested group split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    assign_group_split,
    inspect_dataset,
    locate_dataset_root,
    locate_paper,
    save_audit_artifacts,
    save_split_assignments,
    summarize_metadata,
)
from src.utils import configure_logging, portable_path, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def inspect_and_split(
    dataset_path: Path | None,
    outputs_dir: Path,
    seed: int = 42,
) -> tuple[object, dict, Path, Path]:
    """Run the complete raw audit and save the pre-QC split artifacts."""

    if seed != 42:
        raise ValueError("The published GroupShuffleSplit random_state is fixed at 42")
    dataset_root = locate_dataset_root(PROJECT_ROOT, dataset_path)
    paper = locate_paper(PROJECT_ROOT)
    metadata = inspect_dataset(dataset_root, workspace_root=PROJECT_ROOT)
    metadata, split_summary = assign_group_split(metadata)
    save_audit_artifacts(metadata, outputs_dir, split_summary)
    save_split_assignments(metadata, outputs_dir / "data_split.csv")
    summary = summarize_metadata(metadata)
    summary.update(
        {
            "paper_path": portable_path(paper, PROJECT_ROOT),
            "dataset_root": portable_path(dataset_root, PROJECT_ROOT),
            "group_split": split_summary,
        }
    )
    write_json(outputs_dir / "dataset_summary.json", summary)
    return metadata, split_summary, dataset_root, paper


def main() -> None:
    args = parse_args()
    outputs = PROJECT_ROOT / "outputs"
    configure_logging(outputs, args.verbose)
    metadata, split, dataset_root, paper = inspect_and_split(
        args.dataset_path, outputs, args.seed
    )
    print(
        "DATASET SUMMARY\n"
        f"Paper: {paper}\n"
        f"Dataset: {dataset_root}\n"
        f"Trials: {len(metadata)}; sessions: {metadata['session_id'].nunique()}; "
        f"participants: {metadata['participant_id'].nunique()}\n"
        f"Structurally valid: {int(metadata['structurally_valid'].sum())}/{len(metadata)}\n"
        f"Train: {split['train_trial_count']} trials, {split['train_participant_count']} participants, "
        f"sessions={', '.join(split['train_sessions'])}\n"
        f"Test: {split['test_trial_count']} trials, {split['test_participant_count']} participants, "
        f"sessions={', '.join(split['test_sessions'])}\n"
        f"Metadata: {outputs / 'dataset_metadata.csv'}\n"
        f"Split: {outputs / 'data_split.csv'}"
    )


if __name__ == "__main__":
    main()
