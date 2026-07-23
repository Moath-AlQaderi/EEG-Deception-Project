"""Shared constants, identifiers, logging, hashing, and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

RANDOM_SEED = 42
CLASS_NAMES = ["Honest", "Bluffer", "Liar", "Deceiver"]
ROLE_NAMES = {index + 1: name for index, name in enumerate(CLASS_NAMES)}
EXPECTED_CHANNELS = [
    "C3",
    "C4",
    "Fp1",
    "Fp2",
    "F4",
    "Fz",
    "F3",
    "T7",
    "Cz",
    "T8",
    "P4",
    "Pz",
    "P3",
    "O1",
    "Oz",
    "O2",
]
FEATURE_PREFIX = "f__"
MODEL_SLUGS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "svm",
    "knn",
    "naive_bayes",
    "mlp",
]
PROBABILITY_COLUMNS = [
    "probability_honest",
    "probability_bluffer",
    "probability_liar",
    "probability_deceiver",
]


def configure_logging(outputs_dir: Path | None = None, verbose: bool = False) -> None:
    """Configure concise console logging and, optionally, a run log file."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if outputs_dir is not None:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(outputs_dir / "run.log", mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # Route library warnings (for example, a convergence warning) into both the
    # console and the persisted run log instead of losing them from the audit.
    logging.captureWarnings(True)


def numeric_id(value: Any, prefix: str) -> int:
    """Parse identifiers such as ``Session10`` and over-padded ``Q001``."""

    match = re.fullmatch(rf"{re.escape(prefix)}0*(\d+)", str(value), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid {prefix} identifier: {value!r}")
    return int(match.group(1))


def canonical_session(value: Any) -> str:
    return f"Session{numeric_id(value, 'Session')}"


def canonical_prefixed(value: Any, prefix: str) -> str:
    return f"{prefix}{numeric_id(value, prefix):02d}"


def session_sort_key(value: str) -> int:
    return numeric_id(value, "Session")


def make_trial_id(session_id: str, participant_id: str, role_id: int, question_id: str) -> str:
    return f"{session_id}_{participant_id}_R{int(role_id):02d}_{question_id}"


def portable_path(path: Path, project_root: Path) -> str:
    """Prefer a project-relative path but support external dataset roots."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_stored_path(value: str, project_root: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, set):
        return sorted(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_safe(value: Any) -> Any:
    """Recursively convert values to strict-JSON data (never NaN/Infinity)."""

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            ensure_ascii=False,
            default=json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def inventory_hash(paths: Iterable[Path], root: Path) -> str:
    """Hash sorted relative paths and sizes without rereading multi-GB signals."""

    records = []
    for path in sorted(paths, key=lambda item: str(item).lower()):
        try:
            name = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            name = str(path.resolve())
        records.append((name, path.stat().st_size))
    return stable_json_hash(records)


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def pairwise_disjoint(groups: Iterable[set[Any]]) -> bool:
    values = list(groups)
    return all(
        values[i].isdisjoint(values[j])
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )


def runtime_versions() -> dict[str, Any]:
    import joblib
    import matplotlib
    import mne
    import pandas
    import scipy
    import sklearn
    import xgboost

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "mne": mne.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
    }
