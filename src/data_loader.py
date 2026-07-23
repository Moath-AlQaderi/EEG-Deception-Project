"""Dataset discovery, raw MATLAB audit, and the frozen session-wise split.

This module deliberately assigns every *raw trial* before artifact rejection,
feature extraction, or window generation.  The requested ``GroupShuffleSplit``
is reproduced with the stored string-valued ``session_id`` values; changing
those values to integers changes scikit-learn's group ordering and therefore
changes the held-out sessions.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import GroupShuffleSplit

try:  # Keep this module usable while the rest of the project is scaffolded.
    from .utils import CLASS_NAMES, EXPECTED_CHANNELS, write_json as _shared_write_json
except (ImportError, AttributeError):  # pragma: no cover - exercised only during scaffolding.
    CLASS_NAMES = ("Honest", "Bluffer", "Liar", "Deceiver")
    EXPECTED_CHANNELS = (
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
    )
    _shared_write_json = None


LOGGER = logging.getLogger(__name__)

ROLE_NAMES = {index + 1: str(name) for index, name in enumerate(CLASS_NAMES)}
EXPECTED_CHANNELS = tuple(str(channel) for channel in EXPECTED_CHANNELS)

FILENAME_PATTERN = re.compile(
    r"Session(?P<session>\d+)_P(?P<participant>\d+)_R(?P<role>\d+)_Q(?P<question>\d+)\.mat$",
    flags=re.IGNORECASE,
)
EVENT_PHASES = ("rest", "question", "answers", "thinking", "response")
EVENT_FIELDS = tuple(
    f"{phase}_{edge}" for phase in EVENT_PHASES for edge in ("start", "end")
)

EXPECTED_TEST_SESSIONS = frozenset(
    {"Session1", "Session8", "Session10", "Session14"}
)
EXPECTED_TRAIN_SESSIONS = frozenset(
    {
        "Session2",
        "Session3",
        "Session4",
        "Session5",
        "Session6",
        "Session7",
        "Session9",
        "Session11",
        "Session12",
        "Session13",
        "Session15",
        "Session16",
        "Session17",
    }
)
SPLIT_TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42

# These are protocol expectations used only to flag durations for inspection.
# A duration warning does not by itself make a file structurally unusable.
PHASE_DURATION_EXPECTATIONS: Mapping[str, tuple[float, float]] = {
    "rest": (0.50, 0.05),
    "question": (5.00, 0.25),
    "answers": (30.00, 2.00),
    "thinking": (5.00, 0.50),
    "response": (2.00, 0.10),
}


def _numeric_id(value: Any, prefix: str) -> int:
    """Parse identifiers such as ``Session08``, ``P01``, or ``Q001``."""

    match = re.fullmatch(
        rf"{re.escape(prefix)}0*(\d+)", str(value), flags=re.IGNORECASE
    )
    if match is None:
        raise ValueError(f"Invalid {prefix} identifier: {value!r}")
    return int(match.group(1))


def canonical_session(value: Any) -> str:
    """Return an unpadded canonical session identifier."""

    return f"Session{_numeric_id(value, 'Session')}"


def canonical_participant(value: Any) -> str:
    """Return a two-digit canonical participant identifier."""

    return f"P{_numeric_id(value, 'P'):02d}"


def canonical_question(value: Any) -> str:
    """Canonicalize both ordinary ``Q01`` and anomalous ``Q001`` values."""

    return f"Q{_numeric_id(value, 'Q'):02d}"


def canonical_role(value: Any) -> int:
    """Return the integer role label from an ``R01``-style value."""

    return _numeric_id(value, "R")


def session_sort_key(value: str) -> int:
    """Sort ``Session10`` after ``Session9`` rather than after ``Session1``."""

    return _numeric_id(value, "Session")


def _portable_path(path: Path, workspace_root: Path | None) -> str:
    resolved = path.resolve()
    if workspace_root is not None:
        try:
            return resolved.relative_to(workspace_root.resolve()).as_posix()
        except ValueError:
            pass
    return str(resolved)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _shared_write_json is not None:
        _shared_write_json(path, payload)
        return
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def locate_dataset_root(
    search_root: str | Path, explicit_path: str | Path | None = None
) -> Path:
    """Locate the directory beneath which the 17 session trees reside."""

    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_dir() or not any(candidate.rglob("*.mat")):
            raise FileNotFoundError(f"No MATLAB trial files found below {candidate}")
        return candidate

    root = Path(search_root).expanduser().resolve()
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.casefold() == "mock_crime_scenario_dataset"
    ]
    populated = [path for path in candidates if any(path.rglob("*.mat"))]
    if not populated:
        raise FileNotFoundError(
            f"Could not locate Mock_Crime_Scenario_Dataset below {root}"
        )
    populated.sort(key=lambda path: (-sum(1 for _ in path.rglob("*.mat")), str(path)))
    if len(populated) > 1:
        LOGGER.warning("Multiple dataset roots found; using %s", populated[0])
    return populated[0].resolve()


def locate_paper(search_root: str | Path) -> Path:
    """Locate the research paper while excluding the question booklet."""

    root = Path(search_root).expanduser().resolve()
    candidates = [
        path
        for path in root.rglob("*.pdf")
        if path.name.casefold() != "questions.pdf"
        and ("multi-class" in path.name.casefold() or "deception" in path.name.casefold())
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not locate the deception dataset paper below {root}")
    candidates.sort(key=lambda path: (len(path.parts), path.name.casefold()))
    return candidates[0].resolve()


def parse_trial_filename(path: str | Path) -> dict[str, Any]:
    """Parse and numerically canonicalize identifiers in a trial filename."""

    candidate = Path(path)
    match = FILENAME_PATTERN.fullmatch(candidate.name)
    if match is None:
        raise ValueError(f"Unrecognized trial filename: {candidate.name}")
    values = {name: int(value) for name, value in match.groupdict().items()}
    role_id = values["role"]
    if role_id not in ROLE_NAMES:
        raise ValueError(f"Role outside 1..4 in {candidate.name}: {role_id}")
    if values["question"] not in range(1, 15):
        raise ValueError(
            f"Question outside 1..14 in {candidate.name}: {values['question']}"
        )
    return {
        "session_id": f"Session{values['session']}",
        "participant_id": f"P{values['participant']:02d}",
        "role_id": role_id,
        "question_id": f"Q{values['question']:02d}",
        "filename_session_raw": f"Session{match.group('session')}",
        "filename_participant_raw": f"P{match.group('participant')}",
        "filename_role_raw": f"R{match.group('role')}",
        "filename_question_raw": f"Q{match.group('question')}",
    }


def make_trial_id(
    session_id: str, participant_id: str, role_id: int, question_id: str
) -> str:
    """Build a canonical globally unique trial identifier."""

    return f"{session_id}_{participant_id}_R{int(role_id):02d}_{question_id}"


def discover_mat_files(dataset_root: str | Path) -> list[Path]:
    """Recursively discover trials in deterministic numeric identifier order."""

    root = Path(dataset_root).expanduser().resolve()
    files = list(root.rglob("*.mat"))
    if not files:
        raise FileNotFoundError(f"No MATLAB files found below {root}")

    parsed: list[tuple[tuple[int, int, int, int, str], Path]] = []
    failures: list[str] = []
    for path in files:
        try:
            identifiers = parse_trial_filename(path)
            key = (
                session_sort_key(identifiers["session_id"]),
                _numeric_id(identifiers["participant_id"], "P"),
                int(identifiers["role_id"]),
                _numeric_id(identifiers["question_id"], "Q"),
                path.as_posix(),
            )
            parsed.append((key, path.resolve()))
        except ValueError as exc:
            failures.append(str(exc))
    if failures:
        raise ValueError("Unrecognized MATLAB filenames: " + "; ".join(failures[:10]))
    return [path for _, path in sorted(parsed, key=lambda item: item[0])]


def _event_values(record: Mapping[str, Any]) -> dict[str, int]:
    events = record.get("events")
    if not isinstance(events, Mapping):
        raise TypeError("EEGStruct.events is not a mapping")
    missing = [field for field in EVENT_FIELDS if field not in events]
    if missing:
        raise KeyError(f"Missing event fields: {missing}")
    return {field: int(events[field]) for field in EVENT_FIELDS}


def _event_issues(events: Mapping[str, int], n_samples: int) -> list[str]:
    """Validate MATLAB one-based inclusive event markers."""

    issues: list[str] = []
    for phase in EVENT_PHASES:
        start = int(events[f"{phase}_start"])
        end = int(events[f"{phase}_end"])
        if start < 1 or end < start or end > n_samples:
            issues.append(f"invalid_{phase}_bounds")
    ordered = [int(events[field]) for field in EVENT_FIELDS]
    if ordered != sorted(ordered):
        issues.append("nonmonotonic_events")
    for first, second in zip(EVENT_PHASES[:-1], EVENT_PHASES[1:]):
        if int(events[f"{second}_start"]) != int(events[f"{first}_end"]) + 1:
            issues.append(f"noncontiguous_{first}_{second}")
    if int(events["rest_start"]) != 1:
        issues.append("rest_does_not_start_at_one")
    if int(events["response_end"]) != n_samples:
        issues.append("response_end_not_last_sample")
    return issues


def _phase_durations(
    events: Mapping[str, int], sampling_rate_hz: float
) -> tuple[dict[str, float], list[str]]:
    durations: dict[str, float] = {}
    issues: list[str] = []
    for phase in EVENT_PHASES:
        samples = int(events[f"{phase}_end"]) - int(events[f"{phase}_start"]) + 1
        duration = samples / sampling_rate_hz
        durations[f"{phase}_duration_seconds"] = float(duration)
        expected, tolerance = PHASE_DURATION_EXPECTATIONS[phase]
        if abs(duration - expected) > tolerance:
            # The exact value already lives in ``{phase}_duration_seconds``;
            # retain a stable issue code so corpus-level counts are meaningful.
            issues.append(f"unexpected_{phase}_duration")
    return durations, issues


def _folder_session(path: Path) -> str | None:
    for parent in path.parents:
        if re.fullmatch(r"session\d+", parent.name, flags=re.IGNORECASE):
            return canonical_session(parent.name)
    return None


def inspect_mat_file(
    path: str | Path, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Inspect one MAT trial and return a complete auditable metadata row."""

    candidate = Path(path).expanduser().resolve()
    workspace = Path(workspace_root).resolve() if workspace_root is not None else None
    filename_ids = parse_trial_filename(candidate)
    row: dict[str, Any] = {
        "filepath": _portable_path(candidate, workspace),
        "filename": candidate.name,
        "file_size_bytes": int(candidate.stat().st_size),
        **filename_ids,
        "original_session_id": "",
        "original_participant_id": "",
        "original_role_id": "",
        "original_question_id": "",
        "n_channels": np.nan,
        "n_samples": np.nan,
        "sampling_rate_hz": np.nan,
        "duration_seconds": np.nan,
        # Explicit aliases retain the wording used in the experiment request.
        "number_of_channels": np.nan,
        "number_of_samples": np.nan,
        "sampling_rate": np.nan,
        "duration": np.nan,
        "data_finite": False,
        "nan_count": np.nan,
        "infinite_count": np.nan,
        "flat_channel_count": np.nan,
        "peak_absolute_value": np.nan,
        "channel_order": "",
        "channel_order_matches": False,
        "identifier_matches_filename": False,
        "folder_session_matches": False,
        "event_index_convention": "MATLAB one-based inclusive",
        "events_valid": False,
        "structural_issues": "",
        "duration_issues": "",
        "inspection_issues": "",
        "structurally_valid": False,
        "qc_status": "not_run",
        "rejection_reason": "",
    }
    row.update({field: np.nan for field in EVENT_FIELDS})
    row.update({f"{phase}_duration_seconds": np.nan for phase in EVENT_PHASES})

    structural_issues: list[str] = []
    duration_issues: list[str] = []
    try:
        payload = loadmat(candidate, simplify_cells=True, variable_names=["EEGStruct"])
        if "EEGStruct" not in payload:
            raise KeyError("MAT file has no EEGStruct variable")
        record = payload["EEGStruct"]
        if not isinstance(record, Mapping):
            raise TypeError("EEGStruct is not a mapping")
        required = {
            "data",
            "events",
            "channels",
            "sampling_rate",
            "session_id",
            "participant_id",
            "role_id",
            "question_id",
        }
        missing = sorted(required.difference(record))
        if missing:
            raise KeyError(f"Missing EEGStruct fields: {missing}")

        data = np.asarray(record["data"], dtype=np.float64)
        channels = tuple(
            str(channel) for channel in np.asarray(record["channels"]).reshape(-1)
        )
        sampling_rate = float(record["sampling_rate"])
        if data.ndim != 2:
            structural_issues.append(f"data_not_2d:{data.shape}")
            n_channels, n_samples = 0, 0
        else:
            n_channels, n_samples = map(int, data.shape)

        stored_session = canonical_session(record["session_id"])
        stored_participant = canonical_participant(record["participant_id"])
        stored_role = canonical_role(record["role_id"])
        stored_question = canonical_question(record["question_id"])
        stored_ids = (
            stored_session,
            stored_participant,
            stored_role,
            stored_question,
        )
        parsed_ids = (
            filename_ids["session_id"],
            filename_ids["participant_id"],
            filename_ids["role_id"],
            filename_ids["question_id"],
        )
        identifiers_match = stored_ids == parsed_ids
        if not identifiers_match:
            structural_issues.append("stored_filename_identifier_mismatch")

        if stored_role not in ROLE_NAMES:
            structural_issues.append(f"role_outside_1_to_4:{stored_role}")
        if _numeric_id(stored_question, "Q") not in range(1, 15):
            structural_issues.append(f"question_outside_1_to_14:{stored_question}")
        if n_channels != len(EXPECTED_CHANNELS):
            structural_issues.append(f"incorrect_channel_count:{n_channels}")
        if channels != EXPECTED_CHANNELS:
            structural_issues.append("unexpected_channel_order")
        if sampling_rate != 2048.0:
            structural_issues.append(f"unexpected_sampling_rate:{sampling_rate}")

        finite = bool(data.ndim == 2 and np.isfinite(data).all())
        nan_count = int(np.isnan(data).sum()) if data.ndim == 2 else -1
        infinite_count = int(np.isinf(data).sum()) if data.ndim == 2 else -1
        if not finite:
            structural_issues.append("nonfinite_signal")
        if data.ndim == 2 and data.shape[1] > 0:
            flat_count = int(np.sum(np.ptp(data, axis=1) == 0.0))
            peak = float(np.nanmax(np.abs(data)))
        else:
            flat_count = -1
            peak = np.nan
        if flat_count > 0:
            structural_issues.append(f"exact_flat_channels:{flat_count}")

        events = _event_values(record)
        event_problems = _event_issues(events, n_samples)
        structural_issues.extend(event_problems)
        if not event_problems and sampling_rate > 0:
            durations, duration_issues = _phase_durations(events, sampling_rate)
            row.update(durations)

        folder_session = _folder_session(candidate)
        folder_matches = folder_session == stored_session
        if not folder_matches:
            structural_issues.append(
                f"folder_session_mismatch:{folder_session!r}!={stored_session!r}"
            )

        row.update(
            {
                "session_id": stored_session,
                "participant_id": stored_participant,
                "role_id": stored_role,
                "role_name": ROLE_NAMES.get(stored_role, "Unknown"),
                "question_id": stored_question,
                "original_session_id": str(record["session_id"]),
                "original_participant_id": str(record["participant_id"]),
                "original_role_id": str(record["role_id"]),
                "original_question_id": str(record["question_id"]),
                "n_channels": n_channels,
                "n_samples": n_samples,
                "sampling_rate_hz": sampling_rate,
                "duration_seconds": n_samples / sampling_rate if sampling_rate > 0 else np.nan,
                "number_of_channels": n_channels,
                "number_of_samples": n_samples,
                "sampling_rate": sampling_rate,
                "duration": n_samples / sampling_rate if sampling_rate > 0 else np.nan,
                "data_finite": finite,
                "nan_count": nan_count,
                "infinite_count": infinite_count,
                "flat_channel_count": flat_count,
                "peak_absolute_value": peak,
                "channel_order": ";".join(channels),
                "channel_order_matches": channels == EXPECTED_CHANNELS,
                "identifier_matches_filename": identifiers_match,
                "folder_session_matches": folder_matches,
                "events_valid": not event_problems,
            }
        )
        row.update(events)
    except Exception as exc:
        structural_issues.append(f"malformed:{type(exc).__name__}:{exc}")

    row["trial_id"] = make_trial_id(
        str(row["session_id"]),
        str(row["participant_id"]),
        int(row["role_id"]),
        str(row["question_id"]),
    )
    structural_issues = list(dict.fromkeys(structural_issues))
    duration_issues = list(dict.fromkeys(duration_issues))
    row["structural_issues"] = "|".join(structural_issues)
    row["duration_issues"] = "|".join(duration_issues)
    row["inspection_issues"] = "|".join(structural_issues + duration_issues)
    row["structurally_valid"] = not structural_issues
    return row


def inspect_dataset(
    dataset_root: str | Path, workspace_root: str | Path | None = None
) -> pd.DataFrame:
    """Inspect every raw MAT file and return one deterministically ordered row each."""

    files = discover_mat_files(dataset_root)
    LOGGER.info("Inspecting %d MATLAB trial files", len(files))
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(files, start=1):
        rows.append(inspect_mat_file(path, workspace_root=workspace_root))
        if index % 100 == 0 or index == len(files):
            LOGGER.info("Inspected %d/%d MATLAB files", index, len(files))
    metadata = pd.DataFrame(rows)
    metadata.insert(0, "raw_row_index", np.arange(len(metadata), dtype=int))
    return metadata


# Clear aliases for callers that prefer task-oriented names.
build_metadata_table = inspect_dataset
create_metadata_table = inspect_dataset


def assert_raw_dataset_design(metadata: pd.DataFrame) -> None:
    """Fail unless metadata represent the documented complete raw experiment."""

    required = {
        "trial_id",
        "session_id",
        "participant_id",
        "role_id",
        "question_id",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise AssertionError(f"Metadata missing required columns: {sorted(missing)}")
    failures: list[str] = []
    expected_sessions = {f"Session{number}" for number in range(1, 18)}
    if len(metadata) != 952:
        failures.append(f"raw_trial_count={len(metadata)} expected=952")
    if not metadata["trial_id"].is_unique:
        failures.append("canonical trial IDs are not unique")
    if set(metadata["session_id"]) != expected_sessions:
        failures.append("observed sessions differ from Session1..Session17")
    if metadata["participant_id"].nunique() != 51:
        failures.append(
            f"participant_count={metadata['participant_id'].nunique()} expected=51"
        )
    if set(metadata["role_id"].astype(int)) != {1, 2, 3, 4}:
        failures.append(f"role_labels={sorted(set(metadata['role_id']))}")
    session_counts = metadata.groupby("session_id").size()
    if len(session_counts) != 17 or not session_counts.eq(56).all():
        failures.append(f"session_counts={session_counts.to_dict()}")
    pair_groups = metadata.groupby(["session_id", "role_id"], sort=False)
    pair_counts = pair_groups.size()
    if len(pair_counts) != 68 or not pair_counts.eq(14).all():
        failures.append(f"session_role_counts={pair_counts.to_dict()}")
    expected_questions = set(range(1, 15))
    for (session_id, role_id), group in pair_groups:
        observed = {_numeric_id(value, "Q") for value in group["question_id"]}
        if observed != expected_questions:
            failures.append(
                f"questions_{session_id}_R{int(role_id):02d}={sorted(observed)}"
            )
    participant_sessions = metadata.groupby("participant_id")["session_id"].nunique()
    if not participant_sessions.eq(1).all():
        failures.append("at least one participant occurs in multiple sessions")
    thief = metadata[metadata["role_id"].isin([3, 4])]
    thief_roles = thief.groupby("participant_id")["role_id"].apply(
        lambda values: set(map(int, values))
    )
    if len(thief_roles) != 17 or not thief_roles.apply(lambda roles: roles == {3, 4}).all():
        failures.append("liar/deceiver participant blocks are incomplete")
    if failures:
        raise AssertionError("Raw dataset design check failed: " + "; ".join(failures))


def assign_group_split(
    metadata: pd.DataFrame,
    *,
    test_size: float = SPLIT_TEST_SIZE,
    random_state: int = SPLIT_RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign the frozen sklearn session split to every raw trial.

    ``GroupShuffleSplit`` applies a fractional ``test_size`` to groups, not rows.
    With 17 equal-sized sessions, 0.20 rounds up to four test sessions, producing
    224 test trials and 728 training trials.
    """

    if test_size != SPLIT_TEST_SIZE or random_state != SPLIT_RANDOM_STATE:
        raise ValueError(
            "The published split is frozen at test_size=0.20 and random_state=42"
        )
    assert_raw_dataset_design(metadata)
    if metadata["session_id"].map(type).ne(str).any():
        raise AssertionError("session_id must remain string-valued for the frozen split")

    groups = metadata["session_id"].to_numpy(dtype=str)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )
    train_indices, test_indices = next(
        splitter.split(
            np.zeros((len(metadata), 1), dtype=np.uint8),
            metadata["role_id"].to_numpy(dtype=int),
            groups=groups,
        )
    )
    train_sessions = set(metadata.iloc[train_indices]["session_id"])
    test_sessions = set(metadata.iloc[test_indices]["session_id"])
    if train_sessions != EXPECTED_TRAIN_SESSIONS or test_sessions != EXPECTED_TEST_SESSIONS:
        raise AssertionError(
            "GroupShuffleSplit assignment changed: "
            f"train={sorted(train_sessions, key=session_sort_key)}, "
            f"test={sorted(test_sessions, key=session_sort_key)}"
        )

    assigned = metadata.copy()
    assigned["split"] = ""
    assigned.iloc[train_indices, assigned.columns.get_loc("split")] = "train"
    assigned.iloc[test_indices, assigned.columns.get_loc("split")] = "test"
    report = assert_leakage_free_split(assigned)
    summary = {
        "method": "sklearn.model_selection.GroupShuffleSplit",
        "group_column": "session_id",
        "group_dtype": "string",
        "n_splits": 1,
        "test_size": SPLIT_TEST_SIZE,
        "random_state": SPLIT_RANDOM_STATE,
        "note": "0.20 of 17 groups rounds up to 4 sessions (23.529% of this balanced corpus)",
        "train_sessions": sorted(EXPECTED_TRAIN_SESSIONS, key=session_sort_key),
        "test_sessions": sorted(EXPECTED_TEST_SESSIONS, key=session_sort_key),
        "train_trial_count": int(len(train_indices)),
        "test_trial_count": int(len(test_indices)),
        "train_participant_count": int(
            assigned.loc[assigned["split"] == "train", "participant_id"].nunique()
        ),
        "test_participant_count": int(
            assigned.loc[assigned["split"] == "test", "participant_id"].nunique()
        ),
        "leakage_checks": report,
    }
    return assigned, summary


create_group_split = assign_group_split


def assert_leakage_free_split(metadata: pd.DataFrame) -> dict[str, Any]:
    """Assert the exact split plus session, participant, trial, and thief isolation."""

    required = {"trial_id", "session_id", "participant_id", "role_id", "split"}
    missing = required.difference(metadata.columns)
    if missing:
        raise AssertionError(f"Split metadata missing columns: {sorted(missing)}")

    checks: dict[str, bool] = {}
    checks["split_values_are_train_test"] = set(metadata["split"]) == {"train", "test"}
    observed_train_sessions = set(
        metadata.loc[metadata["split"] == "train", "session_id"]
    )
    observed_test_sessions = set(
        metadata.loc[metadata["split"] == "test", "session_id"]
    )
    checks["train_sessions_match_frozen_assignment"] = (
        observed_train_sessions == EXPECTED_TRAIN_SESSIONS
    )
    checks["test_sessions_match_frozen_assignment"] = (
        observed_test_sessions == EXPECTED_TEST_SESSIONS
    )
    checks["session_sets_disjoint"] = observed_train_sessions.isdisjoint(
        observed_test_sessions
    )
    checks["one_split_per_session"] = bool(
        metadata.groupby("session_id")["split"].nunique().eq(1).all()
    )
    checks["one_split_per_participant"] = bool(
        metadata.groupby("participant_id")["split"].nunique().eq(1).all()
    )
    checks["participant_belongs_to_one_session"] = bool(
        metadata.groupby("participant_id")["session_id"].nunique().eq(1).all()
    )
    checks["trial_ids_unique"] = bool(metadata["trial_id"].is_unique)
    checks["one_split_per_trial"] = bool(
        metadata.groupby("trial_id")["split"].nunique().eq(1).all()
    )

    train = metadata[metadata["split"] == "train"]
    test = metadata[metadata["split"] == "test"]
    checks["participant_sets_disjoint"] = set(train["participant_id"]).isdisjoint(
        test["participant_id"]
    )
    checks["trial_sets_disjoint"] = set(train["trial_id"]).isdisjoint(
        test["trial_id"]
    )
    checks["all_raw_trials_assigned_once"] = (
        len(metadata) == 952 and len(train) == 728 and len(test) == 224
    )
    checks["participant_counts_are_39_12"] = (
        train["participant_id"].nunique() == 39
        and test["participant_id"].nunique() == 12
    )
    expected_role_counts = {
        "train": {1: 182, 2: 182, 3: 182, 4: 182},
        "test": {1: 56, 2: 56, 3: 56, 4: 56},
    }
    checks["raw_role_counts_are_balanced"] = all(
        subset["role_id"].astype(int).value_counts().sort_index().to_dict()
        == expected_role_counts[name]
        for name, subset in (("train", train), ("test", test))
    )

    thief = metadata[metadata["role_id"].isin([3, 4])]
    thief_roles = thief.groupby("participant_id")["role_id"].apply(
        lambda values: set(map(int, values))
    )
    checks["liar_deceiver_blocks_complete"] = bool(
        len(thief_roles) == 17
        and thief_roles.apply(lambda roles: roles == {3, 4}).all()
    )
    checks["liar_deceiver_blocks_share_split"] = bool(
        thief.groupby("participant_id")["split"].nunique().eq(1).all()
    )

    failed = [name for name, passed in checks.items() if not passed]
    report: dict[str, Any] = {
        "passed": not failed,
        "checks": checks,
        "train_sessions": sorted(observed_train_sessions, key=session_sort_key),
        "test_sessions": sorted(observed_test_sessions, key=session_sort_key),
        "train_participants": int(train["participant_id"].nunique()),
        "test_participants": int(test["participant_id"].nunique()),
        "train_trials": int(len(train)),
        "test_trials": int(len(test)),
    }
    if failed:
        raise AssertionError(f"Leakage/split checks failed: {failed}")
    return report


run_leakage_checks = assert_leakage_free_split


def assert_no_window_leakage(window_metadata: pd.DataFrame) -> None:
    """Keep every derived window with its original trial and split."""

    required = {
        "trial_id",
        "session_id",
        "participant_id",
        "question_id",
        "role_id",
        "window_index",
        "split",
    }
    missing = required.difference(window_metadata.columns)
    if missing:
        raise AssertionError(f"Window metadata missing columns: {sorted(missing)}")
    if window_metadata[["trial_id", "window_index"]].duplicated().any():
        raise AssertionError("Duplicate (trial_id, window_index) rows")
    for column in (
        "split",
        "session_id",
        "participant_id",
        "question_id",
        "role_id",
    ):
        counts = window_metadata.groupby("trial_id")[column].nunique()
        if not counts.eq(1).all():
            raise AssertionError(f"Windows from a trial disagree on {column}")


def summarize_metadata(metadata: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-ready raw-audit and, when present, split summary."""

    phase_durations: dict[str, dict[str, float]] = {}
    for phase in EVENT_PHASES:
        column = f"{phase}_duration_seconds"
        values = metadata[column].dropna().to_numpy(dtype=float)
        phase_durations[phase] = {
            "minimum": float(np.min(values)) if len(values) else np.nan,
            "median": float(np.median(values)) if len(values) else np.nan,
            "maximum": float(np.max(values)) if len(values) else np.nan,
        }
    summary: dict[str, Any] = {
        "trial_count": int(len(metadata)),
        "session_count": int(metadata["session_id"].nunique()),
        "participant_count": int(metadata["participant_id"].nunique()),
        "role_counts": {
            str(int(role)): int(count)
            for role, count in metadata["role_id"].value_counts().sort_index().items()
        },
        "question_counts": {
            str(question): int(count)
            for question, count in metadata["question_id"].value_counts().sort_index().items()
        },
        "session_trial_counts": {
            session: int(count)
            for session, count in sorted(
                metadata["session_id"].value_counts().items(),
                key=lambda item: session_sort_key(item[0]),
            )
        },
        "sampling_rates_hz": sorted(
            float(value) for value in metadata["sampling_rate_hz"].dropna().unique()
        ),
        "channel_counts": sorted(
            int(value) for value in metadata["n_channels"].dropna().unique()
        ),
        "structurally_valid_count": int(metadata["structurally_valid"].sum()),
        "structurally_invalid_count": int((~metadata["structurally_valid"]).sum()),
        "phase_duration_seconds": phase_durations,
        "structural_issue_counts": dict(
            Counter(
                issue
                for cell in metadata["structural_issues"].fillna("")
                for issue in str(cell).split("|")
                if issue
            )
        ),
        "duration_issue_counts": dict(
            Counter(
                issue
                for cell in metadata["duration_issues"].fillna("")
                for issue in str(cell).split("|")
                if issue
            )
        ),
        "noncanonical_question_padding_files": metadata.loc[
            metadata["original_question_id"].astype(str)
            != metadata["question_id"].astype(str),
            "filename",
        ].tolist(),
    }
    if "split" in metadata.columns:
        summary["split_counts"] = {
            name: int(count) for name, count in metadata["split"].value_counts().items()
        }
        summary["split_sessions"] = {
            name: sorted(
                metadata.loc[metadata["split"] == name, "session_id"].unique().tolist(),
                key=session_sort_key,
            )
            for name in ("train", "test")
        }
    return summary


def save_metadata_table(metadata: pd.DataFrame, destination: str | Path) -> Path:
    """Save the complete trial-level audit table."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(path, index=False)
    return path


def save_split_assignments(metadata: pd.DataFrame, destination: str | Path) -> Path:
    """Save one immutable split-assignment row for every original trial."""

    assert_leakage_free_split(metadata)
    columns = [
        "raw_row_index",
        "trial_id",
        "filepath",
        "session_id",
        "participant_id",
        "role_id",
        "role_name",
        "question_id",
        "split",
        "qc_status",
        "rejection_reason",
    ]
    available = [column for column in columns if column in metadata.columns]
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata[available].to_csv(path, index=False)
    return path


def save_audit_artifacts(
    metadata: pd.DataFrame,
    output_dir: str | Path,
    split_summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Save metadata, split rows, summaries, and leakage evidence."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = save_metadata_table(metadata, destination / "dataset_metadata.csv")
    paths = {"metadata": str(metadata_path)}
    _write_json(destination / "dataset_summary.json", summarize_metadata(metadata))
    paths["dataset_summary"] = str(destination / "dataset_summary.json")
    if "split" in metadata.columns:
        split_path = save_split_assignments(metadata, destination / "split_assignments.csv")
        leakage_report = assert_leakage_free_split(metadata)
        _write_json(destination / "leakage_checks.json", leakage_report)
        _write_json(
            destination / "split_summary.json",
            dict(split_summary) if split_summary is not None else leakage_report,
        )
        paths.update(
            {
                "split_assignments": str(split_path),
                "leakage_checks": str(destination / "leakage_checks.json"),
                "split_summary": str(destination / "split_summary.json"),
            }
        )
    return paths


__all__ = [
    "EVENT_FIELDS",
    "EVENT_PHASES",
    "EXPECTED_TEST_SESSIONS",
    "EXPECTED_TRAIN_SESSIONS",
    "SPLIT_RANDOM_STATE",
    "SPLIT_TEST_SIZE",
    "assert_leakage_free_split",
    "assert_no_window_leakage",
    "assert_raw_dataset_design",
    "assign_group_split",
    "build_metadata_table",
    "canonical_participant",
    "canonical_question",
    "canonical_role",
    "canonical_session",
    "create_group_split",
    "create_metadata_table",
    "discover_mat_files",
    "inspect_dataset",
    "inspect_mat_file",
    "locate_dataset_root",
    "locate_paper",
    "make_trial_id",
    "parse_trial_filename",
    "run_leakage_checks",
    "save_audit_artifacts",
    "save_metadata_table",
    "save_split_assignments",
    "session_sort_key",
    "summarize_metadata",
]
