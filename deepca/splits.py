"""Strict parsing and canonicalization for ImageCAS split JSON files.

The split files used by the ImageCAS-derived datasets exist in several small
schema variants.  This module accepts those documented variants while making
case identity unambiguous before any files are opened.  In particular, all
duplicate and leakage checks operate on ``(vessel_type, case_number)`` rather
than on the spelling found in the JSON document.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CANONICAL_SPLITS = ("train", "val", "test")
SPLIT_ALIASES: Mapping[str, frozenset[str]] = {
    "train": frozenset(("train", "training")),
    "val": frozenset(("val", "validation", "valid", "dev")),
    "test": frozenset(("test", "testing")),
}
CONTAINER_KEYS = frozenset(("splits", "partitions", "dataset"))
IDENTIFIER_FIELDS = frozenset(
    (
        "path",
        "file",
        "source_path",
        "case_name",
        "sample_name",
        "case_id",
        "case_number",
        "case",
        "id",
        "name",
    )
)
VESSEL_TYPES = frozenset(("lca", "rca"))

_TYPED_COMPONENT = re.compile(
    r"^(?P<vessel>lca|rca)[_-]?(?P<number>\d+)(?:\.npz)?$", re.IGNORECASE
)
_NUMERIC_COMPONENT = re.compile(r"^(?P<number>\d+)(?:\.npz)?$", re.IGNORECASE)


class SplitError(ValueError):
    """Raised when a split document cannot be resolved without ambiguity."""


@dataclass(frozen=True, order=True)
class CanonicalCase:
    """A vessel-qualified ImageCAS case identity."""

    vessel_type: str
    case_number: int

    @property
    def canonical_id(self) -> str:
        """Return the stable filename-style identity, with at least four digits."""

        return f"{self.vessel_type}_{self.case_number:04d}"


@dataclass(frozen=True)
class IdentifierProvenance:
    """One identifier field used to resolve a split entry."""

    field: str | None
    value: str | int | float

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "value": self.value}


@dataclass(frozen=True)
class CaseProvenance:
    """Where a canonical case came from in the source JSON document."""

    canonical_id: str
    vessel_type: str
    case_number: int
    split: str
    source_alias: str
    index: int
    location: str
    original_entry: Any
    identifiers: tuple[IdentifierProvenance, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable provenance for an experiment manifest."""

        return {
            "canonical_id": self.canonical_id,
            "vessel_type": self.vessel_type,
            "case_number": self.case_number,
            "source_alias": self.source_alias,
            "source_index": self.index,
            "source_location": self.location,
            "source_entry": copy.deepcopy(self.original_entry),
            "identifiers": [item.as_dict() for item in self.identifiers],
        }


@dataclass
class ResolvedSplits:
    """Canonical split lists together with source and per-entry provenance."""

    train: list[str]
    val: list[str]
    test: list[str]
    provenance: dict[str, list[CaseProvenance]]
    source_path: str
    source_sha256: str
    container_path: tuple[str, ...]
    aliases: dict[str, str]
    expected_vessel: str

    @property
    def split_ids(self) -> dict[str, list[str]]:
        """Return copies of the canonical ID lists keyed by canonical split name."""

        return {
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
        }

    def as_manifest(self) -> dict[str, Any]:
        """Return a JSON-serializable record suitable for experiment artifacts."""

        ids = self.split_ids
        return {
            "schema_version": 1,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "container_path": list(self.container_path),
            "source_aliases": dict(self.aliases),
            "expected_vessel": self.expected_vessel,
            "splits": {
                split: {
                    "case_ids": ids[split],
                    "entries": [item.as_dict() for item in self.provenance[split]],
                }
                for split in CANONICAL_SPLITS
            },
        }


def _normalize_expected_vessel(expected_vessel: str | None) -> str | None:
    if expected_vessel is None:
        return None
    if not isinstance(expected_vessel, str):
        raise SplitError("expected_vessel must be 'lca' or 'rca'.")
    vessel = expected_vessel.strip().casefold()
    if vessel not in VESSEL_TYPES:
        raise SplitError(
            f"Unsupported expected_vessel {expected_vessel!r}; expected 'lca' or 'rca'."
        )
    return vessel


def _checked_case(vessel_type: str, case_number: int) -> CanonicalCase:
    vessel = vessel_type.casefold()
    if vessel not in VESSEL_TYPES:
        raise SplitError(f"Unsupported vessel type {vessel_type!r}.")
    if case_number <= 0:
        raise SplitError(f"Case number must be positive, got {case_number!r}.")
    return CanonicalCase(vessel, case_number)


def _case_from_number(
    value: numbers.Real, expected_vessel: str | None
) -> CanonicalCase:
    if isinstance(value, bool):
        raise SplitError("Boolean values are not valid case identifiers.")
    if expected_vessel is None:
        raise SplitError(
            f"Bare case identifier {value!r} requires a configured vessel type."
        )
    if isinstance(value, numbers.Integral):
        number = int(value)
    elif isinstance(value, numbers.Real):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise SplitError(
                f"Numeric case identifier must be a finite integer, got {value!r}."
            )
        number = int(numeric)
    else:  # Defensive: callers normally filter this before entry.
        raise SplitError(f"Unsupported numeric case identifier {value!r}.")
    return _checked_case(expected_vessel, number)


def _case_from_text(value: str, expected_vessel: str | None) -> CanonicalCase:
    text = value.strip()
    if not text:
        raise SplitError("Case identifier strings must not be empty.")

    # Splitting both slash conventions makes Windows paths deterministic on Linux.
    normalized = text.replace("\\", "/")
    components = tuple(component for component in normalized.split("/") if component)
    candidates: list[tuple[CanonicalCase, str]] = []

    for component in components:
        match = _TYPED_COMPONENT.fullmatch(component)
        if match:
            case = _checked_case(match.group("vessel"), int(match.group("number")))
            candidates.append((case, component))

    # A directory pair such as lca/1 is authoritative.  This deliberately does
    # not interpret suffixes such as prefix_02.npz as case identifiers.
    for index, component in enumerate(components[:-1]):
        vessel = component.casefold()
        if vessel not in VESSEL_TYPES:
            continue
        match = _NUMERIC_COMPONENT.fullmatch(components[index + 1])
        if match:
            case = _checked_case(vessel, int(match.group("number")))
            candidates.append(
                (case, f"{component}/{components[index + 1]}")
            )

    if candidates:
        distinct = {case for case, _ in candidates}
        if len(distinct) != 1:
            descriptions = ", ".join(
                f"{source!r}->{case.canonical_id}" for case, source in candidates
            )
            raise SplitError(
                f"Ambiguous case identifier {value!r}; conflicting path tokens: "
                f"{descriptions}."
            )
        case = next(iter(distinct))
        if expected_vessel is not None and case.vessel_type != expected_vessel:
            raise SplitError(
                f"Case {case.canonical_id} is {case.vessel_type.upper()}, but this "
                f"split is configured for {expected_vessel.upper()}."
            )
        return case

    # A bare number, or a path ending in a numeric NPZ filename, has no vessel
    # information and is valid only in a vessel-specific dataset configuration.
    final_component = components[-1] if components else normalized
    match = _NUMERIC_COMPONENT.fullmatch(final_component)
    if match:
        if expected_vessel is None:
            raise SplitError(
                f"Bare case identifier {value!r} requires a configured vessel type."
            )
        return _checked_case(expected_vessel, int(match.group("number")))

    raise SplitError(
        f"Could not resolve case identifier {value!r}; expected an ImageCAS name, "
        "a vessel/case path, or a bare numeric ID with configured vessel type."
    )


def _canonicalize_scalar(value: Any, expected_vessel: str | None) -> CanonicalCase:
    if isinstance(value, bool):
        raise SplitError("Boolean values are not valid case identifiers.")
    if isinstance(value, str):
        return _case_from_text(value, expected_vessel)
    if isinstance(value, numbers.Real):
        return _case_from_number(value, expected_vessel)
    raise SplitError(
        f"Case identifier must be a string or number, got {type(value).__name__}."
    )


def _resolve_entry(
    entry: Any, expected_vessel: str | None
) -> tuple[CanonicalCase, tuple[IdentifierProvenance, ...]]:
    if not isinstance(entry, Mapping):
        case = _canonicalize_scalar(entry, expected_vessel)
        return case, (IdentifierProvenance(None, entry),)

    identifiers: list[tuple[str, Any]] = [
        (str(key), value)
        for key, value in entry.items()
        if isinstance(key, str) and key.casefold() in IDENTIFIER_FIELDS
    ]
    if not identifiers:
        expected = ", ".join(sorted(IDENTIFIER_FIELDS))
        raise SplitError(
            f"Case record has no supported identifier field; expected one of: {expected}."
        )

    resolved: list[tuple[str, Any, CanonicalCase]] = []
    for field, value in identifiers:
        try:
            case = _canonicalize_scalar(value, expected_vessel)
        except SplitError as error:
            raise SplitError(f"Invalid record field {field!r}: {error}") from error
        resolved.append((field, value, case))

    distinct = {case for _, _, case in resolved}
    if len(distinct) != 1:
        detail = ", ".join(
            f"{field}={value!r}->{case.canonical_id}"
            for field, value, case in resolved
        )
        raise SplitError(f"Conflicting case identifiers in record: {detail}.")

    case = next(iter(distinct))
    provenance = tuple(
        IdentifierProvenance(field, value) for field, value, _ in resolved
    )
    return case, provenance


def resolve_case_identifier(
    value: Any, expected_vessel: str | None = None
) -> CanonicalCase:
    """Resolve one scalar or record identifier to a vessel-qualified case.

    ``expected_vessel`` may be omitted only when every identifier contains an
    explicit LCA/RCA token.  When a record contains multiple supported fields,
    all fields must resolve to the same canonical case.
    """

    vessel = _normalize_expected_vessel(expected_vessel)
    case, _ = _resolve_entry(value, vessel)
    return case


def canonicalize_case_id(value: Any, expected_vessel: str | None = None) -> str:
    """Return the canonical ``lca_NNNN``/``rca_NNNN`` string for one entry."""

    return resolve_case_identifier(value, expected_vessel).canonical_id


def _format_location(path: Sequence[str]) -> str:
    return "$" + "".join(f".{component}" for component in path)


def _find_candidate_containers(
    root: Mapping[str, Any],
) -> list[tuple[tuple[str, ...], Mapping[str, Any]]]:
    candidates: list[tuple[tuple[str, ...], Mapping[str, Any]]] = []

    def visit(node: Mapping[str, Any], path: tuple[str, ...]) -> None:
        folded_keys = {
            key.casefold()
            for key in node
            if isinstance(key, str)
        }
        if any(
            folded_key in aliases
            for aliases in SPLIT_ALIASES.values()
            for folded_key in folded_keys
        ):
            candidates.append((path, node))

        for key, value in node.items():
            if (
                isinstance(key, str)
                and key.casefold() in CONTAINER_KEYS
                and isinstance(value, Mapping)
            ):
                visit(value, (*path, key))

    visit(root, ())
    return candidates


def _select_aliases(container: Mapping[str, Any], location: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        matches = [
            key
            for key in container
            if isinstance(key, str) and key.casefold() in aliases
        ]
        if not matches:
            accepted = ", ".join(sorted(aliases))
            raise SplitError(
                f"Split container {location} is missing {canonical!r}; accepted keys: "
                f"{accepted}."
            )
        if len(matches) > 1:
            names = ", ".join(repr(item) for item in matches)
            raise SplitError(
                f"Split container {location} defines multiple aliases for "
                f"{canonical!r}: {names}."
            )
        selected[canonical] = matches[0]
    return selected


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SplitError(f"JSON object contains duplicate key {key!r}.")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise SplitError(f"Could not read split JSON {path}: {error}") from error
    try:
        document = json.loads(
            payload.decode("utf-8-sig"), object_pairs_hook=_unique_json_object
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SplitError(f"Invalid UTF-8 JSON in {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise SplitError(f"Split JSON {path} must contain an object at the top level.")
    return document, payload


def load_resolved_splits(
    path: str | Path, expected_vessel: str
) -> ResolvedSplits:
    """Load, canonicalize, and validate an ImageCAS split JSON file.

    The returned ``train``, ``val``, and ``test`` lists preserve source order.
    Duplicate cases, aliases, ambiguous containers, vessel mismatches, and any
    cross-split leakage are rejected after canonicalization.
    """

    vessel = _normalize_expected_vessel(expected_vessel)
    if vessel is None:
        raise SplitError("load_resolved_splits requires expected_vessel='lca' or 'rca'.")
    source = Path(path).expanduser().resolve()
    document, payload = _load_json(source)

    candidates = _find_candidate_containers(document)
    if not candidates:
        raise SplitError(
            f"No train/validation/test split container found in {source}; only "
            f"top-level or nested {sorted(CONTAINER_KEYS)} containers are supported."
        )
    if len(candidates) > 1:
        locations = ", ".join(_format_location(item[0]) for item in candidates)
        raise SplitError(
            f"Multiple candidate split containers found in {source}: {locations}."
        )

    container_path, container = candidates[0]
    container_location = _format_location(container_path)
    aliases = _select_aliases(container, container_location)

    split_ids: dict[str, list[str]] = {}
    provenance: dict[str, list[CaseProvenance]] = {}
    for split in CANONICAL_SPLITS:
        source_alias = aliases[split]
        entries = container[source_alias]
        if not isinstance(entries, list):
            raise SplitError(
                f"{container_location}.{source_alias} must be a JSON list, got "
                f"{type(entries).__name__}."
            )

        ids: list[str] = []
        records: list[CaseProvenance] = []
        first_location: dict[str, str] = {}
        for index, entry in enumerate(entries):
            location = f"{container_location}.{source_alias}[{index}]"
            try:
                case, identifiers = _resolve_entry(entry, vessel)
            except SplitError as error:
                raise SplitError(f"{location}: {error}") from error
            canonical_id = case.canonical_id
            if canonical_id in first_location:
                raise SplitError(
                    f"Duplicate case {canonical_id} within {split!r}: "
                    f"{first_location[canonical_id]} and {location}."
                )
            first_location[canonical_id] = location
            ids.append(canonical_id)
            records.append(
                CaseProvenance(
                    canonical_id=canonical_id,
                    vessel_type=case.vessel_type,
                    case_number=case.case_number,
                    split=split,
                    source_alias=source_alias,
                    index=index,
                    location=location,
                    original_entry=copy.deepcopy(entry),
                    identifiers=identifiers,
                )
            )
        split_ids[split] = ids
        provenance[split] = records

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        leaked = set(split_ids[left]).intersection(split_ids[right])
        if leaked:
            ordered = [item for item in split_ids[left] if item in leaked]
            raise SplitError(
                f"Cross-split leakage between {left!r} and {right!r}: "
                f"{', '.join(ordered)}."
            )

    return ResolvedSplits(
        train=split_ids["train"],
        val=split_ids["val"],
        test=split_ids["test"],
        provenance=provenance,
        source_path=str(source),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        container_path=container_path,
        aliases=aliases,
        expected_vessel=vessel,
    )


__all__ = [
    "CANONICAL_SPLITS",
    "CanonicalCase",
    "CaseProvenance",
    "IdentifierProvenance",
    "ResolvedSplits",
    "SplitError",
    "canonicalize_case_id",
    "load_resolved_splits",
    "resolve_case_identifier",
]
