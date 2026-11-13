"""Resolve separate artery archives into a patient-safe shared training cohort."""
from __future__ import annotations

from typing import Any, Mapping

from .data import build_projection_index, resolve_case_pairs, parse_typed_case_id
from .splits import load_resolved_splits


def training_sources(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    vessel = str(data.get("vessel_type", "")).lower()
    if vessel == "both":
        sources = data.get("sources")
        if not isinstance(sources, Mapping) or set(sources) != {"rca", "lca"}:
            raise ValueError("data.sources must contain exactly rca and lca for vessel_type: both.")
    elif vessel in {"rca", "lca"}:
        sources = {vessel: data}
    else:
        raise ValueError("data.vessel_type must be 'lca', 'rca', or 'both'.")
    for name, source in sources.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"Source {name} must be a mapping.")
        for key in ("projection_root", "ground_truth_root", "split_json"):
            if not source.get(key):
                raise ValueError(f"Source {name} requires {key}.")
    return dict(sources)


def resolve_training_pairs(data: Mapping[str, Any]):
    sources = training_sources(data)
    resolved = {v: load_resolved_splits(s["split_json"], v) for v, s in sources.items()}
    assignments = {v: r.split_ids for v, r in resolved.items()}
    moves = []
    if len(sources) > 1:
        policy = data.get("patient_split_policy", "error")
        if policy not in {"error", "holdout_priority"}:
            raise ValueError("patient_split_policy must be error or holdout_priority.")
        # ImageCAS artery trees with the same numeric ID originate from one CT.
        priority = {"train": 0, "val": 1, "test": 2}
        patients: dict[int, set[str]] = {}
        for splits in assignments.values():
            for split, ids in splits.items():
                for case_id in ids:
                    patients.setdefault(parse_typed_case_id(case_id)[1], set()).add(split)
        conflicts = {n: splits for n, splits in patients.items() if len(splits) > 1}
        if conflicts and policy == "error":
            raise ValueError(f"Cross-artery patient split leakage: {sorted(conflicts)}. "
                             "Align splits or set patient_split_policy: holdout_priority.")
        for vessel, splits in assignments.items():
            updated = {split: [] for split in priority}
            for split, ids in splits.items():
                for case_id in ids:
                    number = parse_typed_case_id(case_id)[1]
                    destination = max(patients[number], key=priority.get)
                    updated[destination].append(case_id)
                    if destination != split:
                        moves.append({"case_id": case_id, "from": split, "to": destination})
            assignments[vessel] = updated
    pairs_by_split = {split: [] for split in ("train", "val", "test")}
    for vessel, source in sources.items():
        index = build_projection_index(source["projection_root"], expected_vessel=vessel)
        for split, ids in assignments[vessel].items():
            pairs, _ = resolve_case_pairs(ids, projection_root=source["projection_root"],
                ground_truth_root=source["ground_truth_root"], vessel_type=vessel,
                projection_index=index, strict=True)
            pairs_by_split[split].extend(pairs)
    for split in ("train", "val"):
        if not pairs_by_split[split]:
            raise ValueError(f"Resolved {split} split is empty.")
    if len(sources) == 1:
        manifest = next(iter(resolved.values())).as_manifest()
    else:
        manifest = {"schema_version": 2, "expected_vessel": "both",
                    "sources": {v: r.as_manifest() for v, r in resolved.items()},
                    "patient_split_policy": data.get("patient_split_policy", "error"),
                    "patient_split_moves": moves,
                    "splits": {s: {"case_ids": [p.case_id for p in pairs]}
                               for s, pairs in pairs_by_split.items()}}
    return pairs_by_split, manifest
