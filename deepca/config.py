"""YAML configuration loading, inheritance, and validation helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, resolving an optional relative ``extends`` chain."""

    config_path = Path(path).expanduser().resolve()
    seen: set[Path] = set()

    def load_one(current: Path) -> dict[str, Any]:
        if current in seen:
            chain = " -> ".join(str(item) for item in (*seen, current))
            raise ValueError(f"Configuration inheritance cycle: {chain}")
        if not current.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {current}")
        seen.add(current)
        with current.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if not isinstance(document, MutableMapping):
            raise TypeError(f"Configuration {current} must contain a YAML mapping.")
        parent = document.pop("extends", None)
        if parent is None:
            merged = dict(document)
        else:
            if not isinstance(parent, str) or not parent.strip():
                raise TypeError(f"{current}: extends must be a non-empty path string.")
            parent_path = Path(parent).expanduser()
            if not parent_path.is_absolute():
                parent_path = current.parent / parent_path
            merged = _deep_merge(load_one(parent_path.resolve()), document)
        seen.remove(current)
        return merged

    config = load_one(config_path)
    config["_config_path"] = str(config_path)
    return config


def require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise KeyError(f"Configuration requires a {key!r} mapping.")
    return value


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable hash, excluding the machine-specific config path."""

    payload = {key: value for key, value in config.items() if key != "_config_path"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# These fields control where/how a run executes, but not the experiment being
# resumed. Everything not listed here is deliberately part of the resume
# contract, including any newly introduced configuration field.
_RESUME_IGNORED_TOP_LEVEL = frozenset({"_config_path", "evaluation"})
_RESUME_IGNORED_NESTED: Mapping[str, frozenset[str]] = {
    "experiment": frozenset({"name", "output_dir"}),
    "training": frozenset({"device", "epochs", "resume", "workers"}),
}


def _canonical_config_value(value: Any) -> Any:
    """Return a recursively comparable representation of a config value."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return copy.deepcopy(value)


def normalize_config_for_resume(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the scientific/training contract used for safe resume.

    Only explicitly runtime-only fields are removed. In particular, the data,
    model, loss, optimizer, scheduler, batch-size, mixed-precision, and seed
    configuration remain part of the comparison. Unknown future fields are
    retained by default so they cannot silently change across a resume.
    """

    if not isinstance(config, Mapping):
        raise TypeError("Resume configuration must be a mapping.")
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in sorted(config.items(), key=lambda pair: str(pair[0])):
        key = str(raw_key)
        if key in _RESUME_IGNORED_TOP_LEVEL:
            continue
        value = raw_value
        ignored_nested = _RESUME_IGNORED_NESTED.get(key)
        if ignored_nested is not None and isinstance(value, Mapping):
            value = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if str(nested_key) not in ignored_nested
            }
        normalized[key] = _canonical_config_value(value)
    return normalized


_MISSING = object()


def _resume_config_differences(
    checkpoint_value: Any,
    current_value: Any,
    *,
    path: str = "",
) -> list[str]:
    if isinstance(checkpoint_value, Mapping) and isinstance(current_value, Mapping):
        differences: list[str] = []
        for key in sorted(set(checkpoint_value) | set(current_value), key=str):
            child_path = f"{path}.{key}" if path else str(key)
            differences.extend(
                _resume_config_differences(
                    checkpoint_value.get(key, _MISSING),
                    current_value.get(key, _MISSING),
                    path=child_path,
                )
            )
        return differences
    if checkpoint_value is _MISSING or current_value is _MISSING:
        checkpoint_text = (
            "<missing>" if checkpoint_value is _MISSING else repr(checkpoint_value)
        )
        current_text = "<missing>" if current_value is _MISSING else repr(current_value)
        return [f"{path}: checkpoint={checkpoint_text}, current={current_text}"]
    if checkpoint_value != current_value:
        return [f"{path}: checkpoint={checkpoint_value!r}, current={current_value!r}"]
    return []


def assert_resume_config_compatible(
    current_config: Mapping[str, Any], checkpoint_config: Mapping[str, Any]
) -> None:
    """Raise with field-level diagnostics when a resume would change its contract."""

    current = normalize_config_for_resume(current_config)
    checkpoint = normalize_config_for_resume(checkpoint_config)
    differences = _resume_config_differences(checkpoint, current)
    if differences:
        rendered = "\n".join(f"  - {difference}" for difference in differences)
        raise ValueError(
            "Resume configuration is incompatible with the checkpoint. Only "
            "runtime fields (_config_path, evaluation, experiment.name/output_dir, "
            "and training.device/epochs/resume/workers) may change. Mismatches:\n"
            f"{rendered}"
        )


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in config.items() if key != "_config_path"}
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
