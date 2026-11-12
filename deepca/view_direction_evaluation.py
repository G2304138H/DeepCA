"""Four-condition deterministic camera-error evaluation, with fixed images."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .config import save_resolved_config


def validate_error(value, *, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError("Angular error must be numeric, not boolean")
    value = float(value)
    if not math.isfinite(value) or value > 180 or value < 0 or (value == 0 and not allow_zero):
        raise ValueError("Angular error must be finite and in (0, 180] degrees")
    return value


def resolve_errors(evaluation):
    options = evaluation.get("view_direction_robustness", {})
    if not isinstance(options, dict):
        raise ValueError("view_direction_robustness must be a mapping")
    unknown = set(options) - {"error_degrees"}
    if unknown:
        raise ValueError(f"Unknown view_direction_robustness keys: {sorted(unknown)}")
    errors = options.get("error_degrees", [.2, .5, 1., 2.])
    if not isinstance(errors, (list, tuple)) or not errors:
        raise ValueError("error_degrees must be a nonempty list")
    values = [validate_error(error) for error in errors]
    if len(set(values)) != len(values):
        raise ValueError("error_degrees must be unique")
    return values


def run_sweep(config, args, evaluate):
    """Reuse standard evaluation, changing only camera input and output location.

    Each condition must evaluate exactly the same cases/views without failures.
    Baseline remains a separate standard evaluation run.
    """
    errors = resolve_errors(config["evaluation"])
    if getattr(args, "no_save_predictions", False):
        raise ValueError("View-direction sweeps require saving prediction NPZ files")
    root = Path(config["evaluation"]["output_dir"]).expanduser().resolve()
    for error in errors:
        name = "error_" + format(error, ".12g").replace(".", "p") + "deg"
        if (root / name).exists():
            raise FileExistsError(f"Use a fresh sweep output directory: {root / name}")
    run_configs = root / "run_configs"
    run_configs.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", error_model="fixed_magnitude_meridian_tilt",
                    error_degrees=errors, perturbation_scope="first_input_view", images_changed=False, results=[])
    summary_path = root / "view_direction_robustness_summary.json"
    def save():
        summary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    expected = None
    save()
    for error in errors:
        identifier = "error_" + format(error, ".12g").replace(".", "p") + "deg"
        child = copy.deepcopy(config)
        child["evaluation"].update(mode="standard", view_direction_error_deg=error, save_predictions=True,
                                    output_dir=str(root / identifier))
        child["evaluation"].pop("view_direction_robustness", None)
        path = run_configs / (identifier + ".yaml")
        save_resolved_config(child, path)
        argv = ["--config", str(path)]
        for key in ("split", "checkpoint", "device", "threshold", "view_counts"):
            value = getattr(args, key, None)
            if value is not None:
                argv.extend(["--" + key.replace("_", "-"), str(value)])
        for key in ("allow_checkpoint_config_mismatch",):
            if getattr(args, key, False):
                argv.append("--" + key.replace("_", "-"))
        try:
            status = evaluate(argv)
            if status:
                raise RuntimeError(f"Evaluation exited with status {status}")
            summaries = list((root / identifier).glob("*/summary.json"))
            if len(summaries) != 1:
                raise RuntimeError("Expected exactly one split summary; use a fresh sweep output directory")
            result_path = summaries[0]
            summary = json.loads(result_path.read_text())
            if summary["context"]["failed_results"]:
                raise RuntimeError("Condition contains failed cases")
            cases = json.loads((result_path.parent / "per_case.json").read_text())["results"]
            identities = [(row["case_id"], row["num_views"], row["view_indices"]) for row in cases]
            if expected is not None and identities != expected:
                raise RuntimeError("Cases/selected views differ between error levels")
            expected = identities
            manifest["results"].append(dict(error_degrees=error, summary=summary,
                                             summary_path=str(result_path)))
            save()
        except Exception as exc:
            manifest.update(status="failed", failed_condition=identifier, reason=str(exc))
            save()
            raise
    manifest.update(status="complete", num_conditions=len(errors))
    save()
    return 0
