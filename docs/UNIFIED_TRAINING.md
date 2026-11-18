# Shared RCA + LCA training

`configs/imagecas_unified.yaml` trains one generator and critic on both artery
cohorts, using two views per sample. It references the same server paths as the
existing RCA and LCA configs; no data copying or NPZ conversion is required.
Run from the repository root in the existing DeepCA training environment:

```bash
python train.py --config configs/imagecas_unified.yaml --dry-run
python train.py --config configs/imagecas_unified.yaml --device cuda:0
```

For the existing variable-view architecture, randomly selecting 1–7 available
views during training, use instead:

```bash
python train.py --config configs/imagecas_unified_variable_views.yaml --dry-run
python train.py --config configs/imagecas_unified_variable_views.yaml --device cuda:0
```

The variable-view config validates with one view, matching the existing
single-artery variable-view recipes. Set `data.views.count: 2` to select the
checkpoint using two-view validation instead.

Resume the fixed two-view run with:

```bash
python train.py --config configs/imagecas_unified.yaml \
  --resume /export/home2/reny0012/result/deepca_imagecas/unified/checkpoints/last.pt
```

If your paths differ, edit `data.sources.rca` and `data.sources.lca` in the
unified config: each has `projection_root`, `ground_truth_root`, `split_json`,
and `calibration`. Ground truths remain `<ground_truth_root>/<rca|lca>/<number>.npz`.
Shared preprocessing, views, model and training options inherit from
`imagecas_common.yaml`. Source calibration overrides the shared defaults:
RCA uses 0.55 mm and LCA uses 0.65 mm detector spacing. Stored calibration remains
authoritative and is checked against the configured expected value.

Both arteries are shuffled together, with each case visited once per epoch
(no artery balancing or dropped final batch). Each target remains the binary
volume of that sample's artery. This trains a shared reconstruction model;
it does not merge the RCA and LCA targets into a whole coronary-tree target.
The model architecture, losses and optimizer are unchanged. Combining cohorts
is technically supported; improvement over separate models needs measurement.

The same numeric ImageCAS case ID is treated as one patient across arteries.
The supplied `patient_split_policy: holdout_priority` resolves conflicting
original splits using `test > val > train`. Thus a held-out patient's other artery
cannot enter training. Set this policy to `error` to require pre-aligned split
files instead. Empty training or validation sets fail explicitly. Review
`resolved_cases.json` for original source provenance, final file pairings and
`patient_split_moves`; the source split files are never rewritten. For a fair
comparison, train the separate baselines with these same patient partitions.

The dry run checks split resolution and file pairing and prints per-artery case
counts. It does not preprocess all archives or exercise the GPU. Training writes
`epochs.jsonl` with combined validation loss plus `rca_l1_loss` and `lca_l1_loss`,
and saves `checkpoints/best.pt` and `checkpoints/last.pt`. Checkpoint selection
uses case-weighted combined validation L1, so a larger cohort has more weight.

## Evaluate the unified model on the test set

Run the standard two-view model with:

```bash
python evaluate_unified.py --config configs/eval_imagecas_unified.yaml
```

For the variable-view model (evaluates 1–7 views separately):

```bash
python evaluate_unified.py --config configs/eval_imagecas_unified_variable_views.yaml
```

Both configurations default to `split: test`, `save_predictions: true`, and the
corresponding unified `checkpoints/best.pt`. The checkpoint was selected using
validation loss; the test set is used only for evaluation. The script uses the
final patient-safe case lists saved in the checkpoint, including any holdout
promotions, rather than the original separate split JSON files. Only the
selected split's projection/ground-truth pairs are required. Source paths may
be relocated in the config; calibration and model compatibility are checked.

For a different checkpoint or view counts:

```bash
python evaluate_unified.py --config configs/eval_imagecas_unified.yaml \
  --checkpoint /path/to/best.pt --split test --view-counts 2
```

No `--device` is needed under the usual single-GPU Slurm allocation. The
configuration uses `cuda:0` within the job's visible devices.

Default two-view output directory:
`/export/home2/reny0012/result/deepca_imagecas/unified/evaluation/test/`

```text
predictions/rca/views_2/rca_NNNN.npz
predictions/lca/views_2/lca_NNNN.npz
per_case.csv
per_case.json
summary.json
failures.json
resolved_config.yaml
```

The existing evaluator supplies the binary prediction NPZ format, physical grid
metadata, Dice, clDice, reconstruction/inference timing, per-case failure
reporting, and overall/per-artery/per-view summaries. Variable-view outputs use
`unified_variable_views/evaluation/test/`, with separate `views_1` through
`views_7` folders under each artery. Inspect `failures.json` for unavailable
views or missing/corrupt cases; as in the existing evaluator, a partial run can
produce metrics for the successful cases. Standard evaluation is supported;
perturbation sweep modes are not yet supported for unified configs.
