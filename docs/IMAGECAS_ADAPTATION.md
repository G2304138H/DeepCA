# DeepCA adaptation for ImageCAS NPZ data

This document describes the scientific conventions, environment, and commands
for the ImageCAS adaptation. It distinguishes the released DeepCA workflow from
the replacement needed to consume the supplied NPZ files directly. The commands
below have **not** been run on the Linux training server; GPU, driver, filesystem,
and full-data behavior must be verified there before launching training.

## Released workflow and preserved model

The paper and released code train on a single-channel `128 x 128 x 128` volume,
not directly on 2D images. For each case, the release:

1. forward-projects a binary CCTA vessel volume into two `512 x 512` views with
   TIGRE;
2. introduces translation and angle mismatch in the second view;
3. reconstructs each view separately with one SIRT iteration;
4. thresholds each backprojection at `> 0` and sums them, producing a volume
   with values in `{0, 1, 2}`; and
5. trains a conditional WGAN-GP generator/critic with raw-output L1 loss weighted
   by 100.

In the released simulator, the detector is `512 x 512` at approximately
`0.278 mm/pixel`, the cubic CCTA extent is sampled from `90–105 mm`, and DSO is
approximately `765 +/- 20 mm`. The first view samples primary/secondary angles
around `30 +/- 12` and `0 +/- 8` degrees with DSD approximately
`990 +/- 20 mm`. The second samples around `0 +/- 8` and `30 +/- 12` degrees,
uses DSD approximately `1060 +/- 10 mm`, and projects with additional angle
perturbations up to `+/- 10` degrees and in-plane origin translations up to
`+/- 8 mm`. Those added perturbations are removed for its nominal second-view
reconstruction, creating the intended non-simultaneous-motion mismatch. The
sampled geometry is not saved with the generated `.npy` input.

The adapted code preserves the released generator and dynamic-snake critic,
including their raw generator output. The paper states in section 4.3 that 3D
reconstructions are binarized at `0.5` before evaluation, reprojection, and
visualization. Therefore evaluation thresholds the **raw** generator output at
`0.5`; it does not apply a sigmoid first.

The default architecture remains `128^3`. Configurable cubic sizes must be at
least 32 and divisible by 16. The released `128^3` CCT projection remains
`Linear(640, 512)` with unchanged state-dictionary keys.

## Supplied NPZ schema and units

Projection archives are loaded with `allow_pickle=False`. Required arrays are:

- `images`: finite, non-negative `float` data with shape `[V,H,W]`, where
  `1 <= V <= 7`;
- `theta_deg` and `phi_deg`: one finite angle in degrees per view.

Supported calibration and identity fields are:

- `sid`: source-to-detector distance. `sid_units` may specify metres or
  millimetres; if omitted, metres are assumed.
- `imager_pixel_spacing`: scalar detector spacing. The optional
  `imager_pixel_spacing_units` may specify metres or millimetres; if omitted,
  millimetres are assumed.
- `projection_center_offset`: three physical XYZ coordinates. The optional
  `projection_center_offset_units` is honored. For the supplied Stage-2 files,
  which omit this units field, the configured and documented assumption is
  **metres**; the loader converts the values to millimetres.
- `clinical_views`: optional one-label-per-view string array.
- `sample_name`, `case_name`, `source_relpath`, or a consistent
  `vessel_type`/`case_id` pair may supply identity.
- `view_features`, when present, must equal
  `[sin(theta), cos(theta), sin(phi), cos(phi)]` within tolerance.

The attached RCA example contains seven binary `256 x 256` views, angles,
`view_features`, `projection_center_offset`, `sample_name`, and `source_relpath`.
It contains neither SID nor detector pixel spacing, so both configured fallbacks
are required.

Ground-truth archives require:

- `vol`: a non-negative 3D integer or integer-like labelled volume in explicit
  `[X,Y,Z]` array order; all labels `> 0` become vessel foreground;
- `spacing`: three positive voxel spacings `[sx,sy,sz]` in millimetres.

Ordinary DeepCA training and evaluation ignore extra projection fields such as
`artery`, branch annotations, and basis coefficients. The fixed translation
evaluation documented below is the exception: it requires `artery` and the
recorded `projection_center_offset`, consumes branch/render provenance when
available, and otherwise permits only the template's explicit verified
all-branch cohort assertion.

The attached ground-truth example is `uint8 [512,512,275]`, with spacing
`[0.376953125, 0.376953125, 0.5] mm`. The model tensor is explicitly
`[C,Z,Y,X]`: resampling writes `vol[X,Y,Z]` into a `[Z,Y,X]` model grid. No
implicit `np.transpose` convention is used. The NPZ files do not provide a
direction cosine matrix or physical origin, so identity-oriented XYZ axes and a
source origin of `[0,0,0] mm` are reproduction assumptions.

### Calibration precedence

Calibration stored in an archive always wins. A configured fallback is consulted
only if that field is absent:

| Field | RCA fallback | LCA fallback |
|---|---:|---:|
| detector pixel spacing | `0.55 mm` | `0.65 mm` |
| SID | `0.9 m` | `0.9 m` |
| source-to-isocentre distance | `0.75 m` | `0.75 m` |

Source-to-isocentre distance is currently configuration-only. A stored detector
spacing that conflicts with the configured cohort expectation raises an error;
it is never silently overwritten. Invalid units, non-finite values, `SID <= SOD`,
and inconsistent view counts also raise errors.

### Case matching and supplied splits

`lca_0001.npz` and `rca_0001.npz` map only to
`<ground_truth_root>/lca/1.npz` and `<ground_truth_root>/rca/1.npz`, respectively.
The vessel prefix is validated on both sides; missing and ambiguous matches are
errors rather than skipped samples.

The configured JSON is authoritative—no random replacement split is generated.
The parser accepts `train`/`training`, `val`/`validation`/`valid`/`dev`, and
`test`/`testing`, including recognized nested split containers and the documented
scalar/path/record identifier forms. Canonicalization happens before duplicate
and cross-split leakage checks. Fully resolved case paths and source split
provenance are stored with each run and checkpoint. LCA and RCA use separate
configs, caches, output directories, and model runs.

Schema-v2 manifests may contain legacy path lists at the root and richer records
under `splits`. This mirrored form is accepted only when the ordered canonical
IDs agree for all three partitions; the richer nested records are then used for
provenance. Conflicting containers remain an error.

In grouped LCA manifests, `case_name` may be an artifact label such as
`prefix_02` while the physical identity appears in a path such as
`lca/23/prefix_02.npz`. The path is authoritative only when the label exactly
matches that path's filename stem; other invalid or conflicting identifier
fields remain errors.

## Stage-2 camera convention

Camera calculations use right-handed XYZ coordinates in metres. They reproduce
the Stage-2 projection producer's `get_local_params(...,
coord_system_change=True)` convention rather than inferring a convention from
the DeepCA paper (which does not specify angle signs, handedness, or array axes).

For each `(theta_deg, phi_deg)`, let `C` be

```text
[[ 0, -1,  0],
 [ 1,  0,  0],
 [ 0,  0, -1]]
```

The exact native rotations and basis change are:

```text
R_theta = [[ cos(theta), -sin(theta), 0],
           [ sin(theta),  cos(theta), 0],
           [          0,           0, 1]]

R_phi   = [[1,         0,        0],
           [0,  cos(phi), sin(phi)],
           [0, -sin(phi), cos(phi)]]

R = C @ R_theta @ C.T @ C @ R_phi @ C.T
```

With the resulting rotation `R`:

- central ray direction is `R @ [0,0,1]`;
- source is `-direction * SOD`;
- detector centre is `direction * (SID - SOD)`;
- detector columns advance along normalized `R @ [0,1,0]`;
- detector rows advance along normalized `R @ [-1,0,0]`, accounting for the
  producer's final vertical image flip.

The optional projection centre offset does not alter those camera axes. It
defines the physical centre of the cubic reconstruction grid. If absent, the
grid centre falls back to the ground-truth physical centre
`(shape_xyz - 1) * spacing_xyz / 2`.

## Reconstruction grid and upstream deviation

With detector mode selected, the cubic field of view at isocentre is

```text
FOV_mm = min(H, W) * detector_pixel_spacing_mm * SOD / SID
```

For a configured side length `N`, model spacing is isotropic `FOV_mm / N`. The
grid origin is `centre_xyz_mm - (N - 1) * spacing_xyz_mm / 2`. Ground truth is
nearest-neighbour sampled from its identity-oriented XYZ grid onto that physical
grid and emitted in ZYX order.

The adaptation intentionally does **not** invoke the authors' data generator or
TIGRE. Instead, it uses deterministic voxel-driven binary cone-support
backprojection:

1. threshold each selected detector image at the configured projection
   threshold (default `> 0`);
2. project every model-grid voxel onto each detector with the Stage-2 camera;
3. sample the binary detector support with configured nearest or bilinear
   interpolation;
4. binarize each sampled per-view support at `> 0`; and
5. aggregate the supports.

This preserves the released representation's geometric cone support and its
default two-view `{0,1,2}` sum, but it is not numerically equivalent to upstream
one-iteration TIGRE SIRT. It is the central reproduction deviation required to
consume the existing projection NPZ files without regenerating data. Cache keys
include source file size/mtime, selected views, calibration, physical grid, and
preprocessing settings, target keys/axis semantics, and an algorithm/schema
version.

Two views with `combine: sum` are the paper-compatible default. Using one view
or more than two views is supported for controlled comparison, but raw summation
changes the input range to `{0,...,V}` and is a scientific deviation. `mean` and
`union` aggregation are also explicit deviations and must not be mixed within an
experiment. Selected view indices and calibration are recorded per case.

## Fixed two-view translational calibration robustness evaluation

`evaluation.mode: fixed_two_view_translation` is a **fixed two-view
translational calibration robustness evaluation**. It is not an angle-robustness
experiment: every control and perturbed condition has
`delta_theta = 0 degrees` and `delta_phi = 0 degrees`. It is also specifically a
**fixed-positive-direction translational stress test**, not evidence of
direction-independent translation robustness.

The evaluation locks the case set and the same ordered pair of source views for
the accurate control and all nine perturbations. In that pair:

- input view 1 (selected-input position 0 in code) is the stored, geometrically
  accurate projection and is never re-rendered;
- input view 2 (selected-input position 1 in code) is re-rendered after
  translating the projection-centred 3D artery;
- both nominal `theta_deg` and `phi_deg` values and their view-direction
  encodings remain unchanged;
- the target vessel, model checkpoint, model weights, and source dataset files
  remain unchanged; and
- every projection-derived model input is recomputed after replacing view 2.
  For DeepCA this means rebuilding the two-view cone-support backprojection;
  DeepCA has no separate learned 2D image-feature extractor.

The configured magnitude `d` is the total Euclidean displacement, not a
per-active-axis displacement. The evaluator's plan is fixed to the following
nine conditions plus one accurate two-view control:

| Condition | delta theta | delta phi | delta x (mm) | delta y (mm) | delta z (mm) | Total |
|---|---:|---:|---:|---:|---:|---:|
| `Y-5` | 0 degrees | 0 degrees | 0 | 5 | 0 | 5 mm |
| `Y-10` | 0 degrees | 0 degrees | 0 | 10 | 0 | 10 mm |
| `Y-20` | 0 degrees | 0 degrees | 0 | 20 | 0 | 20 mm |
| `XZ-5` | 0 degrees | 0 degrees | 3.5355 | 0 | 3.5355 | 5 mm |
| `XZ-10` | 0 degrees | 0 degrees | 7.0711 | 0 | 7.0711 | 10 mm |
| `XZ-20` | 0 degrees | 0 degrees | 14.1421 | 0 | 14.1421 | 20 mm |
| `XYZ-5` | 0 degrees | 0 degrees | 2.8868 | 2.8868 | 2.8868 | 5 mm |
| `XYZ-10` | 0 degrees | 0 degrees | 5.7735 | 5.7735 | 5.7735 | 10 mm |
| `XYZ-20` | 0 degrees | 0 degrees | 11.5470 | 11.5470 | 11.5470 | 20 mm |

The displayed components are rounded to four decimal places. The evaluator
computes and records them from the formulas below at full floating-point
precision, so each vector's norm is exactly its configured `d` up to numerical
precision:

```text
delta_t_Y   = (0, d, 0)
delta_t_XZ  = d / sqrt(2) * (1, 0, 1)
delta_t_XYZ = d / sqrt(3) * (1, 1, 1)
```

### Coordinates, centring, and sign convention

The vectors use the current vessel-code patient convention, before conversion
to DeepCA's `[C,Z,Y,X]` tensor layout:

- `+x`: patient left;
- `+y`: patient anterior, away from the table; and
- `+z`: patient superior, toward the head.

Only those positive directions are sampled. A codebase with a different
patient coordinate system must transform these physical vectors into its own
coordinates before rendering; relabelling components without a coordinate
transform changes the experiment.

Let `C` be the original artery in metres and `o_centre` the exact centring
offset used to create the stored accurate projections. The renderer uses

```text
C^(0)       = C - o_centre
C_input_1   = C^(0)
C_input_2   = C^(0) + 1e-3 * delta_t_2
```

where `delta_t_2` is one of the millimetre vectors above and `1e-3` converts
millimetres to metres. The recorded vector therefore means **translate the
projection-centred artery by `+delta_t_2`**. This is geometrically equivalent
to translating the source-detector system or isocentre by `-delta_t_2`; results
must not mix those two sign conventions. The evaluation output records both the
artery translation and its equivalent inverse system translation.

The archive's `projection_center_offset` is mandatory and authoritative in this
mode; the evaluator never substitutes a target-volume centre or a newly
estimated offset. It independently regenerates the tube surface and recomputes
the expected offset as an audit. The run fails if that audit differs from the
recorded offset by more than `0.01 mm`, before the recorded offset is applied to
both the surface and centreline.

For view 2, the renderer reuses the original theta, phi, SID, detector spacing,
image size, branch subset, centring rule, mask-render mode, and all other render
settings. It replaces only that second image, keeps both nominal camera
encodings unchanged, rebuilds the projection-derived DeepCA input, and compares
the generator prediction with the original canonical target.

### Mandatory controls and cache policy

Before any non-zero condition for a case, view 2 is re-rendered with a zero
translation. Its binary mask must achieve Dice at least `0.98` against the
stored second-view image. Failure is a hard validity error: it indicates that
the renderer, projected branch subset, artery centring, or stored calibration
does not reproduce the original evidence, so a translated result would not be
a controlled perturbation. This gate is not a model-performance metric.
The threshold cannot be configured below `0.98`, and both the stored image and
clean re-render must contain foreground vessel pixels; an empty/empty Dice of
one is explicitly rejected.

Projection-dependent caching is disabled in both checked-in translation
templates. Reusing the ordinary accurate-image cache would silently erase the
intervention. Another implementation may enable a cache only if its key includes
the full resolved translation vector (including zero), selected ordered view
indices, renderer settings/version, branch subset, centring offset, calibration,
and all existing preprocessing fields. Regardless of cache implementation, the
second image and all image-derived features or backprojections must be
recomputed for each condition.

The templates also fix `renderer_num_circle_points: 120`, warn when translated
centreline visibility is below `0.95`, and retain such cases by default
(`fail_below_visibility_threshold: false`). Visibility is an interpretation
aid, not a substitute target or post-hoc inclusion rule.

The supplied Stage-2 all-branch cohorts predate per-file
`projected_branch_indices` in some exports, so the templates explicitly declare
`missing_projected_branch_policy: all`. This is a cohort provenance assertion,
not a generic fallback. Other datasets must store their exact projected branch
indices or change neither this assertion nor the branch subset without rebuilding
and validating their provenance. Output records whether branch indices,
rendering mode, circle-point count, SID, and detector spacing came from the
archive or an explicit configured cohort value.

### Required records and interpretation

For the accurate control and every translated condition, retain enough
per-case data to reconstruct the comparison:

- condition identifier, full-precision artery translation vector, inverse
  source-detector/isocentre vector, and total Euclidean magnitude;
- both ordered source-view indices and their unchanged theta and phi;
- zero-translation view-2 re-render Dice and the branch subset and centring
  provenance used by the renderer;
- visible centreline-point fraction and visible vessel-surface-point fraction;
- stored and translated view-2 foreground-pixel counts or ratios;
- DeepCA generator metrics for the accurate control and perturbed condition;
  and
- absolute and relative metric changes from the accurate two-view control.

The visible fractions must measure the proportion of the translated 3D
centreline points and rendered surface samples whose projections remain inside
the detector. Inspect them together with foreground-pixel changes, especially at
20 mm: detector clipping is a loss of input evidence and must not be reported as
intrinsic model sensitivity alone.

DeepCA has one inference-time generator. Its critic is a training-only
adversarial component, not a coarse model or refiner, and this repository does
not run a separate refinement stage. When comparing with a result schema that
expects coarse and refined predictions, the DeepCA generator may be identified
as the sole/coarse prediction, while refined metrics and
`refined-minus-coarse` must be recorded as not applicable rather than duplicated
or invented.

The LCA and RCA templates inherit their normal evaluation configs, require
exactly two views, disable the data cache, and write to cohort-specific output
roots distinct from ordinary evaluation:

```bash
python evaluate.py \
  --config configs/eval_imagecas_lca_fixed_translation.yaml --split test
python evaluate.py \
  --config configs/eval_imagecas_rca_fixed_translation.yaml --split test
```

A successful fixed-translation evaluation publishes one immutable, atomically
renamed bundle beneath `<output_dir>/<split>/runs/run-.../`; the CLI reports its
`result_directory`. That bundle contains `per_case.json`, `per_case.csv`,
`summary.json`, `failures.json`, and, when enabled, its condition-aware
`predictions/` tree. A failed or interrupted matrix is never promoted as a
completed run. The bundle context records the resolved config fingerprint and
whether `--allow-checkpoint-config-mismatch` authorized the checkpoint. Every
case/condition records Dice, clDice, its broader elapsed time, and a separate
CUDA-synchronized generator-only `inference_seconds`; summaries aggregate that
inference time overall and by condition.

## Missing upstream components and preserved quirks

The release provides no NPZ loader, split-JSON parser, standalone inference or
evaluation script, physical-grid alignment, Dice/clDice, real-ICA preprocessing
implementation, saved camera metadata, LCA source loader, or released
checkpoint. The tracked LAD loader exists only as stale Python bytecode. LCA
training/evaluation in this adaptation is therefore an extension beyond the
paper's released RCA training path.

For fidelity, the adaptation leaves checkpoint-defining model behavior intact:

- the CCT tokenizer has asymmetric third-axis padding;
- latent tokens are reshaped with the released `view` rather than transposed;
- allocated positional embeddings remain inactive;
- the generator has no output activation;
- the critic ends with a patchwise `Tanh`;
- dynamic-snake code retains the released implementation.

The training integration corrects the released gradient-penalty reduction to a
per-sample norm over all conditional-pair elements and averages critic patches
before differentiating. This is an explicit correction toward the published
WGAN-GP objective, not byte-for-byte reproduction of the released training bug.

## Linux environment setup

Run these commands from the repository root on the Linux server. Do not use
`sudo`, and do not modify the existing `vesseltree` environment.

First inspect the server driver and load the available Python 3.12 module:

```bash
module purge
module load python3
nvidia-smi
python3 --version
```

The verifier expects the server's reported Python `3.12.10` exactly.

```bash
python3 -m venv /export/home2/reny0012/vir_env/deepca-imagecas-py312
source /export/home2/reny0012/vir_env/deepca-imagecas-py312/bin/activate
python -m pip install --upgrade pip==24.3.1 setuptools==75.8.0 wheel==0.45.1
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1
python -m pip install -r requirements.txt
python -m pip check
```

This is a runtime migration from the authors' Python 3.9/PyTorch 2.1 stack,
not a claim of bit-for-bit dependency equivalence. The model, losses, and data
contract are unchanged; NumPy remains on the 1.x line to avoid an unnecessary
NumPy 2 API/ABI transition. Small floating-point differences from the newer
PyTorch and CUDA runtime are still possible.

There is no repository CUDA extension to build. The adapted backprojector is
NumPy code and TIGRE is not required. The `cu124` wheel bundles a CUDA 12.4
runtime; an NVIDIA driver advertising CUDA 12.5 or newer should be backward
compatible, but that must be confirmed on the actual host with `nvidia-smi` and
the verification below rather than assumed from the advertised version alone.

```bash
which python
python scripts/verify_environment.py --forward
python -m unittest discover -s tests -v
```

The verifier prints Python, package, PyTorch, bundled CUDA runtime, cuDNN, GPU,
and driver information; imports the core model and scikit-image; constructs a
small `32^3`, eight-base-filter generator plus critic; and, with `--forward`,
runs only the small generator. It does not train. For a CPU-only import check,
use `python scripts/verify_environment.py --device cpu --forward`; CUDA is still
reported and the pinned `cu124` build is still required, but the forward pass
does not require an available GPU.

After the environment has passed on the server, capture the resolved environment
without replacing the curated requirements file:

```bash
python -m pip freeze --all > /export/home2/reny0012/vir_env/deepca-imagecas-py312/pip-freeze.txt
```

## Training, resume, and evaluation

The LCA and RCA cohorts must be trained separately. The checked-in YAML files
hold dataset paths, fallbacks, view policy, output directory, model size, and
optimizer settings.

```bash
python train.py --config configs/imagecas_lca.yaml
python train.py --config configs/imagecas_rca.yaml
```

Each run writes `resolved_config.yaml`, `resolved_cases.json`, metadata,
machine-readable `epochs.jsonl`, and `checkpoints/last.pt`; validation
improvements also write `checkpoints/best.pt`. The configured roots are
`/export/home2/reny0012/result/deepca_imagecas/lca` and
`/export/home2/reny0012/result/deepca_imagecas/rca`. Resume only from the rich
adapted checkpoint, using the same config and resolved cases:

```bash
python train.py --config configs/imagecas_lca.yaml \
  --resume /export/home2/reny0012/result/deepca_imagecas/lca/checkpoints/last.pt
python train.py --config configs/imagecas_rca.yaml \
  --resume /export/home2/reny0012/result/deepca_imagecas/rca/checkpoints/last.pt
```

Legacy upstream `.tar` files can provide `network` weights for evaluation, but
cannot safely resume because they omit epoch, split, RNG, scheduler, scaler,
configuration, and case-manifest state. The adapted trainer deliberately
accepts only its rich checkpoint schema for resume; importing legacy weights for
a new training run would require an explicit conversion or warm-start path.
Resume also compares the saved resolved configuration and rejects changes to
data, calibration, preprocessing, model, losses, batch/AMP, optimizers, or the
scheduler. Only documented runtime fields such as device, worker count, epoch
limit, resume path, run name, and output directory may change.

Evaluate without training-time random view selection:

```bash
python evaluate.py --config configs/eval_imagecas_lca.yaml --split test
python evaluate.py --config configs/eval_imagecas_rca.yaml --split test
```

Rich checkpoints are checked against the configured model, vessel cohort,
calibration, preprocessing, ground-truth convention, and split manifest. A
legacy checkpoint has no such provenance and therefore requires an explicit,
recorded override:

```bash
python evaluate.py --config configs/eval_imagecas_rca.yaml --split test \
  --checkpoint /path/to/upstream_checkpoint.tar \
  --allow-checkpoint-config-mismatch
```

Use that override only after manually establishing that the weights and current
configuration are compatible. The evaluator records the override in its output
context.

To evaluate several view counts under one fixed checkpoint and threshold, use a
single comma-separated override; results remain grouped by view count:

```bash
python evaluate.py --config configs/eval_imagecas_rca.yaml --split test \
  --view-counts 1,2,3,4,5,6,7 --threshold 0.5
```

Use that seven-count sweep only for a split whose cases all contain seven views;
requesting more views than a case contains is recorded as a case failure, not
silently reduced to the available count.

The evaluator applies one raw-output threshold to all cases, verifies shape,
spacing, and origin alignment, and writes per-case JSON/CSV, aggregate statistics,
view-count groups, optional binary NPZ predictions, and an explicit failure
manifest beneath the configured `evaluation/test` directory. Both-empty
Dice/clDice is 1; exactly-one-empty is 0. clDice uses scikit-image 0.22.0's
deterministic Lee 3D skeletonization. Each case records Dice, clDice, its broader
elapsed time, and CUDA-synchronized generator-only `inference_seconds`; the
summary aggregates inference time overall and by cohort/view-count grouping.
Input transfer, post-processing, metrics, and serialization are excluded from
`inference_seconds`. To exercise evaluation on validation data without writing
predicted volumes after a checkpoint exists:

```bash
python evaluate.py --config configs/eval_imagecas_lca.yaml --split val \
  --no-save-predictions
python evaluate.py --config configs/eval_imagecas_rca.yaml --split val \
  --no-save-predictions
```

### Small real-data loader and forward smoke test

The checked-in smoke utility reads one real projection/GT pair through the same
loader and geometry code, temporarily reduces the grid to `32^3` and the
generator base width to eight, disables the cache, and performs one inference
forward pass. It does not train or write a checkpoint.

```bash
python scripts/smoke_test.py \
  --config configs/imagecas_lca.yaml \
  --projection /dataset/reny0012/vessel_code_stage_2_lca_paired/anchors/lca_0001.npz \
  --ground-truth /dataset/reny0012/imagecas_voxel/lca/1.npz \
  --device cuda:0

python scripts/smoke_test.py \
  --config configs/imagecas_rca.yaml \
  --projection /dataset/reny0012/imagecas_autocar_6/stage_2_imagecas_all_branch/rca_0001.npz \
  --ground-truth /dataset/reny0012/imagecas_voxel/rca/1.npz \
  --device cuda:0
```

The output must report matching `[1,1,32,32,32]` input, target, and prediction
shapes, finite predictions, the selected view indices, and a non-negative target
foreground count. This lightweight check changes the architecture solely to
bound memory and is not a scientific experiment.
