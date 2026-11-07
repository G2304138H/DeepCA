# DeepCA on real X-ray angiograms (CoronaryDominance)

Inference only: the trained ImageCAS DeepCA checkpoints are applied to real angiograms, as a comparison for the
PPCAR-Net `real-xray` branch. Everything except the model is shared with that branch, so the three methods see
the same cases, input views, camera and centring and write the same outputs.

## Input

One `.npz` per study x artery from angioseg-nnunet step 08 (`scripts/export_recon_inputs.py`), e.g.
`/dataset/reny0012/angioseg/recon_inputs/xravos_A_v2/lca/Study00001_LCA.npz`: every view's keyframe mask in the
256x256 model space (`images`), `theta_deg` / `phi_deg` (paper convention), the raw X-ray and segmentation at full
resolution, and the map back to the original frame (`model_to_full`).

## What happens per case

1. **Input views**: one view per viewing direction (the one with most vessel), then the `num_views` (default 2,
   as in training) with the largest smallest angle between them, the same rule as PPCAR-Net.
2. **Centring** (`input_centering`, default `mask_centroid`): every view's mask is moved to the image centre before
   inference. `bbox_center` is less pulled by a catheter segment; `none` keeps the raw C-arm position. The shift is
   folded into `model_to_full`, so all pictures are drawn on the original keyframes.
3. **Camera** (`camera`, default `acquisition`): SID / SOD of the case's X-ray system (Allura 1137 / 765 mm,
   Azurion 1195 / 810 mm; `acquisition_geometry`), detector pixel = isocentre pixel of the exported grid
   (0.5417 mm LCA / 0.4583 mm RCA) x SID / SOD. `training` uses the training camera (900 / 750 mm, 0.65 / 0.55 mm).
   Both feed the method's own Stage-2 camera code, so the back-projection uses the real geometry.
4. **Reconstruction**: the selected masks are back-projected with DeepCA's Stage-2 camera (`deepca.geometry.binary_cone_backproject`) onto the checkpoint's cubic grid, centred at the isocentre (FOV = 256 x isocentre pixel, as in training); the generator is built from the checkpoint's own config (summed two-view input, or the variable-view `mean_count` model); raw output >= 0.5 is vessel, as in `evaluate.py`. The surface and radius come from `deepca.visualization.extract_vascular_geometry`.
   Checkpoints without a stored config need `model_config` (`configs/imagecas_<art>.yaml`).
5. **Evaluation in 2D**: the predicted occupancy (its 0.5 iso-surface) is projected into every view, inputs and
   held-out, and compared with the segmentation (Dice, and Dice after the best 2D translation).

## Output (`<output_dir>/`)

- `_summary/<case>.png` and `<case>/summary.png`: one row per view (INPUT / NOT input); columns: X-ray, X-ray +
  reprojected 3D, segmentation, segmentation vs reprojection, predicted surface seen along that view, the same
  coloured by radius.
- `summary.csv`: one row per case (camera, centring shifts, input views, mean Dice for input and held-out views,
  foreground voxels, inference time, status).
- `<case>/prediction.npz` (binary volume, grid, camera), `<case>/case_summary.json`.
- `output_mode: full` adds `surface_3d.gif`, `surface_3d_radius.gif` and per-view overlays in `reprojection/`.

## Run

```bash
cd /export/home2/reny0012/code/DeepCA
# 5 LCA cases, summary PNG only
sbatch --export=ALL,ART=lca,METHOD=xravos_A_v2,LIMIT=5,OUTPUT_MODE=summary slurm/real_xray_infer.sbatch
# all RCA cases
sbatch --export=ALL,ART=rca,METHOD=smart_seg_semantic_v2,OUTPUT_MODE=summary slurm/real_xray_infer.sbatch
```

Directly: `python -m deepca.real_xray --config configs/real_xray/lca.json --checkpoint ... --input_dir ... --output_dir ...`
(`--limit`, `--case_ids`, `--output_mode`, `--camera`, `--input_centering`, `--num_views`, `--device`).

Code: `deepca/real_xray_common.py` (shared with the other methods; numpy / scipy / scikit-image / matplotlib only) and
`deepca/real_xray.py` (the DeepCA part). Test without a checkpoint: `python -m unittest tests.test_real_xray -v`.
