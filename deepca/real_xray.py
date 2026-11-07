"""DeepCA on real X-ray angiograms (CoronaryDominance keyframes exported by angioseg-nnunet step 08).

Same inputs, view choice, camera, centring and outputs as the PPCAR-Net `real-xray` branch (see
deepca/real_xray_common.py). DeepCA part:
  * the selected masks are back-projected with the Stage-2 camera (deepca.geometry.binary_cone_backproject) onto the
    checkpoint's cubic grid, centred at the isocentre; FOV = 256 x detector pixel x SOD / SID (= 256 x isocentre
    pixel, so the grid is the training grid also with the acquisition camera);
  * the generator (built from the checkpoint's own config) predicts the volume; raw output >= 0.5 is vessel, as in
    evaluate.py;
  * the radius-coloured surface comes from deepca.visualization.extract_vascular_geometry (as in
    scripts/visualize_prediction.py).

Usage (from the repository root):
  python -m deepca.real_xray --config configs/real_xray/lca.json \
      --input_dir /dataset/reny0012/angioseg/recon_inputs/xravos_A_v2/lca \
      --output_dir /dataset/reny0012/angioseg/recon_outputs/deepca_xravos_A_v2/lca --limit 5 --output_mode summary
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from deepca.checkpoint import safe_torch_load
from deepca.config import load_config
from deepca.geometry import (binary_cone_backproject, detector_limited_fov_mm, make_cubic_grid,
                             stage2_angles_to_camera_frames)
from deepca.inference import timed_inference
from deepca.modeling import build_generator, load_generator_state
from deepca.real_xray_common import Prediction, base_parser, config_from_args, run


class StageTwoProjector:
    """Point projection with exactly the camera of binary_cone_backproject (mm)."""

    def __init__(self, theta_deg: float, phi_deg: float, cam: dict):
        s, d, u, v = stage2_angles_to_camera_frames([theta_deg], [phi_deg], sid_m=cam["sid_mm"] / 1000.0,
                                                    source_to_isocentre_m=cam["sod_mm"] / 1000.0)
        self.source_mm, self.detector_mm = s[0] * 1000.0, d[0] * 1000.0
        self.u, self.v = u[0], v[0]
        self.sid, self.px, self.dim = float(cam["sid_mm"]), float(cam["pixel_mm"]), int(cam["image_dim"])

    def project_rc(self, points_mm: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_mm, np.float64).reshape(-1, 3)
        normal = (self.detector_mm - self.source_mm) / self.sid
        rays = pts - self.source_mm
        den = rays @ normal
        out = np.full((len(pts), 2), np.nan)
        ok = den > 0
        rel = self.source_mm + (self.sid / den[ok])[:, None] * rays[ok] - self.detector_mm
        out[ok, 0] = rel @ self.v / self.px + (self.dim - 1) / 2.0
        out[ok, 1] = rel @ self.u / self.px + (self.dim - 1) / 2.0
        return out


class DeepCABackend:
    name = "DeepCA"

    def __init__(self, config: dict):
        ckpt_path = config.get("checkpoint")
        if not ckpt_path:
            raise ValueError("set the checkpoint (config 'checkpoint' or --checkpoint)")
        dev = config.get("device") or "auto"
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        ckpt = safe_torch_load(ckpt_path, "cpu")
        model_config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else None
        self.config_source = "checkpoint"
        if model_config is None:                       # legacy weights: use the repository config given in the json
            if not config.get("model_config"):
                raise ValueError(f"{ckpt_path} has no 'config'; set model_config (e.g. configs/imagecas_lca.yaml)")
            model_config = load_config(config["model_config"])
            self.config_source = str(config["model_config"])
        vessel = str(model_config.get("data", {}).get("vessel_type", config["artery_type"])).lower()
        if vessel != config["artery_type"]:
            raise ValueError(f"checkpoint was trained on {vessel.upper()}, config is {config['artery_type'].upper()}")
        self.generator = build_generator(model_config, self.device)
        load_generator_state(self.generator, ckpt, strict=True)
        self.generator.eval()
        data = model_config.get("data", {})
        pre = data.get("preprocessing", {})
        gen = model_config.get("model", {}).get("generator", {})
        self.pre = dict(volume_size=int(pre.get("volume_size", model_config.get("model", {}).get("volume_size", 128))),
                        fov_mode=pre.get("fov_mode", "detector"), fov_mm=pre.get("fov_mm"),
                        projection_threshold=float(pre.get("projection_threshold", 0.0)),
                        interpolation=pre.get("backprojection_interpolation", "nearest"),
                        combine=pre.get("combine", "sum"), chunk_depth=int(pre.get("chunk_depth", 8)))
        self.view_fusion = str(gen.get("view_fusion", "none"))
        self.max_fusion_views = int(gen.get("max_views", 7))
        if self.view_fusion == "mean_count":
            self.min_views, self.max_views = 1, self.max_fusion_views
        else:                                          # summed input: the view count it was trained with
            n = int(data.get("views", {}).get("count", 2))
            self.min_views = self.max_views = n
        self.threshold = float(model_config.get("evaluation", {}).get("threshold", 0.5))
        self.checkpoint = str(Path(ckpt_path).resolve())

    def describe(self) -> dict:
        return dict(method=self.name, checkpoint=self.checkpoint, model_config_source=self.config_source,
                    device=str(self.device), preprocessing=self.pre, view_fusion=self.view_fusion,
                    views=[self.min_views, self.max_views], threshold=self.threshold)

    def predict(self, images, theta_deg, phi_deg, cam) -> Prediction:
        sid_m, sod_m = cam["sid_mm"] / 1000.0, cam["sod_mm"] / 1000.0
        dim = int(cam["image_dim"])
        if self.pre["fov_mode"] == "fixed":
            fov = float(self.pre["fov_mm"])
        else:
            fov = detector_limited_fov_mm((dim, dim), cam["pixel_mm"], sod_m, sid_m)
        grid = make_cubic_grid(self.pre["volume_size"], fov, (0.0, 0.0, 0.0))      # isocentre-centred
        t0 = time.perf_counter()
        stack = self.view_fusion == "mean_count"
        vol = binary_cone_backproject(np.asarray(images, np.float32), theta_deg, phi_deg, grid=grid, sid_m=sid_m,
                                      source_to_isocentre_m=sod_m, detector_pixel_spacing_mm=cam["pixel_mm"],
                                      projection_threshold=self.pre["projection_threshold"],
                                      interpolation=self.pre["interpolation"],
                                      combine="stack" if stack else self.pre["combine"],
                                      chunk_depth=self.pre["chunk_depth"])
        if stack:
            x = np.zeros((self.max_fusion_views + 1,) + vol.shape[1:], np.float32)
            x[:len(vol)] = vol
            x[self.max_fusion_views] = len(vol) / self.max_fusion_views
        else:
            x = vol[None]
        backprojection_s = time.perf_counter() - t0
        out, infer_s = timed_inference(self.generator, torch.from_numpy(x[None]).to(self.device), device=self.device)
        volume = out[0, 0].float().cpu().numpy()
        return Prediction(volume_zyx=volume, origin_center_xyz_mm=np.asarray(grid.origin_xyz_mm),
                          spacing_xyz_mm=np.asarray(grid.spacing_xyz_mm), threshold=self.threshold,
                          seconds=backprojection_s + infer_s,
                          info=dict(grid=grid.as_dict(), backprojection_seconds=backprojection_s,
                                    inference_seconds=infer_s, input_max=float(x[:len(vol)].max())))

    def projector(self, theta_deg, phi_deg, cam):
        return StageTwoProjector(theta_deg, phi_deg, cam)

    def surface(self, mask_zyx, origin_center_xyz_mm, spacing_xyz_mm):
        from deepca.metrics import PhysicalGrid
        from deepca.visualization import extract_vascular_geometry
        grid = PhysicalGrid(shape_zyx=tuple(mask_zyx.shape), spacing_xyz_mm=tuple(map(float, spacing_xyz_mm)),
                            origin_xyz_mm=tuple(map(float, origin_center_xyz_mm)))
        g = extract_vascular_geometry(mask_zyx.astype(np.float32), threshold=0.5, grid=grid)
        return g.surface_vertices_xyz_mm, g.surface_faces, g.surface_vertex_radius_mm


def main(argv=None):
    p = base_parser(__doc__)
    p.add_argument("--model_config", help="repository config for checkpoints without a stored config "
                                          "(configs/imagecas_<lca|rca>[_variable_views].yaml)")
    a = p.parse_args(argv)
    config = config_from_args(a)
    if a.model_config:
        config["model_config"] = a.model_config
    run(config, DeepCABackend(config), a.shard, a.num_shards)


if __name__ == "__main__":
    main()
