#!/usr/bin/env python3
# pkl2vtk_simple.py  –  tiniest pickle → VTK converter
import pickle
import pathlib
import numpy as np
import meshio

PKL_PATH = "out/plate_rollouts.pkl"      # <— adjust to your file
OUT_DIR  = pathlib.Path("vtk_simple")    # where VTKs will go
OUT_DIR.mkdir(exist_ok=True)

def write_step(points: np.ndarray, cells: np.ndarray, fname: pathlib.Path):
    """Write one mesh (triangles or tets) to Legacy VTK ASCII."""
    cell_type = "triangle" if cells.shape[1] == 3 else "tetra"
    meshio.write_points_cells(
        fname,
        points.astype(np.float32, copy=False),
        [(cell_type, cells.astype(np.int32, copy=False))],
    )

with open(PKL_PATH, "rb") as fh:
    rollouts = pickle.load(fh)           # list of dicts

for r_idx, R in enumerate(rollouts):
    faces     = R["faces"]               # (F,3) ints   – surface triangles
    pred_traj = R["pred_pos"]            # (T,N,3) float – predicted coords

    for t, xyz in enumerate(pred_traj):
        fname = OUT_DIR / f"roll{r_idx:02d}_step{t:04d}.vtk"
        write_step(xyz, faces, fname)

    print(f"rollout {r_idx}: wrote {len(pred_traj)} steps")

print("✅ done.")
