#!/usr/bin/env python
# ------------------------------------------------------------
#  pkl2vtk.py  –  convert <plate_rollouts>.pkl  ->  VTK files
#                (works for the trajectory dictionaries that
#                 deform_eval.evaluate() writes)
# ------------------------------------------------------------
import pickle
import pathlib
import numpy as np
import meshio                     

PKL_PATH   = "out/plate_rollouts.pkl"          # <— change if needed
OUT_DIR    = pathlib.Path("vtk_out")           # where .vtk files end up
WRITE_GT   = True                              # also write ground-truth?

# ------------------------------------------------------------
def write_timestep(points, cells, fname, point_data=None):
    """
    points : (N,3) float32
    cells  : (M,3) (triangles)  or  (M,4) (tetrahedra)
             already zero-based indices into `points`
    """
    pts     = points.astype(np.float32)
    # meshio needs a list of (cell_type, array)
    cell_t  = 'triangle' if cells.shape[1] == 3 else 'tetra'
    cells_m = [(cell_t, cells.astype(np.int32))]
    meshio.write_points_cells(fname, pts, cells_m,
                              point_data=point_data or {})

# ------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    with open(PKL_PATH, "rb") as fh:
        rollouts = pickle.load(fh)           # list of dictionaries

    for r_idx, R in enumerate(rollouts):
        faces      = R["faces"]              # (F,3) – surface triangles
        mesh_pos   = R["mesh_pos"]           # (N,3) – reference mesh coords
        pred_traj  = R["pred_pos"]           # (T,N,3)
        gt_traj    = R["gt_pos"]             # (T,N,3)

        for t, xyz in enumerate(pred_traj):
            fname        = OUT_DIR / f"roll{r_idx:02d}_step{t:04d}.vtk"
            err          = np.linalg.norm(xyz - gt_traj[t], axis=1)
            pdata        = {"pred_err": err}
            write_timestep(xyz, faces, fname, point_data=pdata)

            if WRITE_GT:
                fname_gt = OUT_DIR / f"roll{r_idx:02d}_step{t:04d}_gt.vtk"
                write_timestep(gt_traj[t], faces, fname_gt)

        print(f"rollout {r_idx}: wrote {len(pred_traj)} timesteps")

if __name__ == "__main__":
    main()
