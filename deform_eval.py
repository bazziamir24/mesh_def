import tensorflow as tf
import numpy as np
import pickle
from common import NodeType

# Rollout Simulation Function
def _rollout(model, initial_state, num_steps, gt_world_pos):
    """
    Autoregressively simulates the system forward using the model.

    Parameters
    ----------
    model : callable
        The trained model used to predict future states.
    initial_state : dict
        Dictionary containing simulation inputs at t=0.
    num_steps : int
        Number of timesteps to simulate.
    gt_world_pos : tf.Tensor
        Ground-truth positions with shape (T, N, 3).

    Returns
    -------
    prediction : tf.Tensor
        Predicted trajectory, shape (T, N, 3).
    zeros (3x) : tf.Tensor
        Placeholder tensors for compatibility with downstream tools.
    """

    # Identify which nodes can move: NORMAL or OUTFLOW
    node_type = tf.squeeze(initial_state['node_type'], -1)  # (N,)
    movable = tf.logical_or(
        node_type == NodeType.NORMAL.value,
        node_type == NodeType.OUTFLOW.value)                # (N,)
    mask = tf.expand_dims(movable, -1)                      # (N, 1)

    # Extract static (time-invariant) fields from input
    static_inputs = {
        k: v for k, v in initial_state.items()
        if k not in ('world_pos', 'target|world_pos')
    }

    # Initialize with t=0 positions
    cur_pos = initial_state['world_pos']                    # (N, 3)
    trajectory = []

    for t in tf.range(num_steps):
        # Prepare model input for current step
        model_input = {
            **static_inputs,
            'world_pos': cur_pos,
            'target|world_pos': gt_world_pos[t]  # Supervision for HANDLE nodes
        }


        # Model prediction
        outputs = model(model_input, training=False)
        pred_pos = outputs['world_pos']                     # (N, 3)

        # Apply ground-truth override to non-movable nodes
        next_pos = tf.where(mask, pred_pos, gt_world_pos[t])
        trajectory.append(next_pos)

        # Stop gradient propagation (manual unrolling)
        cur_pos = tf.stop_gradient(next_pos)

        if t % 50 == 0:       # every 50 steps
            err = tf.reduce_mean(tf.square(pred_pos - gt_world_pos))
            tf.print(f"[rollout] step", t, "mean|z| =", tf.reduce_mean(cur_pos[:,2]),
                    "Δpos L2 =", err)

    # Stack predictions into full trajectory
    prediction = tf.stack(trajectory, axis=0)               # (T, N, 3)

    # Return dummy tensors to match expected signature
    zeros = tf.zeros_like(prediction)
    return prediction, zeros, zeros, zeros


# Single-Trajectory Evaluation
def evaluate(model, batch):
    """
    Evaluates a single trajectory using MSE and L1 losses.

    Parameters
    ----------
    model : callable
        The trained model for inference.
    batch : dict
        Contains trajectory tensors (T, N, F) without batch dimension.

    Returns
    -------
    scalars : dict
        Dictionary containing MSE and L1 loss metrics.
    traj_ops : dict
        Dictionary containing mesh faces and positions for visualization.
    """

    # Ground-truth target trajectory
    gt_world_pos = batch['target|world_pos']     # (T, N, 3)
    num_steps = tf.shape(gt_world_pos)[0]

    # Extract t=0 frame as initial state
    initial_state = {k: v[0] for k, v in batch.items()}

    # Run autoregressive rollout
    pred_pos, _, _, _ = _rollout(model, initial_state, num_steps, gt_world_pos)

    # Compute error metrics
    mse = tf.reduce_mean(tf.square(pred_pos - gt_world_pos))
    l1  = tf.reduce_mean(tf.abs(pred_pos - gt_world_pos))

    scalars = {
        'mse_world_pos': float(mse.numpy()),
        'l1_world_pos':  float(l1.numpy())
    }

    # Prepare triangle mesh edges for visualization (VTK/3D viewers)
    faces = batch['cells'][0]                             # (F, 4)
    face_start = faces[:, 0:3]                            # First 3 vertices
    face_end = tf.concat([faces[:, 2:], faces[:, :1]], axis=-1)
    wire_edges = tf.concat([face_start, face_end], axis=0)  # (2F, 3)

    traj_ops = {
        'faces': wire_edges.numpy(),
        'mesh_pos': batch['mesh_pos'][0].numpy(),
        'gt_pos': batch['world_pos'][0].numpy(),
        'pred_pos': pred_pos.numpy()
    }

    return scalars, traj_ops
