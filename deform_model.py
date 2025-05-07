import sonnet as snt
import tensorflow as tf

import common
import core_model
import normalization


class DeformingPlateModel(snt.Module):
    """Graph Neural Network model for simulating a deforming plate."""

    def __init__(self, learned_model, name="model"):
        super().__init__(name=name)

        self._learned_model = learned_model

        # -- Feature normalisers -------------------------------------------------
        self._node_normaliser  = normalization.Normalizer(
            size=common.NodeType.SIZE + 3 + 3, name="node_norm")
        self._edge_normaliser  = normalization.Normalizer(size=8, name="edge_norm")

        #     *separate* per-field output normalisers
        self._vel_norm    = normalization.Normalizer(size=3, name="vel_norm",
                                                     std_epsilon=1e-4)
        self._stress_norm = normalization.Normalizer(size=1, name="stress_norm",
                                                     std_epsilon=1e-4)
        # ------------------------------------------------------------------------

    # -------------------------------------------------------------------------- #
    #  Graph construction helper
    # -------------------------------------------------------------------------- #
    def _build_graph(self, inputs, is_training):
        cur_pos    = inputs["world_pos"]            # (N,3)
        target_pos = inputs["target|world_pos"]     # (N,3)
        node_type  = tf.squeeze(inputs["node_type"], -1)

        # prescribed motion for HANDLE nodes
        prescribed_motion = tf.where(
            tf.equal(node_type, common.NodeType.HANDLE)[..., None],
            target_pos - cur_pos,
            tf.zeros_like(cur_pos))

        node_type_oh = tf.one_hot(node_type, depth=common.NodeType.SIZE)  # (N,9)
        node_feats   = tf.concat([cur_pos, prescribed_motion, node_type_oh], -1)
        node_feats   = self._node_normaliser(node_feats, accumulate=is_training)

        # edges -------------------------------------------------------------
        cells               = inputs["cells"]               # (F,4)
        senders, receivers  = common.tetrahedra_to_edges(cells)

        rel_world = tf.gather(cur_pos, senders) - tf.gather(cur_pos, receivers)
        rel_mesh  = tf.gather(inputs["mesh_pos"], senders) - \
                    tf.gather(inputs["mesh_pos"], receivers)
        edge_feats = tf.concat(
            [rel_world,
             tf.norm(rel_world, axis=-1, keepdims=True),
             rel_mesh,
             tf.norm(rel_mesh, axis=-1, keepdims=True)], -1)
        edge_feats = self._edge_normaliser(edge_feats, accumulate=is_training)

        mesh_edges = core_model.EdgeSet("mesh_edges", edge_feats, senders, receivers)
        return core_model.MultiGraph(node_feats, [mesh_edges])

    # -------------------------------------------------------------------------- #
    #  Loss
    # -------------------------------------------------------------------------- #
    def loss(self, inputs):
        g        = self._build_graph(inputs, is_training=True)
        net_out  = self._learned_model(g)                       # (N,4) (vel|stress)

        # ------------ ground truth with √-stress scaling ------------------------
        cur_pos  = inputs["world_pos"]
        tgt_pos  = inputs["target|world_pos"]
        tgt_vel  = tgt_pos - cur_pos                            # (N,3)

        raw_stress = tf.clip_by_value(inputs["stress"], -1e6, 1e6)  # avoid Inf
        tgt_stress = tf.sign(raw_stress) * tf.sqrt(tf.abs(raw_stress) + 1e-6)  # (N,1)

        vel_norm    = self._vel_norm(tgt_vel,     accumulate=True)   # (N,3)
        stress_norm = self._stress_norm(tgt_stress, accumulate=True) # (N,1)
        tgt_normed  = tf.concat([vel_norm, stress_norm], -1)         # (N,4)

        # ------------ masks & per-field losses -------------------------------
        node_type = tf.squeeze(inputs["node_type"], -1)
        mask      = tf.equal(node_type, common.NodeType.NORMAL)

        vel_err   = tf.reduce_sum(tf.square(tgt_normed[:, :3] - net_out[:, :3]), -1)
        vel_loss  = tf.reduce_mean(tf.boolean_mask(vel_err, mask))

        stress_err  = tf.square(tgt_normed[:, 3] - net_out[:, 3])
        stress_loss = tf.reduce_mean(tf.boolean_mask(stress_err, mask))
        self.last_stress_loss = stress_loss

        return vel_loss + 0.1 * stress_loss

    # -------------------------------------------------------------------------- #
    #  Util: update positions
    # -------------------------------------------------------------------------- #
    def _update(self, inputs, net_out):
        vel      = self._vel_norm.inverse(net_out[:, :3])        # (N,3)
        next_pos = inputs["world_pos"] + vel
        return next_pos

    # -------------------------------------------------------------------------- #
    #  Inference / rollout call
    # -------------------------------------------------------------------------- #
    def __call__(self, inputs, training=False):
        g        = self._build_graph(inputs, is_training=training)
        net_out  = self._learned_model(g)                        # (N,4)

        vel     = self._vel_norm.inverse(net_out[:, :3])         # (N,3)

        # undo √-scaling:  sign(x)*x²
        stress_sqrt = self._stress_norm.inverse(net_out[:, 3:])  # (N,1)
        stress      = tf.sign(stress_sqrt) * tf.square(stress_sqrt)

        return {
            "world_pos"    : self._update(inputs, net_out),  # (N,3)
            "cur_position" : inputs["world_pos"],
            "cur_velocity" : vel,
            "stress"       : stress
        }
