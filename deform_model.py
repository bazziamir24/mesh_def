import sonnet as snt 
import tensorflow as tf

import common
import core_model
import normalization


class DeformingPlateModel(snt.Module):
    """Model for the Deforming Plate simulation."""

    def __init__(self, learned_model, name="Model"):
        super().__init__(name=name)
        self._learned_model = learned_model
        self._output_normalizer = normalization.Normalizer(
            size=4, name="output_normalizer")  # 3D velocity + 1 stress
        self._node_normalizer = normalization.Normalizer(
            size=common.NodeType.SIZE + 3 + 3, name="node_normalizer")  # One-hot node types
        self._edge_normalizer = normalization.Normalizer(
            size=8, name="edge_normalizer")  # 3D mesh-space + 3D world-space + 2 norms

    def _build_graph(self, inputs, is_training):
        """Constructs the input MultiGraph from batched inputs."""

        cur_pos = inputs['world_pos']             # shape: [B, N, 3]
        target_pos = inputs['target|world_pos']   # shape: [B, N, 3]
        node_type = tf.squeeze(inputs['node_type'], axis=-1)  # shape: [B, N]

        # === 1. Prescribed motion (only for HANDLE nodes) ===
        prescribed_motion = tf.where(
            tf.equal(node_type, common.NodeType.HANDLE)[..., tf.newaxis],  # [B, N, 1]
            target_pos - cur_pos,
            tf.zeros_like(cur_pos)
        )  # shape: [B, N, 3]

        # === 2. One-hot node type ===
        node_type_onehot = tf.one_hot(node_type, depth=common.NodeType.SIZE)  # [B, N, 9]

        # === 3. Node features ===
        node_features = tf.concat([cur_pos, prescribed_motion, node_type_onehot], axis=-1)  # [B, N, 15]
        print("node_features.shape", node_features.shape)
        node_features = self._node_normalizer(node_features, is_training)
        print("node_features (normalized) shape:", node_features.shape)

        # === 4. Edge indices (shared across batch) ===
        tf.print("inputs['cells'] shape before tetrahedra_to_edges:", tf.shape(inputs["cells"]))
        senders, receivers = common.tetrahedra_to_edges(inputs['cells'])  # [E], [E]
        print("Cells shape:", inputs['cells'].shape)
        print("Senders shape:", senders.shape)

        # === 5. Edge features ===
        # cur_pos and mesh_pos are [N_tot, 3]  → gather along axis 0 (default)
        relative_world_pos = tf.gather(cur_pos, senders) - tf.gather(cur_pos, receivers)      # [E, 3]
        relative_mesh_pos  = (
            tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers) # [E, 3]
        )


        edge_features = tf.concat([
            relative_world_pos,
            tf.norm(relative_world_pos, axis=-1, keepdims=True),
            relative_mesh_pos,
            tf.norm(relative_mesh_pos, axis=-1, keepdims=True)
        ], axis=-1)  # [B, E, 8]

        edge_features = self._edge_normalizer(edge_features, is_training)

        # === 6. Edge set ===
        mesh_edges = core_model.EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receivers,
            senders=senders
        )

        # === 7. Final graph ===
        return core_model.MultiGraph(
            node_features=node_features,
            edge_sets=[mesh_edges]
        )

        
        
    def loss(self, inputs):
        """Computes L2 loss on velocity only (for training), also calculates stress loss internally for diagnostics."""
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph)  # shape: [num_nodes, 4] -> (vx, vy, vz, stress)

        # --- Targets ---
        cur_position = inputs['world_pos']
        target_position = inputs['target|world_pos']
        target_velocity = target_position - cur_position  # [num_nodes, 3]
        target_stress = inputs['stress']                 # [num_nodes, 1]

        # Normalize combined target
        target = tf.concat([target_velocity, target_stress], axis=-1)  # [num_nodes, 4]
        target_normalized = self._output_normalizer(target, accumulate=True)

        # Predicted output
        predicted_output = network_output[:, :4]  # [vx, vy, vz, stress]

        # Mask NORMAL nodes
        node_type = tf.squeeze(inputs['node_type'], axis=-1)
        loss_mask = tf.equal(node_type, common.NodeType.NORMAL)

        # Velocity loss (used for training)
        velocity_error = tf.reduce_sum(tf.square(
            target_normalized[:, :3] - predicted_output[:, :3]
        ), axis=-1)
        velocity_loss = tf.reduce_mean(tf.boolean_mask(velocity_error, loss_mask))

        # Stress loss (for diagnostics only)
        stress_error = tf.square(target_normalized[:, 3] - predicted_output[:, 3])
        stress_loss = tf.reduce_mean(tf.boolean_mask(stress_error, loss_mask))

        # Store stress loss for optional logging later
        self.last_stress_loss = stress_loss
        
        print("Sample target_velocity (pre-normalization):", target_velocity[:5])
        print("Sample target_stress (pre-normalization):", target_stress[:5])
        print("Sample normalized velocity:", target_normalized[:5, :3])
        print("Sample normalized stress:", target_normalized[:5, 3])


        return velocity_loss  # Only this is used for training



    def _update(self, inputs, per_node_network_output):
        """Applies predicted velocity to update world position."""
        velocity = self._output_normalizer.inverse(per_node_network_output)
        cur_position = inputs['world_pos']
        next_position = cur_position + velocity[:, :, :3]  # Only the velocity part
        return next_position

 # def loss(self, inputs):
    #     """Computes L2 loss on predicted velocity and stress (DeformingPlate)."""
    #     graph = self._build_graph(inputs, is_training=True)
    #     network_output = self._learned_model(graph)  # shape: [num_nodes, 4]

    #     # Targets: velocity and stress
    #     target_velocity = inputs['target|velocity']          # [num_nodes, 3]
    #     target_stress = inputs['stress']                     # [num_nodes, 1]
    #     target = tf.concat([target_velocity, target_stress], axis=-1)  # [num_nodes, 4]

    #     # Normalize the target
    #     target_normalized = self._output_normalizer(target)

    #     # Compute loss on NORMAL nodes
    #     node_type = inputs['node_type'][:, 0]
    #     loss_mask = tf.equal(node_type, common.NodeType.NORMAL)

    #     error = tf.reduce_sum(tf.square(target_normalized - network_output), axis=1)  # [num_nodes]
    #     loss = tf.reduce_mean(tf.boolean_mask(error, loss_mask))
    #     return loss
    
    def __call__(self, inputs, training=False):
        graph = self._build_graph(inputs, is_training=training)
        outputs = self._learned_model(graph)

        velocity_stress = self._output_normalizer.inverse(tf.expand_dims(outputs, axis=0))
        return {
            "world_pos": self._update(inputs, velocity_stress),
            "cur_position": tf.expand_dims(inputs['world_pos'], axis=0),
            "cur_velocity": velocity_stress[:, :, :3],
            "stress": velocity_stress[:, :, 3:]
        }

