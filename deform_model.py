import sonnet as snt 
import tensorflow as tf

import common
import core_model
import normalization


class DeformingPlateModel(snt.Module):
    """Graph Neural Network model for simulating a deforming plate."""

    def __init__(self, learned_model, name="Model"):
        super().__init__(name=name)

        self._learned_model = learned_model

        # Normalizer for model outputs: 3D velocity + 1 stress = 4 features
        self._output_normalizer = normalization.Normalizer(size=4, name="output_normalizer")

        # Normalizer for node features: One-hot node type + 3D pos + 3D motion = 9 + 3 + 3
        self._node_normalizer = normalization.Normalizer(size=common.NodeType.SIZE + 3 + 3, name="node_normalizer")

        # Normalizer for edge features: 3D world + 3D mesh + 2 norms = 8
        self._edge_normalizer = normalization.Normalizer(size=8, name="edge_normalizer")

    def _build_graph(self, inputs, is_training):
        """
        Constructs a core_model.MultiGraph from raw input tensors.

        Parameters:
        -----------
        inputs : dict
            Contains mesh and simulation data per timestep.
        is_training : bool
            Whether in training mode (affects normalization).

        Returns:
        --------
        core_model.MultiGraph
            With node and edge features, senders, and receivers.
        """

        # === Node Features ===
        cur_pos    = inputs['world_pos']            # (N, 3)
        target_pos = inputs['target|world_pos']     # (N, 3)
        node_type  = tf.squeeze(inputs['node_type'], -1)  # (N,)

        # Compute prescribed motion only for HANDLE nodes
        prescribed_motion = tf.where(
            tf.equal(node_type, common.NodeType.HANDLE)[..., tf.newaxis],
            target_pos - cur_pos,
            tf.zeros_like(cur_pos))  # (N, 3)

        # One-hot encode node types
        node_type_onehot = tf.one_hot(node_type, depth=common.NodeType.SIZE)  # (N, 9)

        # Concatenate all node features and normalize
        node_features = tf.concat([cur_pos, prescribed_motion, node_type_onehot], axis=-1)  # (N, 15)
        node_features = self._node_normalizer(node_features, is_training)

        # === Edge Construction ===
        cells = inputs['cells']  # (F, 4)
        senders, receivers = common.tetrahedra_to_edges(cells)  # (E,), (E,)

        # === Edge Features ===
        rel_world = tf.gather(cur_pos, senders) - tf.gather(cur_pos, receivers)      # (E, 3)
        rel_mesh  = tf.gather(inputs['mesh_pos'], senders) - tf.gather(inputs['mesh_pos'], receivers)  # (E, 3)

        # Combine edge features: relative vectors + norms
        edge_features = tf.concat([
            rel_world,
            tf.norm(rel_world, axis=-1, keepdims=True),
            rel_mesh,
            tf.norm(rel_mesh, axis=-1, keepdims=True)
        ], axis=-1)  # (E, 8)

        edge_features = self._edge_normalizer(edge_features, is_training)

        # === Assemble Graph ===
        mesh_edges = core_model.EdgeSet(
            name='mesh_edges',
            features=edge_features,
            senders=senders,
            receivers=receivers)

        return core_model.MultiGraph(
            node_features=node_features,
            edge_sets=[mesh_edges])

    def loss(self, inputs):
        """
        Computes velocity loss (used for training), and logs stress loss for diagnostics.

        Parameters:
        -----------
        inputs : dict
            Contains raw mesh data, node types, stress, and target positions.

        Returns:
        --------
        tf.Tensor
            Mean squared velocity loss on NORMAL nodes.
        """
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph)  # (N, 4): velocity (vx, vy, vz) + stress

        # === Ground Truth Targets ===
        cur_position = inputs['world_pos']
        target_position = inputs['target|world_pos']
        target_velocity = target_position - cur_position  # (N, 3)
        target_stress = inputs['stress']                  # (N, 1)
        target = tf.concat([target_velocity, target_stress], axis=-1)  # (N, 4)
        target_normalized = self._output_normalizer(target, accumulate=True)

        # === Predictions ===
        predicted_output = network_output[:, :4]  # (N, 4)

        # === Compute Loss Only on NORMAL Nodes ===
        node_type = tf.squeeze(inputs['node_type'], axis=-1)
        loss_mask = tf.equal(node_type, common.NodeType.NORMAL)

        # Velocity Loss (L2)
        velocity_error = tf.reduce_sum(tf.square(
            target_normalized[:, :3] - predicted_output[:, :3]
        ), axis=-1)
        velocity_loss = tf.reduce_mean(tf.boolean_mask(velocity_error, loss_mask))

        # Stress Loss (for logging only)
        stress_error = tf.square(target_normalized[:, 3] - predicted_output[:, 3])
        stress_loss = tf.reduce_mean(tf.boolean_mask(stress_error, loss_mask))

        self.last_stress_loss = stress_loss  # Store for external inspection

        # Diagnostic printouts (optional)
        print("Sample target_velocity (pre-normalization):", target_velocity[:5])
        print("Sample target_stress (pre-normalization):", target_stress[:5])
        print("Sample normalized velocity:", target_normalized[:5, :3])
        print("Sample normalized stress:", target_normalized[:5, 3])

        return velocity_loss  # Only velocity loss used in training

    def _update(self, inputs, per_node_network_output):
        """
        Updates the node positions using predicted velocity.

        Parameters:
        -----------
        inputs : dict
            Contains current world positions.
        per_node_network_output : tf.Tensor
            Normalized output of model, shape (N, 4).

        Returns:
        --------
        tf.Tensor
            Updated node positions (N, 3).
        """
        velocity = self._output_normalizer.inverse(per_node_network_output)  # (N, 4)
        cur_position = inputs['world_pos']  # (N, 3)
        next_position = cur_position + velocity[:, :3]  # (N, 3)
        return next_position

    def __call__(self, inputs, training=False):
        """
        Forward pass through the model.

        Parameters:
        -----------
        inputs : dict
            Contains mesh and state inputs.
        training : bool
            Whether in training mode.

        Returns:
        --------
        dict
            Dictionary of updated positions, velocities, and stress.
        """
        graph = self._build_graph(inputs, is_training=training)
        outputs = self._learned_model(graph)  # (N, 4) normalized

        # Inverse normalization
        velocity_stress = self._output_normalizer.inverse(outputs)  # (N, 4)

        return {
            "world_pos": self._update(inputs, outputs),        # (N, 3)
            "cur_position": inputs['world_pos'],               # (N, 3)
            "cur_velocity": velocity_stress[:, :3],            # (N, 3)
            "stress": velocity_stress[:, 3:]                   # (N, 1)
        }
