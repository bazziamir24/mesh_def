import functools
import json
import os

import tensorflow as tf

from common import NodeType

@tf.function 
def _parse(proto, meta):
    """Parses a trajectory from a serialized tf.Example in TensorFlow 2."""
    # Define the expected features as variable-length string features.
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                     for k in meta['field_names']}
    # Parse the input proto according to the defined feature lists.
    features = tf.io.parse_single_example(proto, feature_lists)
    
    out = {}
    for key, field in meta['features'].items():
        # Decode the raw bytes into the proper dtype (e.g., tf.float32 or tf.int32)
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        # Reshape according to the provided shape.
        data = tf.reshape(data, field['shape'])
        
        # Process the data based on its type.
        if field['type'] == 'static':
            # Repeat static features along the trajectory length.
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            # Decode the corresponding length field and create a RaggedTensor.
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            # For any unsupported type, raise an error.
            raise ValueError('invalid data format')
        
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset """
    # Load metadata from meta.json.
    meta_path = os.path.join(path, 'meta.json')
    with open(meta_path, 'r') as fp:
        meta = json.load(fp)
    
    # Build the path to the TFRecord file.
    tfrecord_path = os.path.join(path, f"{split}.tfrecord")
    ds = tf.data.TFRecordDataset(tfrecord_path)
    
    # Map the _parse function with meta info.
    ds = ds.map(lambda proto: _parse(proto, meta),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch to improve latency.
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def add_targets(ds, fields, add_history):
    """Adds target and optionally history fields to dataset trajectories"""
    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out['prev|' + key] = val[0:-2]
                out['target|' + key] = val[2:]
        return out
    return ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)



def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma):
    """Splits trajectories into frames, and adds training noise."""
    
    def add_noise(frame):
        noise = tf.random.normal(tf.shape(frame[noise_field]),
                                 stddev=noise_scale, dtype=tf.float32)
        
        # Only apply noise to normal nodes
        mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
        noise = tf.where(mask[:, tf.newaxis], noise, tf.zeros_like(noise))
        
        frame[noise_field] += noise
        frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    # Split trajectories into individual frames
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
    
    # Apply noise to each frame
    ds = ds.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle and repeat indefinitely
    ds = ds.shuffle(10000)
    ds = ds.repeat()
    
    # Prefetch for performance
    return ds.prefetch(tf.data.AUTOTUNE)


