# MeshGraphNets in TensorFlow 2 & Sonnet 2

This repository is a reimplementation of [**MeshGraphNets**](https://arxiv.org/abs/2010.03409), originally developed by DeepMind and presented at ICLR 2021.

- **Paper**: [arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409)  
- **Video site**: [sites.google.com/view/meshgraphnets](https://sites.google.com/view/meshgraphnets)  
- **Original Source (JAX/TF1)**: [deepmind-research/meshgraphnets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets)

This version is implemented in **TensorFlow 2** and **Sonnet 2** and focuses on usability, clarity, and extendibility for physics-based simulations. It was developed as part of an academic project combining **Graph Neural Networks (GNNs)** with **PDE-based simulations**.

---

## Overview

This code reproduces the MeshGraphNets model using a modular Encode-Process-Decode architecture. It currently supports the **deforming_plate** dataset and includes:

- Training, evaluation, and rollout tools
- Structured configuration for easy experimentation

Although the results may not exactly replicate the original MeshGraphNets outputs, the core logic and training dynamics are preserved. Future updates will extend the functionality to more datasets and GNN variants.

---

## Setup

Install the required packages:

```bash
pip install -r requirements.txt

---

## Citations 

@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}

---

## Downloading the dataset 

Set the dataset name and output directory

DATASET_NAME=deforming_plate
OUTPUT_DIR=./data/${DATASET_NAME}
BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

## Train the model

python run.py --mode=train --model=deforming_plate --dataset_dir="Mesh_Deforming\data\deforming_plate" --checkpoint_dir="Mesh_Deforming\checkpoints"
This will start training the model using the deforming_plate dataset. You can monitor logs and losses from the terminal or log file.

## Run rollout evaluation

python run.py --mode=eval --model=deforming_plate --dataset_dir="Mesh_Deforming\data\deforming_plate" --checkpoint_dir="Mesh_Deforming\checkpoints"
This generates mesh predictions over time (in VTK or Pickle format) using the saved model.