# MeshGraphNets in TensorFlow 2 & Sonnet 2

This repository is a reimplementation of [**MeshGraphNets**](https://arxiv.org/abs/2010.03409), originally developed by DeepMind and presented at ICLR 2021.

- **Paper**: [arxiv.org/abs/2010.03409](https://arxiv.org/abs/2010.03409)  
- **Video site**: [sites.google.com/view/meshgraphnets](https://sites.google.com/view/meshgraphnets)  
- **Original Source (JAX/TF1)**: [github.com/deepmind-research/meshgraphnets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets)

This version is implemented in **TensorFlow 2** and **Sonnet 2**, with a focus on usability, clarity, and extendibility for physics-based simulations. It was developed as part of an academic project combining **Graph Neural Networks (GNNs)** with **PDE-based simulations**.

---

## Overview

This code reproduces the MeshGraphNets model using a modular Encode-Process-Decode architecture. It currently supports the **deforming_plate** dataset and includes:

- Training, evaluation, and rollout tools  
- Structured configuration for easy experimentation  
- Modular design compatible with additional datasets in the future  

While the results may not exactly match the original DeepMind implementation, the core structure and learning dynamics are preserved.

---

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Downloading the Dataset

Download the **deforming_plate** dataset directly from DeepMind:

```bash
DATASET_NAME=deforming_plate
OUTPUT_DIR=./data/${DATASET_NAME}
BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}/"

mkdir -p ${OUTPUT_DIR}
for file in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
    wget "${BASE_URL}${file}" -O "${OUTPUT_DIR}/${file}"
done
```

Then generate index files for the `tfrecord` reader:

```bash
pip install tfrecord
python -m tfrecord.tools.tfrecord2idx ${OUTPUT_DIR}/train.tfrecord ${OUTPUT_DIR}/train.idx
python -m tfrecord.tools.tfrecord2idx ${OUTPUT_DIR}/valid.tfrecord ${OUTPUT_DIR}/valid.idx
python -m tfrecord.tools.tfrecord2idx ${OUTPUT_DIR}/test.tfrecord ${OUTPUT_DIR}/test.idx
```

---

## 🏃 Running the Code

### Train the model

```bash
python run.py --mode=train --model=deforming_plate --dataset_dir="Mesh_Deforming/data/deforming_plate" --checkpoint_dir="Mesh_Deforming/checkpoints"
```

This will start training the model using the deforming_plate dataset. You can monitor logs and losses from the terminal or log file.

---

### Run rollout evaluation

```bash
python run.py --mode=eval --model=deforming_plate --dataset_dir="Mesh_Deforming/data/deforming_plate" --checkpoint_dir="Mesh_Deforming/checkpoints"
```

This generates mesh predictions over time (in VTK or Pickle format) using the saved model.

---

### (Optional) Visualize with Streamlit

If you enabled VTK output during rollout, you can visualize it interactively:

```bash
streamlit run streamlit_live_dashboard.py -- --vtk_dir ./rollouts/vtk
```

This opens a dashboard for exploring scalar fields, animations, and geometry evolution.

---

## Citation

If you use this work or build on it, please cite the original paper:

```bibtex
@inproceedings{pfaff2021learning,
  title={Learning Mesh-Based Simulation with Graph Networks},
  author={Pfaff, Tobias and Fortunato, Meire and Sanchez-Gonzalez, Alvaro and Battaglia, Peter},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

---

## Author

**Amir Bazzi**  
PhD Student working on Graph Neural Networks for PDE-based industrial simulations (CIFRE)  
🔗 [LinkedIn](https://linkedin.com/in/amirbazzi) • 🌐 [amirbazzi.dev](https://amirbazzi.dev)

---

