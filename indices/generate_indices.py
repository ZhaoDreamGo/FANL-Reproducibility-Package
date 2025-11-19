import os
import random
import numpy as np
from medmnist import PathMNIST, BloodMNIST
from collections import defaultdict


# =========================
# Global Configuration
# =========================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


OUTPUT_DIR = "indices"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_indices(file_path, indices):
    """Save a list of indices (one per line)"""
    with open(file_path, "w") as f:
        for idx in indices:
            f.write(f"{idx}\n")
    print(f"Saved {file_path}  ({len(indices)} entries)")


def train_val_test_split(indices, ratio=(0.7, 0.15, 0.15)):
    """Perform a 70/15/15 split"""
    total = len(indices)
    random.shuffle(indices)

    n_train = int(total * ratio[0])
    n_val = int(total * ratio[1])
    n_test = total - n_train - n_val

    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:]
    )


# =========================
# 1. PathMNIST 30k balanced subset
# =========================

def generate_pathmnist_30k_indices():
    dataset = PathMNIST(split='train', download=True)
    labels = [int(dataset[i][1]) for i in range(len(dataset))]
    class_dict = defaultdict(list)

    # group by class
    for idx, lbl in enumerate(labels):
        class_dict[lbl].append(idx)

    TARGET_TOTAL = 30000
    NUM_CLASSES = 9
    samples_per_class = TARGET_TOTAL // NUM_CLASSES
    extra = TARGET_TOTAL % NUM_CLASSES

    selected = []

    for c in range(NUM_CLASSES):
        target_num = samples_per_class + (1 if c < extra else 0)
        # random but reproducible
        selected_indices = random.sample(class_dict[c], target_num)
        selected.extend(selected_indices)

    random.shuffle(selected)

    # split
    train_idx, val_idx, test_idx = train_val_test_split(selected)

    # save
    save_indices(os.path.join(OUTPUT_DIR, "indices_30k_train.txt"), train_idx)
    save_indices(os.path.join(OUTPUT_DIR, "indices_30k_val.txt"), val_idx)
    save_indices(os.path.join(OUTPUT_DIR, "indices_30k_test.txt"), test_idx)


# =========================
# 2. PathMNIST full dataset
# =========================

def generate_pathmnist_full_indices():
    dataset = PathMNIST(split='train', download=True)
    full_indices = list(range(len(dataset)))

    train_idx, val_idx, test_idx = train_val_test_split(full_indices)

    save_indices(os.path.join(OUTPUT_DIR, "indices_full_pathmnist_train.txt"), train_idx)
    save_indices(os.path.join(OUTPUT_DIR, "indices_full_pathmnist_val.txt"), val_idx)
    save_indices(os.path.join(OUTPUT_DIR, "indices_full_pathmnist_test.txt"), test_idx)


# =========================
# 3. BloodMNIST full dataset
# =========================

def generate_bloodmnist_full_indices():
    dataset = BloodMNIST(split='train', download=True)
    full_indices = list(range(len(dataset)))

    train_idx, val_idx, test_idx = train_val_test_split(full_indices)

    save_indices(os.path.join(OUTPUT_DIR, "indices_full_bloodmnist_train.txt"), train_idx)
    save_indices(os.path.join(OUTPUT_DIR, "indices_full_bloodmnist_val.txt"), val_idx)
    save_indices(os.path.join(OUTPUT_DIR, "indices_full_bloodmnist_test.txt"), test_idx)


# =========================
# Write README file
# =========================

def write_readme():
    readme_path = os.path.join(OUTPUT_DIR, "indices_readme.txt")
    text = """Indices Documentation
=====================

This directory contains the exact sample indices used in all experiments
reported in the paper. These indices reproduce the 70/15/15 train, validation,
and test splits using the same random seed (42) as in the manuscript.

--------------------------------------------
1. PathMNIST 30,000-sample balanced subset
--------------------------------------------

Files:
- indices_30k_train.txt
- indices_30k_val.txt
- indices_30k_test.txt

Constructed as:
- balanced subset across 9 classes
- total = 30,000 samples
- random seed = 42
- split ratio = 70/15/15

--------------------------------------------
2. PathMNIST Full Dataset (107,180 samples)
--------------------------------------------

Files:
- indices_full_pathmnist_train.txt
- indices_full_pathmnist_val.txt
- indices_full_pathmnist_test.txt

--------------------------------------------
3. BloodMNIST Full Dataset (17,092 samples)
--------------------------------------------

Files:
- indices_full_bloodmnist_train.txt
- indices_full_bloodmnist_val.txt
- indices_full_bloodmnist_test.txt

--------------------------------------------
Format:
Each file contains one integer index per line.
These indices correspond to MedMNIST dataset sample ordering.
"""
    with open(readme_path, "w") as f:
        f.write(text)

    print(f"Saved README: {readme_path}")


# =========================
# Main
# =========================

if __name__ == "__main__":
    print("Generating dataset indices...")

    generate_pathmnist_30k_indices()
    generate_pathmnist_full_indices()
    generate_bloodmnist_full_indices()
    write_readme()

    print("All indices generated successfully!")
