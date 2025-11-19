# FANL Reproducibility Package

This repository provides the minimal reproducibility package for the paper:

**FANL: Adaptive Fuzzy Attention Network Layer for Medical Image Analysis**  
*(Authors omitted for blind review)*

The package contains all materials requested by reviewers for ensuring full reproducibility, 
including dataset indices, configuration files, TorchScript model, and minimal inference scripts.

---

## 📌 Contents
```
FANL_Reproducibility_Package/
├── fanl_test.exe
├── adaptive_fanl_traced.pt
├── inference_instructions.txt
├── test_samples/
├── dll/
├── configs/
├── indices/
└── minimal_scripts/
```

---

## 📦 Components

### 1. TorchScript Model
- `adaptive_fanl_traced.pt`  
  Pre-trained FANL model exported using `torch.jit.trace`.

### 2. GPU Inference Executable
- `fanl_test.exe`  
  A standalone C++/LibTorch binary capable of PathMNIST inference using GPU acceleration.

### 3. Dataset Indices (Required by Reviewers)
Located in the `indices/` folder:
- PathMNIST balanced 30k subset (train/val/test)
- PathMNIST full dataset (train/val/test)
- BloodMNIST full dataset (train/val/test)

These indices allow any user to reproduce the **exact** dataset partitions used in the paper.

### 4. Configuration Files
Stored in `configs/`:
- `preprocessing_config.json`
- `training_config.json`
- `model_export_config.json`

Each file documents the preprocessing pipeline, model hyperparameters, and TorchScript export settings.

### 5. Minimal Scripts
Provided in `minimal_scripts/`:
- `fanl_test.cpp`  — example C++ inference script  
- `CMakeLists.txt` — compilation instructions for LibTorch

These scripts allow users to rebuild the inference binary if desired.

### 6. Runtime Libraries
Stored in the `dll/` directory:
All required LibTorch + CUDA DLLs are provided to ensure out-of-the-box execution on Windows.

---

## 🔧 Runtime Libraries (Download from Release)

Due to GitHub’s 100 MB per-file limit, the required LibTorch runtime DLLs and
the pre-trained FANL TorchScript model are provided in the **GitHub Release**
rather than in the repository itself.

You can download all required runtime files (including `torch_cpu.dll`,`torch_cuda.dll`....) from the Release section:

👉 https://github.com/ZhaoDreamGo/FANL-Reproducibility-Package.git

These files must be placed alongside `fanl_test.exe` when running inference.

## ▶️ Running Inference

Full instructions are provided in:

inference_instructions.txt

🧪 Environment

LibTorch 2.6.0 (CUDA 11.8)

cuDNN 8.x

Windows 10 / 11 (64-bit)

NVIDIA RTX 3060 Ti GPU

📚 Citation

Please cite the accompanying paper if you use this package.

📧 Contact

For questions or clarifications, please contact the corresponding author.
Or contact us directly at zhao1028166352@gmail.com
