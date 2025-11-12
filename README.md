# 🌀 Randomized-HOEDMD

Official implementation of the paper  
**"Randomized Structured-TLS Based Higher Order Extended Dynamic Mode Decomposition"**

---

## 📘 Overview
This repository provides the implementation of the algorithms proposed in the paper  
*Randomized Structured-TLS Based Higher Order Extended Dynamic Mode Decomposition*.  

It includes Python implementations of:
- **Standard DMD** — `lapack_DMD.py`
- **Compressed DMD** — `compress_DMD.py`
- **Randomized DMD** — `rDMD.py`
- **Deterministic and Randomized HOEDMD** — `Randomized_HOEDMD.py`

The repository also contains scripts to reproduce all main results and figures reported in the paper.

---

## 🧩 Environment Setup

You can set up the environment in one of two ways:

### Option 1: Using `environment.yml`
```bash
conda env create -f environment.yml
conda activate randomized-hoedmd

### Option 2: Using `requirements.txt`

pip install -r requirements.txt 





