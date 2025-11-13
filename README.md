# Randomized-HOEDMD

Official implementation of the paper  
**"Randomized Structured-TLS Based Higher Order Extended Dynamic Mode Decomposition"**

## Overview

This repository provides Python implementations for methods discussed in the paper and scripts to reproduce the main figures and tables.

Algorithm implementations:
- `lapack_DMD.py`: Standard Dynamic Mode Decomposition (DMD).
- `compress_DMD.py`: Compressed DMD.
- `rDMD.py`: Randomized DMD.
- `Randomized_HOEDMD.py`: Deterministic HOEDMD and the proposed Randomized HOEDMD.

## Environment Setup

You can set up the environment in one of two ways.

### Option 1: Using `environment.yml`
```bash
conda env create -n randomized-hoedmd -f environment.yml
conda activate randomized-hoedmd
```

### Option 2: Using `requirements.txt`
```bash
pip install -r requirements.txt
```

## Reproducing Experimental Results

After the environment is configured, run the following scripts to reproduce the results reported in the paper.

### 4.1. Synthetic Data
```bash
python Exp1_PMTM.py
```
Generates the data corresponding to **Figure 1**.

### 4.2. Fluid Flow Behind a Cylinder
```bash
python Exp2_eigvalue.py
```
Generates the results corresponding to **Figure 4** and **Figure 5**.

### 4.3. fMRI Data
For the tables:
```bash
python Exp3_table.py
```
Generates the data corresponding to **Table 1** and **Table 2**.

For the figure:
```bash
python Exp3_Modes.py
```
Generates **Figure 8**.

## File Description

| File | Description |
|------|-------------|
| `lapack_DMD.py` | Implementation of the **standard DMD** used in the paper. |
| `compress_DMD.py` | Implementation of the **compressed DMD**. |
| `rDMD.py` | Implementation of the **randomized DMD**. |
| `Randomized_HOEDMD.py` | Includes **deterministic HOEDMD** and the proposed **randomized HOEDMD** algorithms. |
| `Exp1_PMTM.py` | Script for Section **4.1 Synthetic Data** (produces Figure 1). |
| `Exp2_eigvalue.py` | Script for Section **4.2 Fluid Flow Behind a Cylinder** (produces Figures 4 and 5). |
| `Exp3_table.py` | Script for Section **4.3 fMRI Data** (produces Table 1 and Table 2). |
| `Exp3_Modes.py` | Script for Section **4.3 fMRI Data** (produces Figure 8). |
| `environment.yml` | Conda environment configuration. |
| `requirements.txt` | Python dependencies list. |


