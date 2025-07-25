# RobustVectorAlignment

## Overview

This repository provides the implementation and resources for **Spherical Point Pattern Registration** algorithms developed as part of **Experiment 1**. These algorithms aim to robustly align vector fields on the unit sphere.

### Directory Structure

- **`vector_alignment_dataset/`**  
  Contains the dataset curated for Experiment 1.

- **`vector_alignment_utils.py`**  
  Implements the core alignment algorithms:
  - **SPMC** (Spherical Probabilistic Matching via Correlation)
  - **FRS** (Fast Rotation Search)
  - **SPMC+FRS** (Hybrid approach)

- **`vector_alignment_visualization_utils.py`**  
  Includes utilities for visualizing the alignment process and evaluating results.

- **`exp1_example.ipynb`**  
  Jupyter Notebook demonstrating:
  - Loading of source and template patterns
  - Selection of the desired algorithm
  - Execution of the registration pipeline with visual feedback

## Installation

### Prerequisites

- **Python** ≥ 3.8

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```
### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
