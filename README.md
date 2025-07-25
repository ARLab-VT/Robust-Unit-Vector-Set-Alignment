# RobustVectorAlignment

## Overview

This repository contains the implementation and resources for **Spherical Point Pattern Registration** algorithms developed for **Experiment 1**. The following directories and files are included:

- **`vector_alignment_dataset/`**: Contains the dataset created for Experiment 1.
- **`vector_alignment_utils.py`**: Includes all the necessary functions for the basic algorithms:
  - **SPMC**
  - **FRS**
  - **SPMC+FRS**
- **`vector_alignment_visualization_utils.py`**: Contains functions for visualizing the alignment process and results.
- **`exp1_example.ipynb`**: A Jupyter Notebook demonstrating the basic implementation of the algorithms. It:
  - Loads a source pattern and template pattern.
  - Provides options to choose the desired algorithm version.
  - Performs the registration process.

## Installation

Follow these steps to set up and run the project:

### Prerequisites

Ensure you have the following installed:
- **Python**: Version 3.8 or higher

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```
### Step 2: Create and Activate a Virtual Environment

- **`On Unix/Linux/MacOS:/`**
```bash
python3 -m venv venv
source venv/bin/activate
```

- **`On Windows:/`**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
