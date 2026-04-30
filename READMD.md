
# Feature Detection on Human Body Thermograms  
### Comparative Evaluation of Classical and Deep Learning Keypoint Detectors

---

## Overview

This repository evaluates several classical and deep learning-based keypoint detectors on breast thermograms.  
The goal is to compare how well each detector identifies meaningful keypoints across different thermographic viewpoints.

⚠️ Note: This project evaluates only detection quality.  
Descriptor extraction, matching, or image registration are not included.

---

## Dataset

The dataset used in this project is publicly available:

https://visual.ic.uff.br/

Because of its size, the dataset is not included in this repository.  
After downloading, place it inside:

```
experiments/initial_data/
```

---

## Dataset Structure

Expected directory layout:

```
initial_data/
│
├── h1/
│   ├── front
│   ├── left45
│   ├── left90
│   ├── right45
│   └── right90
│
├── h2/
│   └── ...
│
├── ...
│
├── s1/
│   └── ...
│
└── s40/
```

### Dataset Contents:

- 43 healthy patients (h)
- 43 sick patients (s)  
- 5 thermographic views per patient:
  - front  
  - left 45°  
  - left 90°  
  - right 45°  
  - right 90°

---

## Methods Evaluated

### Classical Detectors
- SIFT  
- ORB  
- BRISK  
- AKAZE  

### Deep Learning-Based Detectors
- DISK  
- R2D2  

The R2D2 implementation is taken from:

https://github.com/naver/r2d2

Only the necessary inference modules are integrated into `src/scripts/models/r2d2/`.

---

## Evaluation Metric

### Detection Quality

```
Detection Quality = (# of Keypoints Inside ROI) / (Total Keypoints)
```

- Total Keypoints: all detected keypoints  
- Inside Keypoints: keypoints located inside a pre-defined Region of Interest (ROI)

Higher values indicate better detection behavior.

---

## Results

### Quantitative Results  
Stored in:

```
experiments/results/results.csv
```

### Visualizations  
Available in:

```
experiments/results/bar_methods_by_view.jpg
```

### Qualitative Samples  
Detection examples for each method are provided in:

```
assets/examples/
```

---

## Project Structure

```
project-root/
│
scripts/
├── classical/
├── deep/
├── evaluation/
└── models/
|   └── r2d2/   (ignored)
│
├── experiments/
│   ├── initial_data/   (ignored)
│   └── results/
│
├── assets/
│   └── examples/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python scripts/DQ.py
```

Dataset must exist at:

```
experiments/initial_data/
```

---

## Acknowledgment

This project uses part of the R2D2 implementation from:

https://github.com/naver/r2d2

---
