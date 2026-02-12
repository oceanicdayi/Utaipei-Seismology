# Utaipei-Seismology

An educational repository from the **University of Taipei (臺北市立大學)** that combines **seismology data analysis** with a structured **machine learning curriculum**. Students learn to retrieve, process, and visualize real-world earthquake seismic data, then apply classical and deep-learning techniques — culminating in ML-based earthquake detection.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Learning Strategy](#learning-strategy)
  - [Phase 1 — Seismic Data Fundamentals](#phase-1--seismic-data-fundamentals-weeks-12)
  - [Phase 2 — Data Science Essentials](#phase-2--data-science-essentials-weeks-34)
  - [Phase 3 — Classical Machine Learning](#phase-3--classical-machine-learning-weeks-56)
  - [Phase 4 — Unsupervised Learning](#phase-4--unsupervised-learning-week-7)
  - [Phase 5 — Deep Learning with Keras](#phase-5--deep-learning-with-keras-weeks-810)
  - [Phase 6 — Capstone: Earthquake Detection](#phase-6--capstone-earthquake-detection-weeks-1112)
- [Key Technologies](#key-technologies)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project serves two purposes:

1. **Seismic data pipeline** — Fetch earthquake events, station metadata, and waveforms from the IRIS FDSN web service using [ObsPy](https://docs.obspy.org/), then prepare features suitable for machine learning.
2. **ML100 course** — A structured ~100-day machine learning curriculum (Jupyter notebooks with examples and homework) that progresses from exploratory data analysis through deep learning and CNNs.

The final goal is to train models that can **detect earthquakes** from seismic waveform features.

---

## Repository Structure

```
Utaipei-Seismology/
│
├── README.md                              # This file
├── Release-Notes.txt                      # Project release notes
│
├── # ── Seismic Data & ML Application ──────────────────────
├── get_data_from_fdsn-ntust.py            # Script: fetch events & waveforms from IRIS
├── read_seiscomp_data_from_fdsn.ipynb     # Notebook: query IRIS for events & stations
├── prepare_seismic_data_for_ML.ipynb      # Notebook: build training dataset from waveforms
├── prepare_seismic_test_data_for_ML.ipynb # Notebook: build test dataset from waveforms
├── detect_EQ.ipynb                        # Notebook: earthquake detection model
│
├── # ── MNIST / Keras Tutorials ────────────────────────────
├── 0707project.ipynb                      # Early project notebook (Keras)
├── 0714_Keras_Mnist_Introduce.ipynb       # MNIST digit recognition introduction
├── 0714_Keras_Mnist_MLP_h256.ipynb        # MNIST MLP with 256 hidden units
├── Keras_Mnist_CNN.ipynb                  # MNIST CNN (Conv2D, MaxPooling, Dropout)
│
├── # ── ML100 Curriculum ───────────────────────────────────
├── ML100/                                 # ~150 notebooks: structured 100-day ML course
│   ├── Day_001–Day_006  (EDA & basics)
│   ├── Day_007–Day_021  (Feature types, outliers, correlation, first model)
│   ├── Day_022–Day_031  (Feature engineering & selection)
│   ├── Day_032–Day_050  (Regression, trees, ensemble methods)
│   ├── Day_054–Day_065  (Clustering, PCA, t-SNE)
│   ├── Day_066–Day_076  (Keras intro, loss, activation, optimizers)
│   ├── Day077–Day089    (Overfitting, regularization, callbacks)
│   ├── Day090–Day100    (CV, CNN theory, data augmentation, transfer learning)
│   ├── Day104–Day105    (ConvNetJS, advanced CNN)
│   ├── Day-01/ … Day-06/  (Student homework submissions)
│   ├── resnet_builder.py                  # ResNet model builder utility
│   └── HomeCredit_columns_description.csv # Dataset metadata for exercises
│
└── # ── Student Directories ────────────────────────────────
    ├── JoChen31.txt
    ├── 林育謙.txt
    ├── 王暄昀.txt
    ├── 潘勝彥/
    └── 陳暐力/
```

---

## Prerequisites

| Requirement | Recommended Version |
|---|---|
| Python | 3.8 + |
| Jupyter Notebook / JupyterLab | latest |
| pip | latest |

A basic understanding of **Python programming** and **linear algebra / statistics** will be helpful but is not strictly required — the ML100 curriculum starts from the basics.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/oceanicdayi/Utaipei-Seismology.git
cd Utaipei-Seismology

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

# 3. Install core dependencies
pip install numpy pandas matplotlib scikit-learn

# 4. Install seismology libraries
pip install obspy geopy

# 5. Install deep-learning libraries
pip install tensorflow keras

# 6. Launch Jupyter
jupyter notebook
```

> **Note:** Some ML100 notebooks may require additional packages (e.g., `lightgbm`, `xgboost`, `imageio`). Install them as needed when prompted by import errors.

---

## Learning Strategy

The recommended learning path is organized into six phases. Each phase builds on the previous one, so following the order will give you the best experience.

### Phase 1 — Seismic Data Fundamentals (Weeks 1–2)

**Goal:** Understand how real earthquake data is acquired and processed.

| Step | Resource | What You Will Learn |
|---|---|---|
| 1 | `read_seiscomp_data_from_fdsn.ipynb` | Query the IRIS FDSN service for earthquake events and station metadata |
| 2 | `get_data_from_fdsn-ntust.py` | Fetch waveforms, calculate P-wave arrival times, filter by epicentral distance |
| 3 | `prepare_seismic_data_for_ML.ipynb` | Extract features from waveforms and build a training dataset |
| 4 | `prepare_seismic_test_data_for_ML.ipynb` | Build a corresponding test dataset |

**Key skills:** ObsPy, FDSN web services, waveform processing, feature extraction.

### Phase 2 — Data Science Essentials (Weeks 3–4)

**Goal:** Master exploratory data analysis and data preprocessing.

| ML100 Days | Topic | Key Notebooks |
|---|---|---|
| 1–6 | Metrics, EDA basics, DataFrames | `Day_001_example_of_metrics`, `Day_004_first_EDA`, `Day_005-*`, `Day_006_column_data_type` |
| 7–13 | Feature types, outliers, missing values | `Day_007_Feature_Types`, `Day_009_outliers_detection`, `Day_012_Fill_NaN_and_Scalers` |
| 14–21 | Correlation, KDE plots, heatmaps, first model | `Day_014_correlation_example`, `Day_016_EDA_KDEplots`, `Day_020_EDA_heatmap`, `Day_021_first_model` |

**Tip:** Complete the `_HW` (homework) notebook for each day before looking at the `_Ans` answer notebook.

### Phase 3 — Classical Machine Learning (Weeks 5–6)

**Goal:** Learn feature engineering, model evaluation, and ensemble methods.

| ML100 Days | Topic | Key Notebooks |
|---|---|---|
| 22–31 | Feature engineering (encoding, selection, importance) | `Day_024_LabelEncoder_and_OneHotEncoder`, `Day_030_Feature_Selection` |
| 32–40 | Train/test split, evaluation metrics, regression | `Day_034_train_test_split`, `Day_038_regression_model`, `Day_040_lasso_ridge_regression` |
| 41–50 | Decision trees, random forest, gradient boosting, blending, stacking | `Day_042_decision_tree`, `Day_044_random_forest`, `Day_046_gradient_boosting_machine` |

### Phase 4 — Unsupervised Learning (Week 7)

**Goal:** Explore clustering and dimensionality-reduction techniques.

| ML100 Days | Topic | Key Notebooks |
|---|---|---|
| 54–56 | K-Means clustering | `Day_055_kmean_sample`, `Day_056_kmean` |
| 57–58 | Hierarchical clustering | `Day_057_hierarchical_clustering_sample` |
| 59–62 | PCA and t-SNE | `Day_059_PCA_sample`, `Day_061_tsne_sample` |

### Phase 5 — Deep Learning with Keras (Weeks 8–10)

**Goal:** Build and train neural networks for image classification, then understand CNNs.

| ML100 Days | Topic | Key Notebooks |
|---|---|---|
| 66–69 | Keras intro, datasets, Sequential & Functional API | `Day66-Keras_Introduction`, `Day68-Keras_Sequential_Model` |
| 70–76 | MNIST MLP, loss functions, activation, gradient descent, optimizers | `Day70-Keras_Mnist_MLP_Sample`, `Day72-Activation_function`, `Day75-Back_Propagation` |
| 77–89 | Overfitting, regularization, dropout, batch norm, callbacks, custom loss | `Day077_overfitting`, `Day082_Dropout`, `Day085_CB_EarlyStop` |
| 90–100 | Computer vision, CNN theory, convolution, pooling, augmentation, transfer learning | `Day092_CNN_theory`, `Day094-CNN_Convolution`, `Day099_data_augmentation`, `Day100_transfer_learning` |

**Supplementary root-level notebooks:**

| Notebook | Description |
|---|---|
| `0714_Keras_Mnist_Introduce.ipynb` | Quick-start MNIST walkthrough |
| `0714_Keras_Mnist_MLP_h256.ipynb` | MLP with 256 hidden units |
| `Keras_Mnist_CNN.ipynb` | CNN architecture (Conv2D → MaxPooling → Dropout → Dense) |

### Phase 6 — Capstone: Earthquake Detection (Weeks 11–12)

**Goal:** Apply everything you have learned to detect earthquakes from seismic data.

| Step | Resource | What You Will Learn |
|---|---|---|
| 1 | Review Phase 1 notebooks | Refresh your understanding of the seismic feature pipeline |
| 2 | `detect_EQ.ipynb` | Load prepared seismic features, normalize data, one-hot encode labels, and train a classifier |
| 3 | Experiment | Try different architectures (MLP vs CNN), tune hyperparameters, apply regularization techniques from Phase 5 |

---

## Key Technologies

| Library | Purpose |
|---|---|
| [ObsPy](https://docs.obspy.org/) | Seismological data retrieval and waveform processing |
| [geopy](https://geopy.readthedocs.io/) | Geographic distance calculations |
| [NumPy](https://numpy.org/) | Numerical computing |
| [Pandas](https://pandas.pydata.org/) | Data manipulation and analysis |
| [Matplotlib](https://matplotlib.org/) | Data visualization |
| [scikit-learn](https://scikit-learn.org/) | Classical machine learning algorithms and metrics |
| [TensorFlow / Keras](https://www.tensorflow.org/) | Deep learning framework |

---

## Contributing

Contributions are welcome! If you find errors, want to add new exercises, or improve existing notebooks:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -m 'Add new exercise for Day X'`)
4. Push to your branch (`git push origin feature/my-improvement`)
5. Open a Pull Request

---

## License

This project is provided for **educational purposes**. Please refer to the repository owner for specific licensing information.