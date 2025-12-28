# 🏡 Regression Models: ML & DL Implementations

<p align="center">
  <img src="https://socialify.git.ci/3bsalam-1/Regression-Models/image?description=1&language=1&name=1&owner=1&pattern=Solid&pulls=1&stargazers=1&theme=Auto" alt="project-image">
</p>

A comprehensive collection of regression model implementations using both Machine Learning and Deep Learning approaches for predicting housing prices.

## 📊 Project Overview

This repository contains multiple implementations of regression algorithms, from basic linear regression built from scratch to advanced ensemble methods and neural networks. The project demonstrates various approaches to solving regression problems, with a focus on housing price prediction.

## 📁 Project Structure

```
Regression-Models/
├── ML/                                     # Machine Learning Implementations
│   ├── 01_linear_regression_from_scratch/
│   │   ├── linear_regression_single_feature.ipynb
│   │   ├── data/
│   │   │   └── ex1data1.txt
│   │   └── outputs/
│   │       ├── cost_function.png
│   │       ├── dataset1.png
│   │       ├── learning_rate.png
│   │       └── regression_result.png
│   ├── 02_multivariate_linear_regression/
│   │   ├── linear_regression_multiple_features.ipynb
│   │   ├── data/
│   │   │   └── ex1data2.txt
│   │   ├── outputs/
│   │   │   ├── cost_function.png
│   │   │   ├── dataset1.png
│   │   │   ├── learning_rate.png
│   │   │   └── regression_result.png
│   │   └── assets/
│   │       └── Linear Regression Cheat Sheet.png
│   └── 03_sklearn_regression_models/
│       ├── california_housing_comparison.ipynb
│       ├── data/
│       │   ├── housing.csv
│       │   └── README.md
│       └── outputs/
└── DL/                                     # Deep Learning Implementations
    └── 01_tensorflow_ann_regression/
        ├── boston_housing_ann.ipynb
        ├── data/
        │   └── housing.csv
        ├── utils/
        │   └── planar_utils.py
        └── outputs/
            └── images/
```

## 🤖 Machine Learning Models

### 01. Linear Regression (From Scratch)
A complete implementation of linear regression with gradient descent built from the ground up using only NumPy.

**Features:**
- Single feature linear regression
- Gradient descent optimization
- Cost function visualization
- Learning rate analysis

**Accuracy:** N/A (Foundational implementation)

**Key Files:**
- `linear_regression_single_feature.ipynb`: Complete implementation with visualizations
- `data/ex1data1.txt`: Food truck profit vs. population data

### 02. Multivariate Linear Regression
Extension of linear regression to handle multiple features with feature normalization.

**Features:**
- Multiple feature regression
- Feature normalization/scaling
- Gradient descent for multiple variables
- Normal equation method

**Accuracy:** N/A (Training exercise)

**Key Files:**
- `linear_regression_multiple_features.ipynb`: Multi-feature implementation
- `data/ex1data2.txt`: Housing data (size, bedrooms, price)

### 03. Scikit-learn Regression Models
Comparison of various regression algorithms using California Housing dataset.

**Models Implemented:**
- DecisionTreeRegressor
- GradientBoostingRegressor
- XGBRegressor

**Accuracy Range:** ~79%

**Dataset:** California Housing (20,640 samples, 8 features)

**Key Files:**
- `california_housing_comparison.ipynb`: Model comparison and evaluation
- `data/housing.csv`: California housing dataset
- `data/README.md`: Dataset documentation

## 🧠 Deep Learning Models

### 01. TensorFlow ANN Regression
Artificial Neural Network implementation using TensorFlow/Keras for housing price prediction.

**Architecture:**
- Input Layer: 13 features
- Hidden Layers: 3 layers with 100 neurons each (customizable)
- Output Layer: 1 neuron (price prediction)
- Activation: ReLU

**Accuracy Range:** ~85%

**Key Files:**
- `boston_housing_ann.ipynb`: Complete ANN implementation
- `data/housing.csv`: Boston housing dataset
- `utils/planar_utils.py`: Helper utilities

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/3bsalam-1/Regression-Models.git
cd Regression-Models
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### Running Machine Learning Models

```bash
# Navigate to ML directory
cd ML/01_linear_regression_from_scratch

# Open Jupyter notebook
jupyter notebook linear_regression_single_feature.ipynb
```

#### Running Deep Learning Models

```bash
# Navigate to DL directory
cd DL/01_tensorflow_ann_regression

# Open Jupyter notebook
jupyter notebook boston_housing_ann.ipynb
```

## 📊 Datasets

### California Housing Dataset
- **Source:** [Luis Torgo's page](http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)
- **Samples:** 20,640
- **Features:** 8 (longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income)
- **Target:** median_house_value
- **Description:** Census data from California (1990)

### Boston Housing Dataset
- **Samples:** 506
- **Features:** 13 (including CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT)
- **Target:** Median house value

## 📈 Performance Comparison

| Model | Accuracy | Type | Complexity |
|-------|----------|------|------------|
| Linear Regression (Scratch) | N/A | ML | Low |
| Multivariate Linear Regression | N/A | ML | Low |
| DecisionTreeRegressor | ~79% | ML | Medium |
| GradientBoostingRegressor | ~79% | ML | Medium |
| XGBRegressor | ~79% | ML | Medium |
| TensorFlow ANN | ~85% | DL | High |

## 🛠️ Technologies Used

- **NumPy:** Numerical computing
- **Pandas:** Data manipulation
- **Matplotlib:** Data visualization
- **Scikit-learn:** ML algorithms and preprocessing
- **TensorFlow/Keras:** Deep learning framework
- **XGBoost:** Gradient boosting library

## 📝 License

This project is licensed under the terms found in the [LICENSE](LICENSE) file.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 👨‍💻 Author

**3bsalam-1**
- GitHub: [@3bsalam-1](https://github.com/3bsalam-1)

## 🌟 Show your support

Give a ⭐️ if this project helped you!

---

*Note: This project is created for educational purposes to demonstrate various regression techniques in machine learning and deep learning.*
