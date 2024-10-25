# ImageAnalytics_ImageNet

## Overview

The **Image Analytics Assignment** focuses on applying machine learning techniques to image classification tasks using various algorithms. This project demonstrates the development, evaluation, and optimization of different models, including Logistic Regression, Random Forest Classifier, Support Vector Machine, and XGBoost. The goal is to explore the effectiveness of these models in classifying images based on features extracted from pretrained models.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Dataset Setup](#dataset-setup)
   - [Uploading the Dataset](#uploading-the-dataset)
   - [Running the Code](#running-the-code)
3. [Model Details](#model-details)
   - [Logistic Regression](#logistic-regression)
   - [Random Forest Classifier](#random-forest-classifier)
   - [Support Vector Machine](#support-vector-machine)
   - [XGBoost](#xgboost)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Cross-Validation](#cross-validation)
6. [Conclusion](#conclusion)
7. [Acknowledgments](#acknowledgments)

## Project Structure

The repository is organized as follows:

```
IMAGEANALYTICSASSIGNMENT/
│
├── code/                                   # Contains Jupyter notebooks for model development
│   ├── Base_Model_Development.ipynb        # Base model development and initial evaluations
│   ├── Data_Preparation.ipynb               # Data preparation and preprocessing steps
│   ├── Logistic_Regression.ipynb            # Logistic Regression model implementation
│   ├── Random_Forest_Classifier.ipynb       # Random Forest Classifier model implementation
│   ├── Support_Vector_Machine.ipynb         # Support Vector Machine model implementation
│   ├── XGBoost.ipynb                        # XGBoost model implementation
│   └── validate.py                          # Python script for model validation
│
├── data/                                    # Contains dataset files
│   ├── features/                            # Folder for features datasets
│   │   ├── split_data/                     # Folder for split datasets
│   │   │   ├── train_1_split.csv
│   │   │   ├── train_2_split.csv
│   │   │   ├── val_1_split.csv
│   │   │   └── val_2_split.csv
│   │   ├── train_efficientformerv2_s0.snap_dist_in1k.csv
│   │   ├── train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv
│   │   ├── v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv
│   │   ├── val_efficientformerv2_s0.snap_dist_in1k.csv
│   │   └── val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv
│   │   
│   └── images/                              # Folder for image datasets
│       └── imagenetv2-matched-frequency-format-val/
│
└── README.md                                # Project documentation
```

## Getting Started

### Prerequisites

To run this project, you'll need to have Python 3.x installed along with the necessary libraries. You can install the required libraries using pip. Here’s a list of the primary libraries used:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `scikit-learn`: For machine learning algorithms and tools.
- `matplotlib`: For data visualization.
- `seaborn`: For statistical data visualization.
- `xgboost`: For gradient boosting model implementation.

To install these packages, run the following command in your terminal:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Dataset Setup

1. **Download the Dataset**: Acquire the datasets required for this project. Ensure that the datasets are formatted correctly and placed in the appropriate directories. 

2. **Directory Structure**: Organize your datasets within the `data/` directory, following the structure outlined in the project structure section. The required datasets should be placed as follows:
   - Place all split CSV files in `data/features/split_data/`.
   - Place all feature CSV files in `data/features/`.
   - Place all image files in `data/images/imagenetv2-matched-frequency-format-val/`.

### Uploading the Dataset

Due to the size of the dataset, it may need to be uploaded separately. Make sure to follow these steps for a successful upload:

- **Upload the dataset** to the `data/features/` and `data/images/` directories as specified.
- Ensure that the dataset files match the expected naming conventions in the code for seamless integration.

### Running the Code

1. **Clone this repository** to your local machine using:

   ```bash
   git clone <repository-url>
   ```

2. **Open Jupyter Notebook** and navigate to the `code/` directory.

3. **Data Preparation**: First, run the `Data_Preparation.ipynb` notebook. This notebook will preprocess the datasets, split the training data, and prepare it for modeling.

4. **Model Training**: After preparing the data, run each of the model notebooks in the following order:
   - `Logistic_Regression.ipynb`
   - `Random_Forest_Classifier.ipynb`
   - `Support_Vector_Machine.ipynb`
   - `XGBoost.ipynb`

5. **Model Validation**: Use the `validate.py` script to perform any additional validation tasks or assessments as needed.

## Model Details

### Logistic Regression

The Logistic Regression model is implemented in the `Logistic_Regression.ipynb` notebook. It uses a pipeline that combines feature scaling with the Logistic Regression algorithm. Hyperparameter tuning is performed using grid search with cross-validation to optimize model performance.

### Random Forest Classifier

The Random Forest Classifier is developed in the `Random_Forest_Classifier.ipynb` notebook. This model utilizes an ensemble learning approach to improve prediction accuracy by averaging multiple decision trees. It also includes performance metrics evaluation.

### Support Vector Machine

The Support Vector Machine (SVM) model is constructed in the `Support_Vector_Machine.ipynb` notebook. The SVM classifier employs a linear kernel to classify the images based on extracted features.

### XGBoost

The XGBoost model is implemented in the `XGBoost.ipynb` notebook. XGBoost is an efficient and scalable implementation of the gradient boosting framework, which excels in performance and speed.

## Hyperparameter Tuning

Hyperparameter tuning is performed for the Logistic Regression model using Grid Search CV. This process helps in finding the best combination of hyperparameters for optimal model performance, minimizing overfitting and maximizing generalization.

## Cross-Validation

To ensure robust performance evaluation, k-fold cross-validation is utilized, where the training data is split into `k` subsets. The model is trained `k` times, each time using a different subset as the validation set while the remaining `k-1` subsets are used for training. This technique provides a more reliable estimate of the model’s performance.

## Conclusion

This project serves as a comprehensive exploration of image classification using various machine learning models. Each model has been evaluated with different techniques to ensure reliable performance metrics, leading to informed conclusions about their effectiveness in classifying images.

## Acknowledgments

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
