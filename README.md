# Machine Learning & Pattern Recognition Notebooks

A comprehensive collection of Jupyter notebooks for exploring machine learning algorithms, clustering techniques, and explainable AI (XAI) methods.

## Project Overview

This project contains implementations and tutorials for various machine learning and pattern recognition techniques, including:

- **Clustering Analysis**: K-Means, DBSCAN, and Gaussian Mixture Models
- **Classification Models**: Decision Trees and Support Vector Machines (SVM)
- **Explainable AI (XAI)**: Counterfactual explanations and model interpretability
- **Real-world Applications**: Credit risk prediction and customer behavior analysis

## Project Structure

```
├── credit_risk_synthetic_prediction - 214211R.ipynb
├── kmeans_dbscan_gmm_analysis - 214211R.ipynb
├── requirements.txt
├── README.md
├── datasets/
│   ├── credit_risk_synthetic.csv
│   └── E-commerce Customer Behavior - Sheet1.csv
└── Lab sheet/
    ├── Clustering_Models_AI.ipynb
    ├── Decision_Tree_Lab.ipynb
    ├── Student_Dropout_Success_XAI_fixed.ipynb
    ├── SVM Lab sheet.ipynb
    └── XAI_Tutorial_DIverse_Counterfactual_Explanations_(DICE).ipynb
```

## Notebooks Description

### Main Projects

- **credit_risk_synthetic_prediction - 214211R.ipynb**: Predictive modeling for credit risk assessment using synthetic datasets
- **kmeans_dbscan_gmm_analysis - 214211R.ipynb**: Comparative analysis of K-Means, DBSCAN, and Gaussian Mixture Models clustering algorithms

### Lab Sheets

- **Clustering_Models_AI.ipynb**: Introduction to clustering techniques and model evaluation
- **Decision_Tree_Lab.ipynb**: Decision tree implementation and hyperparameter tuning
- **SVM Lab sheet.ipynb**: Support Vector Machine algorithms and classification tasks
- **Student_Dropout_Success_XAI_fixed.ipynb**: Predicting student outcomes with explainable AI techniques
- **XAI*Tutorial_DIverse_Counterfactual_Explanations*(DICE).ipynb**: Deep dive into counterfactual explanations using DICE (Diverse Counterfactual Explanations)

## Datasets

Located in the `datasets/` folder:

- **credit_risk_synthetic.csv**: Synthetic credit risk dataset for classification tasks
- **E-commerce Customer Behavior - Sheet1.csv**: Customer behavior data for clustering and analysis

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone/Navigate to the project directory:**

   ```bash
   cd "d:\Work\On\In21-S7-CM4371 - Machine Learning & Pattern Recognition Notbooks"
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Key Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **imbalanced-learn**: Handling imbalanced datasets
- **dice_ml**: Counterfactual explanations
- **lime**: Model interpretability
- **lightgbm**: Gradient boosting framework
- **jupyter**: Interactive notebook environment

For the complete list, see `requirements.txt`

## Getting Started

1. **Activate the virtual environment** (if using one)
2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
3. **Open the desired notebook** from the file browser
4. **Run cells sequentially** or use Shift+Enter to execute individual cells

## Usage Notes

- Each notebook is self-contained and can be run independently
- Datasets are automatically loaded from the `datasets/` folder
- All visualizations and outputs are generated within the notebooks
- Lab sheets include explanatory comments and learning objectives

## Topics Covered

- **Clustering**: K-Means, DBSCAN, GMM, model evaluation metrics
- **Classification**: Decision Trees, SVM, hyperparameter optimization
- **Explainability**: LIME, DICE, feature importance analysis
- **Data Processing**: Preprocessing, scaling, feature engineering
- **Real-world Applications**: Credit risk assessment, customer segmentation, student success prediction

## Course Information

- **Course**: In21-S7-CM4371 - Machine Learning & Pattern Recognition
- **Student ID**: 214211R

## Tips

- Use the lab sheets as tutorials before diving into the main projects
- Experiment with different hyperparameters to understand model behavior
- Review the XAI notebooks to understand how to explain your model predictions
- Check dataset statistics and visualizations before applying algorithms

---

**Last Updated**: February 2026
