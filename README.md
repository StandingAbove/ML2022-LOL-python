# Machine Learning Project: League of Legends Match Prediction

## Overview

Welcome to our League of Legends Match Prediction project! This project is designed to predict match outcomes in the popular game "League of Legends" using machine learning algorithms. We analyze various in-game statistics to create models that can help determine the winning team.

## Dataset

We use a dataset containing high-diamond ranked matches with essential features that may influence match results. The dataset includes data points such as wards placed, kills, deaths, assists, objectives secured, and many other factors. The dataset is loaded from the following link: [high_diamond_ranked_10min.csv](https://raw.githubusercontent.com/trevorkarn/MLCamp2022/main/high_diamond_ranked_10min.csv).

## Libraries Used

We use several Python libraries for data manipulation, machine learning, and evaluation. The primary libraries employed are:
- Pandas
- NumPy
- Scikit-learn

## Models Explored

1. **K-Nearest Neighbors (KNN) Classifier:** We use the KNN classifier to predict match outcomes based on the nearest neighbors in the feature space.

2. **Multi-layer Perceptron (MLP) Classifier:** MLP is a type of neural network that can learn complex relationships between features and target variables.

3. **Decision Tree Classifier:** We explore decision tree-based models to analyze feature importance and classification accuracy.

4. **Random Forest Classifier:** Random Forest is an ensemble learning technique that combines multiple decision trees for improved performance.

5. **Support Vector Machine (SVM) Classifier:** SVM is employed for binary classification tasks using linear kernels.

## Cross-validation and Model Evaluation

We perform cross-validation to assess the performance of our models and choose the best hyperparameters. Accuracy scores and confusion matrices are used for model evaluation.

## Instructions

To replicate or further explore our project, follow these steps:

1. Clone or download this repository to your local machine.
2. Ensure you have the required libraries installed (Pandas, NumPy, and Scikit-learn).
3. Run the Jupyter Notebook or Python script containing the code provided above.

## Note

Please note that the model's accuracy and performance may vary depending on the dataset, feature selection, and hyperparameters. Feel free to experiment with different models and features to achieve even better results!

## Acknowledgments

We would like to express our gratitude to the developers of the datasets used in this project and the creators of the Python libraries that made this analysis possible.

Let's dive into the world of League of Legends match prediction! 🎮🏆
