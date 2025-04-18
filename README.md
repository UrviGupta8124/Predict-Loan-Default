# Predict-Loan-Default

This project aims to classify whether a borrower will default on a loan using financial history and credit scores.

Objective:

To build a machine learning model that helps financial institutions identify risky borrowers and minimize loan default rates.

Features Used:

Credit Score

Annual Income

Employment Status

Loan Amount

Number of Open Accounts

Debt-to-Income Ratio

Delinquency History

Loan Purpose

Age

Technologies:

Python

Pandas, NumPy

Scikit-learn, XGBoost

Matplotlib, Seaborn

Jupyter Notebook

Project Structure:

Predict-Loan-Default/
├── data/               # Datasets (raw and cleaned)
├── notebooks/          # Jupyter notebooks for EDA and model training
├── src/                # Scripts for preprocessing, training, evaluation
├── models/             # Saved model files
├── requirements.txt    # List of dependencies
└── README.md           # Project overview

How to Run:

Clone the repository.

Install the required dependencies using the requirements.txt file.

Upload the dataset and run the notebook or script to process the data and train the model.

Summary of the Code:

Uploaded the dataset and dropped unnecessary or missing data.

Encoded categorical variables using Label Encoding.

Scaled numerical features using StandardScaler.

Split the dataset into training and testing sets.

Trained a Random Forest Classifier on the training data.

Made predictions on the test set.

Evaluated the model using classification metrics such as Accuracy, Precision, Recall, and Confusion Matrix.

Model Evaluation Metrics-

Accuracy: Overall correctness of the model

Precision: Measure of exactness

Recall: Measure of completeness

Confusion Matrix: Visual representation of prediction results

Future Enhancements-

Hyperparameter tuning

Advanced feature engineering

Integration with a frontend for live predictions

Experiment with deep learning models

License-

This project is licensed under the MIT License.

