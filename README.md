# Stroke Prediction Using Python

A Machine Learning project that predicts the likelihood of stroke using patient health and demographic data. This project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, class imbalance handling using SMOTE, ensemble learning, and model persistence.

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature encoding using:
  - One Hot Encoding
  - Ordinal Encoding
- Multiple ML models comparison
- Handling imbalanced data using SMOTE
- Ensemble learning using Voting Classifier
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Prediction on custom input data
- Model saving using Joblib

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn (SMOTE)
- Joblib

## Project Workflow

1. Load dataset
2. Perform data cleaning
3. Conduct exploratory data analysis (EDA)
4. Preprocess features using encoding techniques
5. Split dataset into training and validation sets
6. Train multiple Machine Learning models
7. Compare model performance
8. Handle class imbalance using SMOTE
9. Apply ensemble learning with Voting Classifier
10. Save trained model for reuse

## Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- LightGBM Classifier
- Voting Classifier (Hard & Soft Voting)

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Folder Structure

```bash
├── stroke-prediction-using-python.ipynb
├── train.csv
├── test.csv
├── sample_submission.csv
├── submission.csv
├── model.joblib
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <repo-name>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```bash
stroke-prediction-using-python.ipynb
```

## Example Prediction

The project also supports prediction on custom patient input data using the trained model.

Example:

```python
single_input = {
    'gender': 'Male',
    'age': 33,
    'hypertension': 0,
    'heart_disease': 0,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 79.53,
    'bmi': 31.10,
    'smoking_status': 'formerly smoked'
}
```

## Future Improvements

- Hyperparameter tuning
- Web app deployment using Flask or Streamlit
- Better feature engineering
- ROC-AUC and Precision-Recall analysis
- Deep learning experimentation

## Author

Adithya C
