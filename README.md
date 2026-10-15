# Credit Card Fraud Detection

A machine learning project that identifies potentially fraudulent credit card transactions using a trained Scikit-learn pipeline and a Streamlit web application.

## Features
- Upload transaction data through a CSV file.
- Predict potentially fraudulent transactions.
- Display prediction summaries and results.
- Download predictions as a CSV file.
- Apply feature scaling through a saved ML pipeline.

## Technologies
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Joblib

## Dataset
Uses the Credit Card Fraud Detection dataset from Kaggle. The features `V1`–`V28` are anonymized PCA-transformed features.

The original dataset is not included in this repository.

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

## Evaluation
Model selection uses cross-validation and metrics appropriate for imbalanced classification. Fraud-class precision, recall, F1-score, and PR-AUC are important because accuracy alone can be misleading.

Add your final validated model results here.

## Limitations
The uploaded CSV must contain the expected transaction features. Predictions on synthetic example data are for testing the application only and do not establish real-world model performance.

## Future Improvements
- Improve minority-class fraud detection.
- Tune the decision threshold.
- Add a confusion matrix and precision-recall curve.
- Deploy the application online.
