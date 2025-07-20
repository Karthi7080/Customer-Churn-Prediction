# Telco Customer Churn Prediction Dashboard

This project is an interactive Streamlit dashboard for analyzing and predicting customer churn based on the Telco Customer Churn dataset.

## Features

- Clean and visualize uploadable churn data
- Exploratory data analysis: compact plots & churn rates
- Train and evaluate a Random Forest classifier
- Show feature importance
- Perform optional customer segmentation (K-Means clustering)

## Setup Instructions


1. Install requirements:
pip install -r requirements.txt

2. Run the app:
streamlit run app.py

- Upload your churn CSV file when prompted.
- For more info, see notebook.ipynb.

## Example Data

Sample dataset format:
| gender | SeniorCitizen | Partner | ... | Churn |
|--------|---------------|---------|-----|-------|
| Female | 0             | Yes     | ... | Yes   |

## License

[MIT] or your choice.
