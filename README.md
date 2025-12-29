# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Customer churn is the number of existing customers lost, for any reason at all, over a given period of time. It provides companies with an understanding of customer satisfaction and customer loyalty, and can identify potential changes in a company’s bottom line.

This project seeks to answer the customer churn that is happening in the banking industry. The dataset used in this project is obtained from Kaggle.

## Files and data description
Overview of the files and data present in the root directory.

```
├── Guide.ipynb
├── churn_notebook.ipynb
├── churn_library.py
├── churn_script_logging_and_tests.py
├── README.md
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── bivariate
│   │   │   ├── Avg_Open_To_Buy.png
│   │   │   ├── Avg_Utilization_Ratio.png
│   │   │   ├── Card_Category.png
│   │   │   ├── Contacts_Count_12_mon.png
│   │   │   ├── Credit_Limit.png
│   │   │   ├── Customer_Age.png
│   │   │   ├── Dependent_count.png
│   │   │   ├── Education_Level.png
│   │   │   ├── Gender.png
│   │   │   ├── Income_Category.png
│   │   │   ├── Marital_Status.png
│   │   │   ├── Months_Inactive_12_mon.png
│   │   │   ├── Months_on_book.png
│   │   │   ├── Total_Amt_Chng_Q4_Q1.png
│   │   │   ├── Total_Ct_Chng_Q4_Q1.png
│   │   │   ├── Total_Relationship_Count.png
│   │   │   ├── Total_Revolving_Bal.png
│   │   │   ├── Total_Trans_Amt.png
│   │   │   ├── Total_Trans_Ct.png
│   │   ├── multivariate
│   │   │   ├── correlation.png
│   │   ├── univariate
│   │   │   ├── Avg_Open_To_Buy.png
│   │   │   ├── Avg_Utilization_Ratio.png
│   │   │   ├── Card_Category.png
│   │   │   ├── Contacts_Count_12_mon.png
│   │   │   ├── Credit_Limit.png
│   │   │   ├── Customer_Age.png
│   │   │   ├── Dependent_count.png
│   │   │   ├── Education_Level.png
│   │   │   ├── Gender.png
│   │   │   ├── Income_Category.png
│   │   │   ├── Marital_Status.png
│   │   │   ├── Months_Inactive_12_mon.png
│   │   │   ├── Months_on_book.png
│   │   │   ├── Total_Amt_Chng_Q4_Q1.png
│   │   │   ├── Total_Ct_Chng_Q4_Q1.png
│   │   │   ├── Total_Relationship_Count.png
│   │   │   ├── Total_Revolving_Bal.png
│   │   │   ├── Total_Trans_Amt.png
│   │   │   ├── Total_Trans_Ct.png
│   │   ├── Marital_Status.png
│   │   └── Total_Trans_Amt.png
│   └── results
│       ├── feature_importances.png
│       ├── logistic_results.png
│       ├── rf_results.png
│       └── roc_curve_result.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
└── requirements.txt 
```

## Running Files

clone https://github.com/SaryRodriguez/devops_engineer_machine_learning.git

### Dependencies

Here is a list of the libraries used in this project:

```
scikit-learn>=1.3.2
shap>=0.44.0
joblib>=1.3.2
pandas>=2.0.3
numpy>=1.24.3
matplotlib>=3.7.3
seaborn>=0.13.0
pylint>=3.0.3
autopep8>=2.0.4
```
NOTE: my python version is: Python 3.14.2

To be able to run this project, you must install Python library using the following command:

```
python -m pip install -r requirements.txt
```

### Modeling

To run the workflow, simply run the churn_library.py in your terminal using command below:

```
python churn_library.py
```

### Testing and Logging

Run the following command in the terminal to perform testing and logging:

```
python churn_script_logging_and_tests.py
```
