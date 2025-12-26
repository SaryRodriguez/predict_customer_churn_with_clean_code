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
│   │   ├── Churn.png
│   │   ├── Customer_Age.png
│   │   ├── Gender.png
│   │   ├── heatmap.png
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
autopep8==1.5.7
joblib==0.11
matplotlib==2.1.0
numpy==1.12.1
pandas==0.23.3
pylint==2.9.6
scikit-learn==0.22
seaborn==0.8.1
```

To be able to run this project, you must install Python library using the following command:

```
pip install -r requirements.txt
```

### Modeling

To run the workflow, simply run the churn_library.py in your terminal using command below:

```
python churn_library.py
```

### Testing and Logging

In other conditions, suppose you want to change the configuration of the modeling workflow, such as: changing the path of the data location, adding other models, adding feature engineering stages. You can change it in churn_library.py files. To test if your changes are going well, you need to do testing and logging.

To do testing and logging, you need to change a number of configurations in the churn_script_logging_and_tests.py file, such as: target column name, categorical column name list, data location, etc. After that, run the following command in the terminal to perform testing and loggingAfter that, run the following command in the terminal to perform testing and logging:

```
python churn_script_logging_and_tests.py
```
