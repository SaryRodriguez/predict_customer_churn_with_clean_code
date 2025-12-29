'''
Library of functions to find customers who are likely to churn.

Author: Sara Rodriguez
Date: December 2025
'''

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay, classification_report
import joblib
os.environ['QT_QPA_PLATFORM']='offscreen'

def classification_report_image(y_true, predictions):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_true: dict with 'train' and 'test' keys containing true labels
            predictions: dict with keys 'train_lr', 'test_lr', 'train_rf', 'test_rf'

    output:
             None
    '''
    # Random forest report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_true['test'], predictions['test_rf'])),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_true['train'], predictions['train_rf'])),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_results.png')
    plt.close()

    # Logistic regression report
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_true['train'], predictions['train_lr'])),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_true['test'], predictions['test_lr'])),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, x_train, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_train.columns[i] for i in indices]
    plt.figure(figsize=(20,5))
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(x_train.shape[1]), importances[indices])
    plt.xticks(range(x_train.shape[1]), names, rotation=90)
    plt.savefig(f'{output_pth}/feature_importances.png')
    plt.close()

class Model:
    '''
    Class model for classification workflow:
    Instances:
        import_data: read data from path
        perform_eda: create univariate, bivariate and multivariate plots
        encoder_helper: function to encoding categorical columns
        perform_feature_engineering: train and test data splitting
        train_models: function to train, evaluate models,
            and create predictions
    '''
    def __init__(self):
        '''
        Class initialization
        '''
        self.df = None
        self.df_encoded = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def import_data(self, pth):
        '''
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                df: pandas dataframe
        '''
        print(f"[INFO] read the data from {pth}")
        df = pd.read_csv(pth)
        df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
        df.drop(['Unnamed: 0','CLIENTNUM','Attrition_Flag'], axis=1, inplace=True)
        self.df = df
        return self.df


    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder
        input:
                df: pandas dataframe

        output:
                None
        '''
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'             
            ]

        quant_columns = [
            'Customer_Age',
            'Dependent_count', 
            'Months_on_book',
            'Total_Relationship_Count', 
            'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 
            'Credit_Limit', 
            'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 
            'Total_Amt_Chng_Q4_Q1', 
            'Total_Trans_Amt',
            'Total_Trans_Ct', 
            'Total_Ct_Chng_Q4_Q1', 
            'Avg_Utilization_Ratio'
            ]

        #Univariate
        for i in quant_columns:
            plt.figure(figsize=(20, 10))
            self.df[i].hist()
            print(f"[INFO] Create Histogram plot of {i} column")
            plt.title(f"{i} Distribution")
            plt.savefig(f'./images/eda/univariate/{i}.png')
            plt.close()
        for i in cat_columns:
            plt.figure(figsize=(20, 10))
            self.df[i].value_counts('normalize').plot(kind='bar')
            print(f"[INFO] Create Bar plot of {i} column")
            plt.title(f"{i} Distribution")
            plt.savefig(f'./images/eda/univariate/{i}.png')
            plt.close()
        #Bivariate
        for i in quant_columns:
            sns.histplot(data=self.df, x=i, hue='Churn',
                stat='density', kde=True, alpha=0.6,
                palette=['blue', 'red'], common_norm=False)
            print(f"[INFO] Create Histogram plot of {i} vs Churn")
            plt.title(f"{i} vs Churn Distribution")
            plt.savefig(f'./images/eda/bivariate/{i}.png')
            plt.close()
        for i in cat_columns:
            self.df.groupby(i)['Churn'].value_counts(normalize=True).unstack().plot(
                kind='bar', stacked=True)
            plt.ylabel('Proportion')
            plt.xlabel(i)
            plt.legend(title='Churn', labels=['No Churn', 'Churn'])
            plt.xticks(rotation=45)
            print(f"[INFO] Create Bar plot of {i} vs Churn")
            plt.title(f"{i} vs Churn Distribution")
            plt.savefig(f'./images/eda/bivariate/{i}.png')
            plt.close()
        #Multivariate
        plt.figure(figsize=(20,10))
        columns = quant_columns + ['Churn']
        sns.heatmap(self.df[columns].corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        print("[INFO] Create Correlation plot")
        plt.savefig('./images/eda/multivariate/correlation.png')
        plt.close()

    def encoder_helper(self, category_lst, response='Churn'):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
                df: pandas dataframe
                category_lst: list of columns that contain categorical features
                response: string of response name [optional argument that could
                be used for naming variables or index y column]

        output:
                df: pandas dataframe with new columns for
        '''
        print("[INFO] Encoding categorical columns")
        self.df_encoded = self.df.copy()

        for category in category_lst:
            category_groups = self.df.groupby(category)[response].mean()
            new_column_name = f'{category}_{response}'
            self.df_encoded[new_column_name] = self.df[category].map(category_groups)
        return self.df_encoded


    def perform_feature_engineering(self):
        '''
        input:
                df: pandas dataframe
                response: string of response name [optional argument that could
                be used for naming variables or index y column]

        output:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''
        print("[INFO] Feature engineering")
        keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
        x = pd.DataFrame()
        x[keep_cols] = self.df_encoded[keep_cols]
        y = self.df_encoded['Churn']
        self.x_train, self.x_test, self.y_train, self.y_test =\
             train_test_split(x, y, test_size = 0.3, random_state = 42)
        return self.x_train, self.x_test, self.y_train, self.y_test


    def train_models(self):
        '''
        train, store model results: images + scores, and store models
        input:
                x_train: X training data
                x_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        print("[INFO] Performing Grid Search")
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=5000)
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['log2', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

        print("[INFO] Training models")
        cv_rfc.fit(self.x_train, self.y_train)
        lrc.fit(self.x_train, self.y_train)

        print("[INFO] Saving models")
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

        print("[INFO] Predictions")
        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.x_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.x_test)
        y_train_preds_lr = lrc.predict(self.x_train)
        y_test_preds_lr = lrc.predict(self.x_test)

        # save roc curve
        print("[INFO] ROC curve")
        plt.figure(figsize=(15,8))
        ax = plt.gca()
        RocCurveDisplay.from_estimator(cv_rfc.best_estimator_,
                                    self.x_test, self.y_test,
                                    ax=ax, alpha=0.8, name='Random Forest')

        RocCurveDisplay.from_estimator(lrc, self.x_test, self.y_test,
                                    ax=ax, alpha=0.8, name='Logistic Regression')
        plt.title('ROC Curve Comparison')
        plt.savefig('./images/results/roc_curve_result.png')
        plt.close()
        # Model report
        print("[INFO] Classification report")
        y_true = {
            'train': self.y_train,
            'test': self.y_test
        }
        predictions = {
            'train_lr': y_train_preds_lr,
            'test_lr': y_test_preds_lr,
            'train_rf': y_train_preds_rf,
            'test_rf': y_test_preds_rf
        }
        classification_report_image(y_true, predictions)
        # feature importance
        print("[INFO] Feature importance plot")
        feature_importance_plot(cv_rfc.best_estimator_, self.x_train,
                                "./images/results")

if __name__ == "__main__":

    CAT_COLUMNS = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'             
        ]

    MODEL = Model()
    MODEL.import_data("data/bank_data.csv")
    MODEL.perform_eda()
    MODEL.encoder_helper(CAT_COLUMNS)
    MODEL.perform_feature_engineering()
    MODEL.train_models()
