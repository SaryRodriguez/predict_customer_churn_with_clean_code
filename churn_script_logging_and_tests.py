'''
Module for testing function from churn_library.py

Author: Sara Rodriguez
Date: December 2025
'''

import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("ERROR: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
		logging.info("Testing df shape: SUCCESS")
	except AssertionError as err:
		logging.error("ERROR: The file is empty")
		raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda()
        logging.info("Testing perform_eda: SUCCESS")
    except (AttributeError, SyntaxError) as err:
        logging.error("ERROR: Input should be a dataframe")
        raise err

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]

        encoder_helper(category_lst=cat_columns)
        logging.info("Testing encoder_helper: SUCCESS")

    except KeyError as err:
        logging.error(
            "ERROR: There are column names that doesn't exist"
			"in your dataframe")

    try:
        assert isinstance(cat_columns, list)
        assert len(cat_columns) > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "ERROR: category_lst argument should be a list with length > 0")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''
	try:
		x_train, x_test, y_train, y_test = perform_feature_engineering()

		assert x_train.shape[0] > 0, "x_train should not be empty"
		assert x_test.shape[0] > 0, "x_test should not be empty"
		assert x_train.shape[1] == 19, "x_train should have 19 features"
		assert len(x_train) == len(y_train), "x_train and y_train length mismatch"
		assert len(x_test) == len(y_test), "x_test and y_test length mismatch"

		logging.info("Testing perform_feature_engineering: SUCCESS")

	except (KeyError, AssertionError, AttributeError) as err:
		logging.error(f"Testing perform_feature_engineering: FAILED - {err}")
		raise err


def test_train_models(train_models):
	'''
	test train_models
	'''
	try:
		train_models()
		logging.info("Testing train_models: SUCCESS")
	except MemoryError as err:
		logging.error(
            "Testing train_models: Out of memory while train the models")
		raise err


if __name__ == "__main__":
    MODEL = cls.Model()
    test_import(MODEL.import_data)
    test_eda(MODEL.perform_eda)
    test_encoder_helper(MODEL.encoder_helper)
    test_perform_feature_engineering(MODEL.perform_feature_engineering)
    test_train_models(MODEL.train_models)
