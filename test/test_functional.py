import unittest
import os
import sys
import pandas as pd
import numpy as np
import io
from contextlib import redirect_stdout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from test.TestUtils import TestUtils
import fruit
import EmployeePromotion

class TestFruitModel(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = fruit.load_and_preprocess_data()

    def test_fruit_data_columns(self):
        try:
            import os
            import pandas as pd

            # Resolve CSV path
            data_path = os.path.join(os.path.dirname(__file__), "..", "fruit_data.csv")

            # Call the function (ensures it's usable)
            X, y = fruit.load_and_preprocess_data(data_path)

            # Reload raw data to check original columns
            df = pd.read_csv(data_path)
            expected_columns = ["mass", "width", "height", "color_score", "fruit_name"]
            all_columns_present = all(col in df.columns for col in expected_columns)

            if all_columns_present:
                self.test_obj.yakshaAssert("TestFruitDataColumns", True, "functional")
                print("TestFruitDataColumns = Passed")
            else:
                self.test_obj.yakshaAssert("TestFruitDataColumns", False, "functional")
                missing = [col for col in expected_columns if col not in df.columns]
                print(f"TestFruitDataColumns = Failed | Missing columns: {missing}")

        except Exception as e:
            self.test_obj.yakshaAssert("TestFruitDataColumns", False, "functional")
            print(f"TestFruitDataColumns = Failed | Exception: {e}")


    def test_split_data(self):
        try:
            # Load and preprocess sample data
            X, y = fruit.load_and_preprocess_data("fruit_data.csv")

            # Call the function to split the data
            X_train, X_test, y_train, y_test = fruit.split_data(X, y, test_size=0.2)

            # Check that total samples are preserved
            total_samples = len(X)
            expected_test_size = int(total_samples * 0.2)
            expected_train_size = total_samples - expected_test_size

            correct_split = (
                len(X_train) == expected_train_size and
                len(X_test) == expected_test_size and
                len(X_train) == len(y_train) and
                len(X_test) == len(y_test)
            )

            if correct_split:
                self.test_obj.yakshaAssert("TestSplitData_fruit", True, "functional")
                print("TestSplitData_fruit = Passed")
            else:
                self.test_obj.yakshaAssert("TestSplitData_fruit", False, "functional")
                print("TestSplitData_fruit = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestSplitData_fruit", False, "functional")
            print(f"TestSplitData_fruit = Failed | Exception: {e}")
    def test_train_and_save_model(self):
        try:
            import os
            from sklearn.ensemble import RandomForestClassifier

            # Load and split data
            X, y = fruit.load_and_preprocess_data("fruit_data.csv")
            X_train, _, y_train, _ = fruit.split_data(X, y)

            # Use the default model_path: "fruit_rf_model.pkl"
            model_path = "fruit_rf_model.pkl"
            model = fruit.train_and_save_model(X_train, y_train)  # model_path defaults to "fruit_rf_model.pkl"

            # Check that file with exact name exists
            file_exists = os.path.exists(model_path)
            is_correct_name = os.path.basename(model_path) == "fruit_rf_model.pkl"

            is_rf_model = isinstance(model, RandomForestClassifier)
            has_estimators = hasattr(model, "estimators_") and len(model.estimators_) > 0

            if file_exists and is_correct_name and is_rf_model and has_estimators:
                self.test_obj.yakshaAssert("TestTrainAndSaveModel_fruit", True, "functional")
                print("TestTrainAndSaveModel_fruit = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndSaveModel_fruit", False, "functional")
                print("TestTrainAndSaveModel_fruit = Failed")

            # Optional: Keep or clean up
            # os.remove(model_path)

        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndSaveModel_fruit", False, "functional")
            print(f"TestTrainAndSaveModel_fruit = Failed | Exception: {e}")
    def test_load_model(self):
        try:
            import os
            from sklearn.ensemble import RandomForestClassifier

            model_path = "fruit_rf_model.pkl"

            # Ensure the model file exists (you may need to call train_and_save_model first)
            if not os.path.exists(model_path):
                # Optional: create a model file if not already present
                X, y = fruit.load_and_preprocess_data("fruit_data.csv")
                X_train, _, y_train, _ = fruit.split_data(X, y)
                fruit.train_and_save_model(X_train, y_train, model_path=model_path)

            # Load the model
            model = fruit.load_model(model_path)

            # Verify the loaded model
            file_exists = os.path.exists(model_path)
            is_rf_model = isinstance(model, RandomForestClassifier)
            is_trained = hasattr(model, "estimators_") and len(model.estimators_) > 0
            correct_name = os.path.basename(model_path) == "fruit_rf_model.pkl"

            if file_exists and correct_name and is_rf_model and is_trained:
                self.test_obj.yakshaAssert("TestLoadModel", True, "functional")
                print("TestLoadModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadModel", False, "functional")
                print("TestLoadModel = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadModel", False, "functional")
            print(f"TestLoadModel = Failed | Exception: {e}")

    def test_predict_from_json_fruit(self):
        try:
            # Load a trained model
            model = fruit.load_model("fruit_rf_model.pkl")

            # Call the function to predict from JSON
            prediction = fruit.predict_from_json(model, json_file="fruit_item.json")

            # Expected result
            expected_prediction = "orange"

            if prediction == expected_prediction:
                self.test_obj.yakshaAssert("TestPredictFromJson_fruit", True, "functional")
                print("TestPredictFromJson_fruit = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictFromJson_fruit", False, "functional")
                print(f"TestPredictFromJson_fruit = Failed | Expected: {expected_prediction}, Got: {prediction}")

        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictFromJson_fruit", False, "functional")
            print(f"TestPredictFromJson_fruit = Failed | Exception: {e}")
    def test_load_and_preprocess_data(self):
        try:
            # Load data
            df, X, y = EmployeePromotion.load_and_preprocess_data("employee_promotion_dataset.csv")

            # Check types
            is_df = isinstance(df, pd.DataFrame)
            is_X_df = isinstance(X, pd.DataFrame)
            is_y_series = isinstance(y, pd.Series)

            # Check for nulls
            has_no_nulls = df.isnull().sum().sum() == 0

            # Check expected columns before one-hot encoding
            expected_columns = [
                "age", "years_at_company", "last_performance_rating",
                "education_level", "department", "training_hours",
                "gender", "is_manager", "promotion"
            ]
            all_cols_present = all(col in df.columns for col in expected_columns)

            # Check promotion is not in X
            target_removed = "promotion" not in X.columns
            shapes_match = len(df) == len(X) == len(y)

            if all([is_df, is_X_df, is_y_series, has_no_nulls, all_cols_present, target_removed, shapes_match]):
                self.test_obj.yakshaAssert("TestLoadAndPreprocessData", True, "functional")
                print("TestLoadAndPreprocessData = Passed")
            else:
                self.test_obj.yakshaAssert("TestLoadAndPreprocessData", False, "functional")
                print("TestLoadAndPreprocessData = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadAndPreprocessData", False, "functional")
            print(f"TestLoadAndPreprocessData = Failed | Exception: {e}")

    def test_perform_eda(self):
        try:
            # Load and preprocess the data
            df, _, _ = EmployeePromotion.load_and_preprocess_data("employee_promotion_dataset.csv")

            # Call the EDA function
            count = EmployeePromotion.perform_eda(df)

            # Expected number of employees with age > 35
            expected_count = 687 # <-- Update this based on actual data

            if count == expected_count:
                self.test_obj.yakshaAssert("TestPerformEDA", True, "functional")
                print("TestPerformEDA = Passed")
            else:
                self.test_obj.yakshaAssert("TestPerformEDA", False, "functional")
                print(f"TestPerformEDA = Failed | Expected: {expected_count}, Got: {count}")

        except Exception as e:
            self.test_obj.yakshaAssert("TestPerformEDA", False, "functional")
            print(f"TestPerformEDA = Failed | Exception: {e}")
    def test_split_dataemployee(self):
        try:
            # Load data
            _, X, y = EmployeePromotion.load_and_preprocess_data("employee_promotion_dataset.csv")

            # Call the function
            X_train, X_test, y_train, y_test = EmployeePromotion.split_data(X, y, test_size=0.2)

            total_samples = len(X)
            expected_test_size = int(total_samples * 0.2)
            expected_train_size = total_samples - expected_test_size

            correct_split = (
                len(X_train) == expected_train_size and
                len(X_test) == expected_test_size and
                len(y_train) == expected_train_size and
                len(y_test) == expected_test_size
            )

            if correct_split:
                self.test_obj.yakshaAssert("TestSplitData", True, "functional")
                print("TestSplitData = Passed")
            else:
                self.test_obj.yakshaAssert("TestSplitData", False, "functional")
                print("TestSplitData = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestSplitData", False, "functional")
            print(f"TestSplitData = Failed | Exception: {e}")
    def test_train_and_save_modelemployee(self):
        try:
            import os
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler

            # Load and split data
            _, X, y = EmployeePromotion.load_and_preprocess_data("employee_promotion_dataset.csv")
            X_train, _, y_train, _ = EmployeePromotion.split_data(X, y)

            model_path = "promotion_svm_model.pkl"
            model, scaler = EmployeePromotion.train_and_save_model(X_train, y_train, model_path)

            # Checks
            file_exists = os.path.exists(model_path)
            is_model_svc = isinstance(model, SVC)
            is_scaler_standard = isinstance(scaler, StandardScaler)
            model_trained = hasattr(model, "support_")  # Trained SVC has this attribute
            correct_filename = os.path.basename(model_path) == "promotion_svm_model.pkl"

            if all([file_exists, is_model_svc, is_scaler_standard, model_trained, correct_filename]):
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", True, "functional")
                print("TestTrainAndSaveModel = Passed")
            else:
                self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
                print("TestTrainAndSaveModel = Failed")

        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndSaveModel", False, "functional")
            print(f"TestTrainAndSaveModel = Failed | Exception: {e}")

    def test_predict_from_json_employee(self):
        try:
            # Load and preprocess data
            df, X, y = EmployeePromotion.load_and_preprocess_data("employee_promotion_dataset.csv")
            X_train, _, y_train, _ = EmployeePromotion.split_data(X, y)

            # Train model
            model_path = "promotion_svm_model.pkl"
            model, scaler = EmployeePromotion.train_and_save_model(X_train, y_train, model_path=model_path)
            feature_columns = X.columns.tolist()

            # Run prediction from JSON input
            prediction = EmployeePromotion.predict_from_json(model, scaler, json_file="new_employee.json", feature_columns=feature_columns)

            # Check for expected result (0 = Not Promoted)
            if prediction == 0:
                self.test_obj.yakshaAssert("TestPredictFromJson", True, "functional")
                print("TestPredictFromJson = Passed")
            else:
                self.test_obj.yakshaAssert("TestPredictFromJson", False, "functional")
                print(f"TestPredictFromJson = Failed | Expected: 0 (Not Promoted), Got: {prediction}")

        except Exception as e:
            self.test_obj.yakshaAssert("TestPredictFromJson", False, "functional")
            print(f"TestPredictFromJson = Failed | Exception: {e}")
