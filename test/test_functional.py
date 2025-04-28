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
import fish

class TestFruitModel(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = fruit.load_fruit_data()

    

    def test_preprocessing_separates_target_fruit(self):
        try:
            X, y = fruit.preprocess_fruit_data(self.df)
            if not X.empty and not y.empty and "fruit_name" not in X.columns:
                self.test_obj.yakshaAssert("TestPreprocessingSeparatesTargetFruit", True, "functional")
                print("TestPreprocessingSeparatesTargetFruit = Passed")
            else:
                self.test_obj.yakshaAssert("TestPreprocessingSeparatesTargetFruit", False, "functional")
                print("TestPreprocessingSeparatesTargetFruit = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestPreprocessingSeparatesTargetFruit", False, "functional")
            print("TestPreprocessingSeparatesTargetFruit = Failed")

    def test_split_data_counts_fruit(self):
        try:
            X, y = fruit.preprocess_fruit_data(self.df)
            X_train, X_test, y_train, y_test = fruit.split_fruit_data(X, y)
            if len(X_train) == 800 and len(X_test) == 200:
                self.test_obj.yakshaAssert("TestSplitDataCountsFruit", True, "functional")
                print("TestSplitDataCountsFruit = Passed")
            else:
                self.test_obj.yakshaAssert("TestSplitDataCountsFruit", False, "functional")
                print("TestSplitDataCountsFruit = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestSplitDataCountsFruit", False, "functional")
            print("TestSplitDataCountsFruit = Failed")

    def test_model_trains_and_saves_fruit(self):
        try:
            X, y = fruit.preprocess_fruit_data(self.df)
            X_train, X_test, y_train, y_test = fruit.split_fruit_data(X, y)
            model = fruit.create_model()
            trained_model = fruit.train_model(model, X_train, y_train)
            fruit.save_model(trained_model)
            if os.path.exists("fruit_rf_model.pkl"):
                self.test_obj.yakshaAssert("TestModelTrainsAndSavesFruit", True, "functional")
                print("TestModelTrainsAndSavesFruit = Passed")
            else:
                self.test_obj.yakshaAssert("TestModelTrainsAndSavesFruit", False, "functional")
                print("TestModelTrainsAndSavesFruit = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestModelTrainsAndSavesFruit", False, "functional")
            print("TestModelTrainsAndSavesFruit = Failed")

    def test_json_data_processing_fruit(self):
        try:
            json_file = "fruit_item.json"
            if not os.path.exists(json_file):
                self.test_obj.yakshaAssert("TestJsonDataProcessingFruit", False, "functional")
                print("TestJsonDataProcessingFruit = Failed (JSON file not found)")
                return

            X, y = fruit.preprocess_fruit_data(self.df)
            X_train, X_test, y_train, y_test = fruit.split_fruit_data(X, y)
            model = fruit.create_model()
            trained_model = fruit.train_model(model, X_train, y_train)

            f = io.StringIO()
            with redirect_stdout(f):
                fruit.check_new_data_from_json(trained_model, json_file)
            output = f.getvalue()

            if "Predicted Fruit:" in output:
                self.test_obj.yakshaAssert("TestJsonDataProcessingFruit", True, "functional")
                print("TestJsonDataProcessingFruit = Passed")
            else:
                self.test_obj.yakshaAssert("TestJsonDataProcessingFruit", False, "functional")
                print("TestJsonDataProcessingFruit = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestJsonDataProcessingFruit", False, "functional")
            print("TestJsonDataProcessingFruit = Failed")


class TestFishDiseaseModel(unittest.TestCase):
    def setUp(self):
        self.test_obj = TestUtils()
        self.df = fish.load_fish_disease_data()

    
    def test_eda_on_age_count_fish(self):
        try:
            count = self.df[self.df["Age"] > 1].shape[0]
            if count == 860:
                self.test_obj.yakshaAssert("TestEDAOnAgeCountFish", True, "functional")
                print("TestEDAOnAgeCountFish = Passed")
            else:
                self.test_obj.yakshaAssert("TestEDAOnAgeCountFish", False, "functional")
                print("TestEDAOnAgeCountFish = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestEDAOnAgeCountFish", False, "functional")
            print("TestEDAOnAgeCountFish = Failed")

    def test_preprocessing_separates_target_fish(self):
        try:
            X, y, df_encoded = fish.preprocess_fish_data(self.df)
            if not X.empty and not y.empty and "Disease_Status_Healthy" not in X.columns:
                self.test_obj.yakshaAssert("TestPreprocessingSeparatesTargetFish", True, "functional")
                print("TestPreprocessingSeparatesTargetFish = Passed")
            else:
                self.test_obj.yakshaAssert("TestPreprocessingSeparatesTargetFish", False, "functional")
                print("TestPreprocessingSeparatesTargetFish = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestPreprocessingSeparatesTargetFish", False, "functional")
            print("TestPreprocessingSeparatesTargetFish = Failed")

    def test_split_data_counts_fish(self):
        try:
            X, y, df_encoded = fish.preprocess_fish_data(self.df)
            X_train, X_test, y_train, y_test = fish.split_fish_data(X, y)
            if len(X_train) == 800 and len(X_test) == 200:
                self.test_obj.yakshaAssert("TestSplitDataCountsFish", True, "functional")
                print("TestSplitDataCountsFish = Passed")
            else:
                self.test_obj.yakshaAssert("TestSplitDataCountsFish", False, "functional")
                print("TestSplitDataCountsFish = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestSplitDataCountsFish", False, "functional")
            print("TestSplitDataCountsFish = Failed")

    def test_model_trains_successfully_fish(self):
        try:
            X, y, df_encoded = fish.preprocess_fish_data(self.df)
            X_train, X_test, y_train, y_test = fish.split_fish_data(X, y)
            model = fish.create_and_train_model(X_train, y_train)
            if model and hasattr(model, "predict"):
                self.test_obj.yakshaAssert("TestModelTrainsSuccessfullyFish", True, "functional")
                print("TestModelTrainsSuccessfullyFish = Passed")
            else:
                self.test_obj.yakshaAssert("TestModelTrainsSuccessfullyFish", False, "functional")
                print("TestModelTrainsSuccessfullyFish = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestModelTrainsSuccessfullyFish", False, "functional")
            print("TestModelTrainsSuccessfullyFish = Failed")

    def test_entropy_calculation_fish(self):
        try:
            X, y, df_encoded = fish.preprocess_fish_data(self.df)
            value_counts = y.value_counts(normalize=True)
            entropy = -sum(p * np.log2(p) for p in value_counts if p > 0)

            if np.isclose(entropy, 1.0, atol=0.001):
                self.test_obj.yakshaAssert("TestEntropyCalculationFish", True, "functional")
                print("TestEntropyCalculationFish = Passed")
            else:
                self.test_obj.yakshaAssert("TestEntropyCalculationFish", False, "functional")
                print("TestEntropyCalculationFish = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestEntropyCalculationFish", False, "functional")
            print("TestEntropyCalculationFish = Failed")

    def test_json_data_processing_fish(self):
        try:
            json_file = "fish_data.json"
            if not os.path.exists(json_file):
                self.test_obj.yakshaAssert("TestJsonDataProcessingFish", False, "functional")
                print("TestJsonDataProcessingFish = Failed (JSON file not found)")
                return

            X, y, df_encoded = fish.preprocess_fish_data(self.df)
            X_train, X_test, y_train, y_test = fish.split_fish_data(X, y)
            model = fish.create_and_train_model(X_train, y_train)

            f = io.StringIO()
            with redirect_stdout(f):
                fish.check_new_data_from_json(model, df_encoded, json_file)
            output = f.getvalue()

            if "FINAL FISH DISEASE PREDICTION RESULT" in output:
                self.test_obj.yakshaAssert("TestJsonDataProcessingFish", True, "functional")
                print("TestJsonDataProcessingFish = Passed")
            else:
                self.test_obj.yakshaAssert("TestJsonDataProcessingFish", False, "functional")
                print("TestJsonDataProcessingFish = Failed")
        except Exception:
            self.test_obj.yakshaAssert("TestJsonDataProcessingFish", False, "functional")
            print("TestJsonDataProcessingFish = Failed")
