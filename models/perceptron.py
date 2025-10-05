import pandas as pd
import numpy as np
from Activation import binary, bipolar
from typing import Any
from tabulate import tabulate
from sklearn.metrics import classification_report
from views.output import print_table

class SingleLayerPerceptron:
    def __init__(self, df: pd.DataFrame, target: str, threshold: float, learning_rate: float = 0.1, init_weight: int = 0, max_epochs: int = 5):
        self.y = df[target]
        self.X = df[[feat for feat in df.columns if feat != target]].to_numpy()
        self.__init_weight = init_weight
        self.__max_epochs = max_epochs
        self.__threshold = threshold
        self.__learning_rate = learning_rate
        self.infer_metadata()

    def infer_metadata(self):
        self.weights = [self.__init_weight] * len(self.X[0])
        self.bias = 0
        target_levels = sorted(self.y.unique())
        if target_levels[0] < 0 and target_levels[-1] > 0:
            self.activation = bipolar
        else:
            self.activation = binary
        print(f"Inferred number of features: {len(self.weights)}")
        print(f"Inferred activation function: {self.activation}")

    def fit(self):
        convergence: bool = False
        num_converged_features: int = 0
        num_features = len(self.X[0])
        num_epochs = self.__max_epochs
        num_records = len(self.X)
        epoch_iter = 1
        input_headers = [f"x{i+1}" for i in range(num_features)]
        delta_headers = [f"Δw{i+1}" for i in range(num_features)]
        weight_headers = [f"w{i+1}" for i in range(num_features)]
        headers = (
            input_headers
            + ["Net input", "Predicted"]
            + delta_headers
            + ["Δbias"]
            + weight_headers
            + ["bias"]
        )
        while not convergence and epoch_iter <= num_epochs:
            print(f"EPOCH {epoch_iter}/{num_epochs}")
            rows_epoch = []
            for i, inputs in enumerate(self.X):
                pros = inputs * self.weights
                yin = pros.sum() + self.bias
                y = self.activation(yin, self.__threshold)
                # check convergence
                if y == self.y[i]:
                    num_converged_features += 1
                # weight updation:
                error = self.y[i] - y
                change_in_weights = self.__learning_rate * error * inputs
                self.weights += change_in_weights
                new_weights_list = np.array(self.weights).tolist()
                change_in_bias = self.__learning_rate * error
                self.bias += change_in_bias
                inputs_list = inputs.tolist()
                change_list = change_in_weights.tolist()

                row = inputs_list + [yin, y] + change_list + [change_in_bias] + new_weights_list + [self.bias]

                rows_epoch.append(row)
            print_table(rows_epoch, headers)

            epoch_iter += 1
            convergence = num_converged_features == num_records
            print("========================")
            print("Number of converged features")
            print(num_converged_features)
            print("========================")
            num_converged_features = 0

    def predict(self, X: np.ndarray):
        """Return predictions for rows in X (numpy array or list-of-lists).
        Uses the same activation and threshold as training.
        """
        arr = np.array(X)
        pros = arr * self.weights
        yin = pros.sum(axis=1) + self.bias
        preds = [self.activation(v, self.__threshold) for v in yin]
        return np.array(preds)

    def classification_report(self, X: np.ndarray, y_true: np.ndarray):
        """Return a classification report string.
        """
        preds = self.predict(X)
        return classification_report(y_true, preds)