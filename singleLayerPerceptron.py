import pandas as pd
import numpy as np
from Activation import binary, bipolar, sigmoid, threshold_fn
from typing import Any, Callable
from tabulate import tabulate
from sklearn.metrics import classification_report

class SingleLayerPerceptron:
    def __init__(self, df: pd.DataFrame, target: str, threshold: float, learning_rate: float = 0.1, init_weight: int = 0, max_epochs: int = 5, activation: str = "binary", verbose: bool = True):
        self.y = df[target]
        self.X = df[[feat for feat in df.columns if feat != target]].to_numpy()
        self.__init_weight = init_weight
        self.__max_epochs = max_epochs
        self.__threshold = threshold
        self.__learning_rate = learning_rate
        self.__activation_name = activation
        self.__verbose = verbose
        self.infer_metadata()
        
    def infer_metadata(self):
        self.weights = [ self.__init_weight ] * len(self.X[0])
        self.bias = 0
        target_levels = sorted(self.y.unique())
        
        # Infer target mode (bipolar or binary)
        if target_levels[0] < 0 and target_levels[-1] > 0:
            self.target_mode = "bipolar"
        else:
            self.target_mode = "binary"
        
        # Store activation name for pickling
        self.activation_func = self.__activation_name
        
        if self.__verbose:
            print(f"Inferred number of features: {len(self.weights)}")
            print(f"Inferred target mode: {self.target_mode}")
            print(f"Selected activation function: {self.__activation_name}")
    
    def _apply_activation(self, x: float, t: float) -> float:
        """Apply the selected activation function to input x with threshold t."""
        if self.__activation_name == "sigmoid":
            return sigmoid(x, self.target_mode)
        elif self.__activation_name == "threshold":
            return threshold_fn(x, t, self.target_mode)
        else:  # binary or bipolar
            return bipolar(x, t) if self.target_mode == "bipolar" else binary(x, t)
    
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
            if self.__verbose:
                print(f"EPOCH {epoch_iter}/{num_epochs}")
            rows_epoch = []
            for i, inputs in enumerate(self.X):
                pros = inputs * self.weights
                yin = pros.sum() + self.bias
                y = self._apply_activation(yin, self.__threshold)
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
            if self.__verbose:
                print(tabulate(rows_epoch, headers=headers, tablefmt="simple", stralign="center"))
                print("=" * 80)
            else:
                # Log epoch completion even in silent mode (for evaluation with suppressed output)
                print(f"[TRAIN] Epoch {epoch_iter}/{num_epochs} completed")

            epoch_iter += 1
            convergence = num_converged_features == num_records
            num_converged_features = 0
            
    def predict(self, X: np.ndarray):
        """Return predictions for rows in X (numpy array or list-of-lists).
        Uses the same activation and threshold as training.
        """
        arr = np.array(X)
        pros = arr * self.weights
        yin = pros.sum(axis=1) + self.bias
        preds = [self._apply_activation(v, self.__threshold) for v in yin]
        return np.array(preds)

    def classification_report(self, X: np.ndarray, y_true: np.ndarray):
        """Return a classification report string.
        """
        preds = self.predict(X)
        return classification_report(y_true, preds)
        
if __name__ == "__main__":
    data = pd.read_csv("bipolar2bitand.csv")
    perceptron = SingleLayerPerceptron(data, "y", threshold = 0 , learning_rate = 1, max_epochs = 5)
    perceptron.fit()
    