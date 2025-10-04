import pandas as pd
import numpy as np
from
class SingleLayerPerceptron:
    def __init__(self, df: pd.DataFrame, target: str,threshold: float, learning_rate: float = 0.1, init_weight : int = 0, max_epochs: int = 5):
        self.y = df[target]
        self.X = df[[feat for feat in df.columns if feat != target]].to_numpy()
        self.__init_weight = init_weight
        self.__max_epochs = max_epochs
        self.__threshold = threshold
        self.__learning_rate = learning_rate
        self.infer_metadata()
        
    def infer_metadata(self):
        self.weights = [ self.__init_weight ] * len(self.X[0])
        self.bias = 0
        target_levels = sorted(self.y.unique())
        if target_levels[0] < 0 and target_levels[-1] > 0:
            self.activation = "bipolar"
        else:
            self.activation = "binary"
        print(f"Inferred number of features: {len(self.weights)}")
        print(f"Inferred activation function: {self.activation}")
    
    def fit(self):
        for inputs in self.X:
            print(f"Inputs: {inputs} ")
            print(f"Weights: {self.weights}")
            print(f"Product: {inputs * self.weights}")
            pros = inputs * self.weights
            yin = pros.sum() + self.bias
            print(f"Yin: {yin}")
            
if __name__ == "__main__":
    data = pd.read_csv("2bitand.csv")
    perceptron = SingleLayerPerceptron(data, "y", 0.1 , 0.1)
    perceptron.fit()
    