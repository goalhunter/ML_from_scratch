import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, lr = 0.01, n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            prediction = np.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (prediction - y))
            db = (1/n_samples) * np.sum(prediction - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db 

        
    def predict(self, X, y):
        predictions = np.dot(X,self.weights) + self.bias
        sigmoid_predictions = sigmoid(predictions)
        predictions = [ 0 if y_pred < 0.5 else 1 for y_pred in sigmoid_predictions]
        return predictions
