from LogisticRegression import LogisticRegression
from LinearRegression import LinearRegression
import numpy as np


class Estimator():
    def __init__(self, n_iteration = 100, learning_rate = 0.01, lambda_ = 1, regressor_on = True):
        self.regressor = LinearRegression(n_iteration, learning_rate, lambda_)
        self.classifier = LogisticRegression(n_iteration, learning_rate, lambda_)
        self.regressor_on = regressor_on
        self.__reset()

    def __reset(self,):
        self.mean = None
        self.std = None

    def __mean_scaling_fit_transform(self, X_train):
        _, n = X_train.shape
        print(n)
        self.__reset()
        self.mean = np.mean(X_train, axis = 0, dtype = np.float64)
        self.std = np.std(X_train, axis = 0, dtype = np.float64)

        for j in range(n):
            X_train[:, j] = (X_train[:, j] - self.mean[j]) / self.std[j]

    def __mean_scaling_transform(self, X_test):
        _, n = X_test.shape

        for j in range(n):
            X_test[:, j] = (X_test[:, j] - self.mean[j]) / self.std[j]

    def fit(self, X_train, y_train):
        self.__mean_scaling_fit_transform(X_train)
        if self.regressor_on:
            self.regressor.fit(X_train, y_train)
        else:
            self.classifier.fit(X_train, y_train)
        

    def predict(self, X_test):
        self.__mean_scaling_transform(X_test)
        y_hat = None

        if self.regressor_on:
            y_hat = self.regressor.predict(X_test)
        else:
            y_hat = self.classifier.predict(X_test)
                    
        return y_hat
    
    def learning_curve(self, ):
        if self.regressor_on:
            self.regressor.plot_learning_curve()
        else:
            self.classifier.plot_learning_curve()



if __name__ == "__main__":
    X = np.array([[1,2,3, 4, 5], [4,5,6, 7, 8,]])
    y = np.array([0, 1, 3, 5, 7])
    model = Estimator()
    model.fit(X, y)
    y_hat = model.predict(X)
    print(y_hat)
    model.learning_curve()
    pass
