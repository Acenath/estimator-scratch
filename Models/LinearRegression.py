import numpy as np
import matplotlib.pyplot as plt
import os

class LinearRegression():
    def __init__(self, n_iteration, learning_rate = 0.01, lambda_ = 1):
        """ Simple constructor
        """
        self.n_iteration = n_iteration
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.w = None
        self.b = 0
        self.cost_history = list()

        

    def __compute_cost(self, X_train, y_train):
        """ Compute mean squared error for given train set
        """
        m, n = X_train.shape
        
        cost_without_reg = 0
        for i in range(m):
            f = np.dot(X_train[i, :], self.w) + self.b # w1x1 + w2x2 + w3x3 + .... + b (linear function result)
            cost_without_reg += (f - y_train[i]) ** 2 # mse loss function

        reg = 0 
        for j in range(n):
            reg += self.w[j] ** 2

        reg *= self.lambda_ 
        cost = (cost_without_reg + reg) / (2 * m)

        return cost


    def __compute_gradient_descent(self, X_train, y_train):
        """ Compute partial derivative of one iteration
        """
        m, n = X_train.shape
        dj_dw = np.zeros((n, ))
        dj_db = 0

        for i in range(m):
            f = np.dot(X_train[i, :], self.w) + self.b
            for j in range(n):
                dj_dw[j] += f * X_train[i, j]

            dj_db += f - y_train[i]

        for j in range(n):
            dj_dw[j] += self.lambda_ * self.w[j]

        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db
    
    def __initialize_weights(self, X_train, lower, upper):
        _, n = X_train.shape
        self.w = np.zeros((n, ))
        for j in range(n):
            self.w[j] += np.random.randint(lower, upper)
        
    
    def fit(self, X_train, y_train):
        """ Adjust parameters (w, b)
        """
        self.__initialize_weights(X_train, 1, 100)
        
        for i in range(self.n_iteration):
            cost_i = self.__compute_cost(X_train, y_train)
            self.cost_history.append(cost_i)
            curr_dj_dw, curr_dj_db = self.__compute_gradient_descent(X_train, y_train)

            self.w = self.w - np.dot(self.learning_rate, curr_dj_dw)
            self.b = self.b - (self.learning_rate * curr_dj_db)

        

    def predict(self, X_test):
        """ Predict target values from given dataset
        """
        m, _ = X_test.shape
        y_hat = np.zeros((m, ))
        for i in range(m):
            y_hat[i] += np.dot(X_test[i, :], self.w) + self.b
        
        return y_hat

    def plot_learning_curve(self,):
        """ Plot a line plot of cost function / #iterations
        """
        plt.figure(figsize = (8, 8))
        plt.plot(np.arange(self.n_iteration), np.array(self.cost_history))
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost function (MSE)")
        Figures_dir = os.path.abspath("./Figures/Regression_plots/learning_curve.png")
        plt.savefig(Figures_dir)

    


if __name__ == "__main__":
    model = LinearRegression(100, 0.001)
    X = np.array([[1,2,3], [4,5,6]])
    y = np.array([1, 2])
    model.fit(X, y)
    model.plot_learning_curve()
    pass