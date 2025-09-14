import numpy as np
import matplotlib.pyplot as plt
import os


class LogisticRegression():
    def __init__(self, n_iterations = 100, learning_rate = 0.01, lambda_ = 1):
        self.n_iteration = n_iterations
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.w = None
        self.b = 0
        self.cost_history = list()


    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __compute_logistic_loss(self, f , y_i):
        """ Compute logistic loss
        f: sigmoid function result
        """
        f = np.clip(f, 1e-15, 1 - 1e-15) # To prevent nan and inf results
        return -(y_i * np.log(f)) - ((1 - y_i) * np.log(1 - f))


    def __compute_cost(self, X_train, y_train):
        m, n = X_train.shape

        cost_without_reg = 0
        for i in range(m):
            z = np.dot(X_train[i, :], self.w) + self.b
            f = self.__sigmoid(z)
            cost_without_reg += self.__compute_logistic_loss(f, y_train[i])

        cost_without_reg /= m

        reg = 0
        for j in range(n):
            reg += self.w[j] ** 2

        reg *= self.lambda_ / (2 * m)
        cost = cost_without_reg + reg

        return cost

    def __compute_gradient_descent(self, X_train, y_train):
        m, n = X_train.shape

        dj_dw = np.zeros((n, ))
        dj_db = 0

        for i in range(m):
            z = np.dot(X_train[i, :], self.w) + self.b
            f = self.__sigmoid(z)
            for j in range(n):
                dj_dw[j] += (f - y_train[i]) * X_train[i, j]

            dj_db += f - y_train[i]

        for j in range(n):
            dj_dw[j] += self.lambda_ * self.w[j]


        dj_dw /= m
        dj_db /= m

        return dj_dw, dj_db

    
    def __initialize_weights(self, X_train, lower, upper):
        _, n = X_train.shape
        self.w = np.ones((n, ))
        for j in range(n):
            self.w[j] += np.random.randint(lower, upper)


    def fit(self, X_train, y_train):
        self.__initialize_weights(X_train, 1, 100)

        for i in range(self.n_iteration):
            cost_i = self.__compute_cost(X_train, y_train)
            self.cost_history.append(cost_i)
            dj_dw, dj_db = self.__compute_gradient_descent(X_train, y_train)

            self.w = self.w - (self.learning_rate * dj_dw)
            self.b = self.b - (self.learning_rate * dj_db)

    def predict(self, X_test):
        m = X_test.shape[0]
        y_hat = np.zeros((m, ))

        for i in range(m):
            z = np.dot(X_test[i, :], self.w) + self.b
            f = self.__sigmoid(z)
            y_hat[i] += 1 if f >= 0.5 else 0


    def plot_learning_curve(self,):
        """ Plot a line plot of cost function / #iterations
        """
        plt.figure(figsize = (8, 8))
        plt.plot(np.arange(self.n_iteration), np.array(self.cost_history))
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost function (Logistic Loss)")
        Figures_dir = os.path.abspath("./Figures/Classification_plots/learning_curve.png")
        plt.savefig(Figures_dir)



if __name__ == "__main__":
    model = LogisticRegression(10000, 0.001)
    X = np.array([[0.005, 0.23, 0.13], [0.012332, 0.12353, 0.35523]])
    y = np.array([1.0, 0.0])
    model.fit(X, y)
    model.plot_learning_curve()
    