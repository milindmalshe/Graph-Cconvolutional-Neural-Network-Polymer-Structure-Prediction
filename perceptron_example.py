import numpy as np
import random
import math
import time

random.seed(16)


import matplotlib.pyplot as plt

class create_Data:
    def __init__(self, N=1000, dim = 2, bounds = None):
        if bounds == None:
            bounds = [(-1, 1) for _ in range(dim)]


        self.dim = dim
        self.bounds = bounds
        self.Xdata = []
        self.Ydata = []
        self.w = []
        self.generate_points(N)

    def make_f(self, w = None):
        X = np.asarray(self.Xdata)

        #add a column of ones
        X = np.hstack((X, np.ones((len(X), 1))))


        if w == None:
            w = np.zeros((X.shape[1], 1))

            while (np.matmul(X, w) < 0).sum() < 0.3*len(X) or (np.matmul(X, w) < 0).sum() > 0.7*len(X):

                rand_num = random.randint(0, 2)
                w = ((-1)**(rand_num))*np.random.rand(X.shape[1], 1)

        print "w: ", w
        actual_y = np.matmul(X, w)


        for i in range(0, len(actual_y)):
            actual_y[i] = actual_y[i] + 0.07*((-1)**(random.randint(0, 1)))*random.random()



        actual_y[actual_y>=0] = 1
        actual_y[actual_y<0] = -1

        self.w = w
        self.Ydata = actual_y
        self.Xdata = X

        return self






    def generate_points(self, N):
        for i in range(N):
            point = np.array([random.uniform(self.bounds[j][0], self.bounds[j][1]) for j in range(self.dim)])
            self.Xdata.append(point)



        #add in a column of ones

        self.make_f()

        return self



class perceptron():
    def __init__(self, dim = 2, max_ep=100):
        self.dim = dim
        self.w = np.zeros(dim+1)
        self.done = False
        self.max_epoch = max_ep
        self.iterations = 0


    def predict(self, X_test, learned_w):
        pred_y = np.matmul(X_test, learned_w)
        pred_y[pred_y>=0] = +1
        pred_y[pred_y<0] = -1

        return pred_y

    def train(self, X, Y, lr, eta, count_check):

        dataset = np.hstack((X, Y))
        np.random.shuffle(dataset)


        X_t = dataset[:, 0:X.shape[1]]
        Y_t = dataset[:, -1]
        count = 0
        w_list = []

        for ep in range(self.max_epoch):

            for i in range(0, len(X_t)):
                count += 1

                if Y_t[i]*(np.matmul(X_t[i, :], self.w)) <= eta:
                    self.w = self.w + self.w + lr*Y_t[i]*X_t[i, :]
                    self.w = self.w/(np.linalg.norm(self.w))


                if np.isin(count, count_check):
                    print np.isin(count, count_check)
                    w_list.append(self.w)


            print "Trained w: ", self.w
            #w_list.append(self.w)

        return self, w_list






count_check = [5, 20, 100, 500]

train_data = create_Data(N=100, dim=2, bounds=None)
test_data = create_Data(N=20, dim=2, bounds=None)


perceptron = perceptron(dim=2, max_ep=5)
_, w_list = perceptron.train(train_data.Xdata, train_data.Ydata, 0.1, 0.0, count_check)
print w_list

y_p = perceptron.predict(test_data.Xdata, perceptron.w)
#predict


idx_1 = np.where(train_data.Ydata >=0)[0]
idx_2 = np.where(train_data.Ydata <0)[0]
idx_1e = np.where(y_p>=0)[0]
idx_2e = np.where(y_p < 0)[0]

#co-ordinate data

count = 0

for w in w_list:
    count += 1
    x0= np.linspace(-1, 1, 1000)
    x1 = (-w[0]*x0 - w[2])/w[1]

    plt.plot(train_data.Xdata[idx_1, 0], train_data.Xdata[idx_1, 1], 'ko')
    plt.plot(train_data.Xdata[idx_2, 0], train_data.Xdata[idx_2, 1], 'ro')

    if count == len(w_list):
        plt.plot(x0, x1, 'k-')
        plt.plot(test_data.Xdata[idx_1e, 0], test_data.Xdata[idx_1e, 1], 'ks', markersize=12)
        plt.plot(test_data.Xdata[idx_2e, 0], test_data.Xdata[idx_2e, 1], 'rs', markersize=12)
    plt.xlabel('$x_1$', {'fontsize': 16})
    plt.ylabel('$x_2$', {'fontsize': 16})
    plt.ylim([-1, 1])
    #plt.ylim([-1.1, 1.1])
    filename = 'fig_perceptron' + str(count) + '.eps'
    plt.savefig(filename)
    plt.show()