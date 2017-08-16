import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import init_weight, get_poetry_classifier_data

class SimpleRNN:
    """SimpleRNN"""
    # M = number of hidden neurons
    def __init__(self, M, V):
        # hidden layer size
        self.M = M
        # vocabulary size
        self.V = V

    def fit(
        self, X, Y, learning_rate=10e-1,
        mu=0.99, reg=1.0, activation=T.tanh,
        epochs=500, show_fig=False
    ):
        M = self.M
        V = self.V
        K = len(set(Y))
        print("V: ", V)

        X, Y = shuffle(X, Y)
        NValid = 10
        XValid, YValid = X[-NValid:], Y[-NValid:]
        X, Y = X[:-NValid], Y[:-NValid]
        N = len(X)

        # initialise weights (no We embedding)
        Wx = init_weight(V, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        # collect the params into a list
        self.params = [ Wx, Wh, bh, h0, Wo, bo]
        # sequence of word indices
        thX, thY, py_x, prediction = self.set(Wx, Wh, bh, h0, Wo, bo, activation)

        # cross entropy costs
        cost = -T.mean(T.log(py_x[thY]))
        # calculate gradients in one step
        grads = T.grad(cost, self.params)
        # momentum - *0 to get it the same shape as p
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        # adaptive learning rate
        lr = T.scalar('learning_rate')

        updates=[
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]

        self.train_op = theano.function(
            inputs=[thX, thY, lr],
            outputs=[cost, prediction],
            updates=updates,
            allow_input_downcast=True
        )
        
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(N):
                # we set 0 to start and 1 to end
                # print "X[%d]:" % j, X[j], "len:", len(X[j])
                c_out, p_out = self.train_op(X[j], Y[j], learning_rate)
				# print "p:", p, "y:", Y[j]
                cost += c_out
                if p_out == Y[j]:
                    n_correct += 1
				# update the learning rate
                learning_rate *=0.99999

            # calculate validation accuracy
            n_correct_valid = 0
            for j in range(NValid):
                p_out = self.predict_op(XValid[j])
                if p_out == YValid[j]:
                    n_correct_valid +=1
            print("i: ", i, " cost: ", cost, " classfication rate: ", (float(n_correct) / N))
            print("Validation classfication rate: ", (float(n_correct_valid) / NValid))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.savefig("costs.png")
    
    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])
    
    @staticmethod
    def load(filename, activation):
        npz = np.load(filename)
        Wx = npz('arr_0')
        Wh = npz('arr_1')
        bh = npz('arr_2')
        h0 = npz('arr_3')
        Wo = npz('arr_4')
        bo = npz('arr_5')
        V, M = Wx.shape
        rnn = SimpleRNN(M, V)
        rnn.set(Wx, Wh, bh, h0, Wo, bo, activation)
        return rnn

    def set(self, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')
        thY = T.iscalar('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            # indexing Wx with the pos tag vector
            h_t = self.f(x_t.dot(self.Wx[x_t]) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t
        
        # scan the recurrence function
        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps =thX.shape[0]
        )
        
        # probability of y given x
        # get the last element of the sequence,
        # as it is a classification model
        py_x = y[-1, 0, :]
        # do not need to specify the axis as it is 1D
        prediction = T.argmax(py_x)
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True
        )
        return thX, thY, py_x, prediction

def train_poetry():
    X, Y, V = get_poetry_classifier_data(samples_per_class=500)
    # 30 hidden neurons, with a vocabulary size of V
    rnn = SimpleRNN(30, V)
    rnn.fit(X, Y, learning_rate=10e-7, show_fig=False, activation=T.nnet.relu, epochs=1000)

if __name__ == '__main__':
    train_poetry()
            
