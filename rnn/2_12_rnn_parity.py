import matplotlib
matplotlib.use('tkagg')
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import sys
import os
file_name = os.path.basename(sys.argv[0])

from util import init_weight, all_parity_pairs_with_sequence_labels
from sklearn.utils import shuffle


class SimpleRNN:
    """SimpleRNN"""
    # M = number of hidden neurons
    def __init__(self, M):
        self.M = M
    
    def fit(
        self, X, Y, learning_rate=10e-1,
        mu=0.99, reg=1.0, activation=T.tanh,
        epochs=100, show_fig=True
    ):
        D = X[0].shape[1]
        # Y needs to be flattened from 2D to 1D 
        # so that it is converted into sequences
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.f_activation = activation

        # Initialise weights
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)
        # expose them as theano shared variables
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        # store them in a list for access convenience
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        # Define theano inputs and outputs
        thX = T.fmatrix('X')
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f_activation(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps=thX.shape[0],
        )
        # predict output, we want 1st and last dimensions
        # probability of y given x
        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        # momentum - *0 to get it the same shape as p
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates=[
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ]
        # declare theano functions
        # predict
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        # train - output of y so that the shape can be observed 
        # re: py_x = y[:, 0, :] and [h, y] = theano.scan
        self.train_op  = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction, y],
            updates=updates
        )

        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(N):
                c_out, p_out, r_out = self.train_op(X[j], Y[j])
                cost += c_out
                if p_out[-1] == Y[j, -1]:
                    n_correct += 1
            
            print("shape y: ", r_out.shape)
            print("i: ", i, " cost: ", cost, " classfication rate: ", (float(n_correct) / N))
            costs.append(cost)
        
        if show_fig:
            plt.plot(costs)
            plt.show()
            plt.plot(costs)
            plt.savefig(file_name+"_cost.png")

def parity(B=12, learning_rate=10e-5, epochs=200):
    X, Y = all_parity_pairs_with_sequence_labels(B)
    rnn = SimpleRNN(4)
    rnn.fit(X, Y, learning_rate=learning_rate,
        epochs=epochs, activation=T.nnet.sigmoid, show_fig=False
    )

if __name__ == "__main__":
    parity()