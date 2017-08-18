import matplotlib
matplotlib.use('tkagg')
import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

import sys
import os
file_name = os.path.basename(sys.argv[0])

from sklearn.utils import shuffle
from util import init_weight, get_robert_frost

class SimpleRNN:
    """SimpleRNN"""
    # M = number of hidden neurons
    def __init__(self, D, M, V):
        # dimensionality of word embeddings
        self.D = D
        # hidden layer size
        self.M = M
        # vocabulary size
        self.V = V

    def fit(
            self, X, learning_rate=10e-1,
            mu=0.99, reg=1.0, activation=T.tanh,
            epochs=100, show_fig=False
    ):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V

        # initialise weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        # z  = np.ones(M)

        Wxz = init_weight(D, M)
        Whz = init_weight(M, M)
        # rate variable
        bz = np.ones(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        thX, thY, py_x, prediction = self.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)

        lr = T.scalar('lr')
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        # calculate gradients in one step
        grads = T.grad(cost, self.params)
        # momentum - *0 to get it the same shape as p
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates=[
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]

        # declare theano functions
        # predict
        self.predict_op = theano.function(inputs=[thX], outputs=prediction)
        self.train_op  = theano.function(
            inputs=[thX, thY, lr],
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        n_total = sum((len(sentence) + 1) for sentence in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            # has to be calculated manually
            n_total = 0
            cost = 0
            for j in range(N):
                # problem! many words --> END token are overrepresented
                # result: generated lines will be very short
                # we will try to fix in a later iteration
                # BAD! magic numbers 0 and 1...
                # [0] is the start token
                # we are fixing the above by generating the END sequence 10% of the time
                if np.random.random() < 0.1:
                    input_sequence = [0] + X[j]
                    # [1] is the end token
                    output_sequence = X[j] + [1]
                else:
                    # input sequence is from the start to 2nd last word of X
                    # so that the last word can be predicted
                    input_sequence = [0] + X[j][:-1]
                    output_sequence = X[j]
                # has to be calculated manually
                n_total += len(output_sequence)

                c_out, p_out = self.train_op(input_sequence, output_sequence, learning_rate)
                cost += c_out
                # loop thru every sequence to find if words match
                for pj, xj in zip(p_out, output_sequence):
                    if pj == xj:
                        n_correct += 1
            print("i: ", i, " cost: ", cost, " classfication rate: ", (float(n_correct) / n_total))
            costs.append(cost)
        
        if show_fig:
            plt.plot(costs)
            plt.show()
            plt.plot(costs)
            plt.savefig(file_name+"_cost.png")


    def save(self, filename):
        # passing in each param as a seperate argument
        np.savez(filename, *[p.get_value() for p in self.params])
    
    @staticmethod
    def load(filename, activation):
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wxz = npz['arr_5']
        Whz = npz['arr_6']
        bz = npz['arr_7']
        Wo = npz['arr_8']
        bo = npz['arr_9']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation)
        return rnn

    def set(self, We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo, activation):
        self.f = activation
        # initialise shared theano parameters
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)

        # additional rated unit param
        self.bz = theano.shared(bz)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wxz, self.Whz, self.bz, self.Wo, self.bo]

        thX = T.ivector('X')
        # real X values, word embeddings, by index
        # T x D matrix => D = size of word embeddings, T = sequence
        Ei = self.We[thX]
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            hhat_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
            # additional rated unit
            h_t = (1 - z_t) * h_t1 + z_t * hhat_t
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t
        
        # unravelling the RNN
        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0]
        )
        # output - only the 1st and last dimension
        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        # declare theano functions
        # predict
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[py_x, prediction],
            allow_input_downcast=True
        )
        return thX, thY, py_x, prediction

    def generate(self, word2idx):
        # from { 2197: 'wake'...} to {'wake': 2197 ...}
        idx2word = {v:k for k,v in word2idx.items()}
        # no longer using pi, just sample the softmax output as a probability
        V = len(word2idx)
        # generate 4 lines at a time
        n_lines = 0
        X = [0]
        print (idx2word[X[0]])
        while n_lines < 4:
            # predicted word, X = sequence
            PY_X, _ = self.predict_op(X)
            # only care about the last value
            PY_X = PY_X[-1].flatten()
            # prediction is everything in V, given PY_X
            P = [np.random.choice(V, p=PY_X)]
            # append to the sequence
            X = np.concatenate([X, P])
            # just grab the most recent prediction
            P = P[-1]

            print("PY_X: ", PY_X.shape)
            print("PY_X[P]: ", PY_X[P])
            print("P: ", P)
            print("X: ", X)
            input_seq_sentence = ""
            for word_idx in X:
                input_seq_sentence += idx2word[word_idx] + " "
            print("X sequence: ", input_seq_sentence)
            
            # real word here, as 0 and 1 are the start tokens
            if P > 1:
                word = idx2word[P]
                print(word)
            # end token => start a new line
            elif P == 1:
                n_lines += 1
                # reset X to be the start token
                X = [0]
            keypressed = input('Press q to quit: ')
            if keypressed == 'q':
                break


def train_poetry():
    sentences, word2idx = get_robert_frost()
    # embedding size = 50, and hidden neurons = 50
    rnn = SimpleRNN(50, 50, len(word2idx))
    rnn.fit(sentences, learning_rate=10e-5, show_fig=False, activation=T.nnet.relu, epochs=200)
    rnn.save('RRNN_D50_M50_epochs200_relu.npz')

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RRNN_D50_M50_epochs200_relu.npz', T.nnet.relu)
    rnn.generate(word2idx)
        
if __name__ == "__main__":
    # train_poetry()
    generate_poetry()
