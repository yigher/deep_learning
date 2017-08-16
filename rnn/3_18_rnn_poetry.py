import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

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
        self.f = activation
        # initialise weights
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)
        # initialise shared theano parameters
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        # collect the params into a list
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]
        # sequence of word indices
        thX = T.ivector('X')
        # real X values, word embeddings, by index
        # T x D matrix => D = size of word embeddings, T = sequence
        Ei = self.We[thX]
        thY = T.ivector('Y')

        # x(t) and h(t-1)
        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
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

        # cross entropy costs
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        # calculate gradients in one step
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
            outputs=[cost, prediction],
            updates=updates
        )

        costs = []
        n_total = sum((len(sentence) + 1) for sentence in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            cost = 0
            for j in range(N):
                # [0] is the start token
                input_sequence = [0] + X[j]
                # [1] is the end token
                output_sequence = X[j] + [1]
                c_out, p_out = self.train_op(input_sequence, output_sequence)
                cost += c_out
                # loop thru every sequence to find if words match
                for pj, xj in zip(p_out, output_sequence):
                    if pj == xj:
                        n_correct += 1
            print("i: ", i, " cost: ", cost, " classfication rate: ", (float(n_correct) / n_total))
            costs.append(cost)
        
        if show_fig:
            plt.plot(costs)
            plt.savefig("costs.png")

    def save(self, filename):
        # passing in each param as a seperate argument
        np.savez(filename, *[p.get_value() for p in self.params])
    
    @staticmethod
    def load(filename, activation):
        npz = np.load(filename)
        We = npz('arr_0')
        Wx = npz('arr_1')
        Wh = npz('arr_2')
        bh = npz('arr_3')
        h0 = npz('arr_4')
        Wo = npz('arr_5')
        bo = npz('arr_6')
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wo, bo, activation)
        return rnn

    def set(self, We, Wx, Wh, bh, h0, Wo, bo, activation):
        self.f = activation
        # initialise shared theano parameters
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        # collect the params into a list
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]
        # sequence of word indices
        thX = T.ivector('X')
        # real X values, word embeddings, by index
        # T x D matrix => D = size of word embeddings, T = sequence
        Ei = self.We[thX]
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
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
            outputs=prediction,
            allow_input_downcast=True
        )

    def generate(self, pi, word2idx):
        # from { 2197: 'wake'...} to {'wake': 2197 ...}
        idx2word = {v:k for k,v in word2idx.items()}
        V = len(pi)
        # generate 4 lines at a time
        n_lines = 0
        X = [np.random.choice(V, p=pi)]
        print (idx2word[X[0]])
        while n_lines < 4:
            # predicted word, X = sequence
            P = self.predict_op(X)[-1]
            X += [P]
            # real word here, as 0 and 1 are the start tokens
            if P > 1:
                word = idx2word[p]
                print(word)
            # end token => start a new line
            elif P == 1:
                n_lines += 1
                if n_lines < 4:
                    # generate the 1st word for the next line
                    X = [np.random.choice(V, p=pi)]
                    print (idx2word[X[0]])

def train_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN(30, 30, len(word2idx))
    rnn.fit(sentences, learning_rate=10e-5, show_fig=False, activation=T.nnet.relu, epochs=2000)
    rnn.save('RNN_D30_M30_epochs200_relu.npz')

def generate_poetry():
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RNN_D30_M30_epochs200_relu.npz', T.nnet.relu)

    V = len(word2idx)
    pi = np.zeros(V)
    # create the word distribution
    for sentence in sentences:
        # get first word's frequency count
        pi[sentence[0]] += 1
    pi /= pi.sum()

    rnn.generate(pi, word2idx)
        
if __name__ == "__main__":
    train_poetry()
    generate_poetry()
