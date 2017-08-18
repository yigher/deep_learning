import matplotlib
matplotlib.use('tkagg')
import sys
import os
file_name = os.path.basename(sys.argv[0])
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import json

from datetime import datetime
from sklearn.utils import shuffle
from gru import GRU
from lstm import LSTM
from util import init_weight
from brown import get_sentences_with_word2idx, get_sentences_with_word2idx_limit_vocab

class RNN:
    def __init__(self, D, hidden_layer_sizes, V):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.D = D
        self.V = V
    
    def fit(self, X, learning_rate=10e-5, mu=0.99, epochs=10, show_fig=True,
            activation=T.nnet.relu, RecurrentUnit=GRU, normalize=True
    ):
        D = self.D
        V = self.V
        N = len(X)

        We = init_weight(V, D)
        self.hidden_layers = []
        Mi = D
        for Mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo
        
        Wo = init_weight(Mi, V)
        bo = np.zeros(V)

        self.We = theano.shared(We)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]

        for ru in self.hidden_layers:
            self.params += ru.params
        
        thX = T.ivector('X')
        thY = T.ivector('Y')

        Z = self.We[thX]
        # assign Z to the output of the hidden layer
        for ru in self.hidden_layers:
            Z = ru.output(Z)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)

        prediction = T.argmax(py_x, axis=1)
        # let's return py_x too so we can draw a sample instead
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=[py_x, prediction],
            allow_input_downcast=True
        )

        # cross entropy costs
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        # calculate gradients in one step
        grads = T.grad(cost, self.params)
        # momentum - *0 to get it the same shape as p
        dparams = [theano.shared(p.get_value()*0) for p in self.params]
        # word embeddings
        dWe = theano.shared(self.We.get_value()*0)
        gWe = T.grad(cost, self.We)
        dWe_update = mu*dWe - learning_rate*gWe
        We_update = self.We + dWe_update
        if normalize:
            We_update /= We_update.norm(2)
        
        # updates weight matrices
        updates=[
            (p, p + mu*dp - learning_rate*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - learning_rate*g) for dp, g in zip(dparams, grads)
        ] + [
            (self.We, We_update), (dWe, dWe_update)
        ]

        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates              
        )

        costs = []
        for i in range(epochs):
            t0 = datetime.now()
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0
            for j in range(N):
                if np.random.random() < 0.01 or len(X[j]) <= 1:
                    input_sequence = [0] + X[j]
                    output_sequence = X[j] + [1]
                else:
                    input_sequence = [0] + X[j][:-1]
                    output_sequence = X[j]
                n_total += len(output_sequence)


                try:
                	c_out, p_out = self.train_op(input_sequence, output_sequence)
                	cost += c_out
                except Exception as e:
                    PYX, pred = self.predict_op(input_sequence)
                    print ("input_sequence len:", len(input_sequence))
                    print ("PYX.shape:",PYX.shape)
                    print ("pred.shape:", pred.shape)
                    raise e
                
                # print "p:", p
                cost += c_out
                # print "j:", j, "c:", c/len(X[j]+1)
                for pj, xj in zip(p_out, output_sequence):
                    if pj == xj:
                        n_correct += 1
                if j % 200 == 0:
                    sys.stdout.write("j/N: %d/%d correct rate so far: %f\r" % (j, N, float(n_correct/n_total)))
                    sys.stdout.flush()
            print ("i:", i, "cost:", cost, "correct rate:", (float(n_correct)/n_total), "time for epoch:", (datetime.now() - t0))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()
            plt.plot(costs)
            plt.savefig(file_name+"_cost.png")

def train_corpus(we_file="word_embeddings.npy", w2i_file='corpus_word2idx.json', RecurrentUnit=GRU):
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab()
    print("finished retrieving data")
    print("vocab size: ", len(word2idx), " number of sentences:", len(sentences))

    rnn = RNN(50, [50], len(word2idx))
    rnn.fit(sentences, learning_rate=10e-6, epochs=10, show_fig=True, activation=T.nnet.relu)
   
    np.save(we_file, rnn.We.get_value())
    with open(w2i_file, 'w') as f_out:
        json.dump(word2idx, f_out)
    
def find_analogies(w1, w2, w3, we_file="word_embeddings.npy", w2i_file='corpus_word2idx.json'):
    We = np.load(we_file)
    with open(w2i_file) as f_in:
        word2idx = json.load(f_in)

    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    def dist1(a, b):
        return np.linalg.norm(a - b)

    def dist2(a, b):
        return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

    for dist, name in [(dist1, 'Euclidean'), (dist2, 'cosine')]:
        min_dist = float("inf")
        best_word = ''
        for word, idx in word2idx.items():
            if word not in (w1, w2, w3):
                v1 = We[idx]
                d = dist(v0, v1)
                if d < min_dist:
                    min_dist = d
                    best_word = word
        print("closest match by ", name, " distance: ", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)

if __name__ == "__main__":
    train_corpus()
    find_analogies('king', 'man', 'woman')
    find_analogies('france', 'paris', 'london')
    find_analogies('france', 'paris', 'rome')
    find_analogies('paris', 'france', 'italy')
