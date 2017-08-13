import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import shuffle
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from util import getImageData, error_rate, init_weight_and_bias, init_theano_filter
from ann_theano import HiddenLayer

class ConvPoolLayer(object):
    """Conv. Pool class"""
    # mi = number of input feature maps
    # mo = number of input feature maps
    # fw = feature width
    # fh = feature height
    # poolsz = pool size
    def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2, 2)):
        # theano size tuple 
        sz = (mo, mi, fw, fh)
        W0 = init_theano_filter(sz, poolsz)
        # Initialise it as a shared variable
        self.W = theano.shared(W0)
        # Set the output bias
        b0 = np.zeros(mo, dtype=np.float32)
        # Initialise it as a shared variable
        self.b = theano.shared(b0)
        # Keep a record of the parameters
        self.poolsz = poolsz
        self.params = [self.W, self.b]
    
    def forward(self, X):
        conv_out = conv2d(input=X, filters=self.W)
        pooled_out = pool.pool_2d(
            input=conv_out,
            ws=self.poolsz,
            mode='max',
            ignore_border=True
        )
        return T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes, pool_sz=[(2, 2)]):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes
        self.pool_sz = pool_sz
        if len(convpool_layer_sizes) != len(pool_sz):
            raise ValueError(
                "Length not equal convpool_layer_sizes: ", len(convpool_layer_sizes),
                " pool_sz: ", len(pool_sz)
            )

    # eps = epsilon for RMS prop.
    def fit(self, X, Y, lr=10e-5, mu=0.99, reg=10e-7, decay=0.99999, eps=10e-3, batch_sz=30, epochs=100, show_fig=True):
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)

        # ============= Prep Data =============
        # Validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        # Valid set - last 1000 entries
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        # Training set - Everything except last 1000 entries
        X, Y = X[:-1000], Y[:-1000]

        # ============= Prep ConvPool layers =============
        # initialize convpool layers
        N, c, width, height = X.shape
        mi = c
        outw = width
        outh = height
        self.convpool_layers = []
        # For each parameterised convpool layer
        conv_layer_count = 0
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(mi, mo, fw, fh, self.pool_sz[conv_layer_count])
            # Add layer
            self.convpool_layers.append(layer)
            # Output W after convolution layer
            outw = (outw - fw + 1) // self.pool_sz[conv_layer_count][0]
            outh = (outh - fh + 1) // self.pool_sz[conv_layer_count][1]
            # Set feature input to previous feature output
            # for the next loop
            mi = mo
            conv_layer_count += 1
        # ============= Prep ANN layers =============
        # K = length of all the unique values of Y
        K = len(set(Y))
        # list to store all the hidden layers
        self.hidden_layers = []
        # Output of last convpool layer feature output
        # This is to flatten the last convpool feature output as an input to the ANN
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh # size must be same as output of last convpool layer
        count = 0
        # Loop through the hidden layers in hidden_layer_sizes
        for M2 in self.hidden_layer_sizes:
            # Create hidden layer
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            # Set feature input to previous feature output
            # for the next loop
            M1 = M2
            count += 1
        # ============= Prep Log Regression layer =============
        W, b = init_weight_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')
        # ============= Collect parameters for SGD  =============
        self.params = [self.W, self.b]
        for c in self.convpool_layers:
            self.params += c.params
        for h in self.hidden_layers:
            self.params += h.params
        
        # momentum
        dparams = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
        # rmsprop
        cache = [theano.shared(np.zeros(p.get_value().shape, dtype=np.float32)) for p in self.params]
        # define theano variables - X and Y
        thX = T.tensor4('X', dtype='float32')
        thY = T.ivector('Y')
        # Probability of Y
        pY = self.forward(thX)
        # regularisation cost
        # rcost = reg_parameter*sum(each_parameter^2)
        rcost = reg*T.sum([(p*p).sum() for p in self.params])
        # cost = mean*log(all the relevant targets)
        cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY])) + rcost
        # prediction
        prediction = self.th_predict(thX)
        
        # function to calculate the prediction cost without updates
        # used to calculate cost of prediction for the validation set
        cost_predict_op = theano.function(inputs=[thX, thY], outputs=[cost, prediction])

        # momentum updates
        # momentum only. Update params and dparams
        updates = [
            (p, p + mu*dp - lr*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        ] + [
            (dp, mu*dp - lr*T.grad(cost, p)) for p, dp in zip(self.params, dparams)
        ]

        train_op = theano.function(
            inputs=[thX, thY],
            updates=updates
        )

        n_batches = N // batch_sz
        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                train_op(Xbatch, Ybatch)
                
                if j % 20 == 0:
                    c, p = cost_predict_op(Xvalid, Yvalid)
                    costs.append(c)
                    e = error_rate(Yvalid, p)
                    print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", e)

        if show_fig:
            plt.plot(costs)
            plt.savefig("cost.png")

    def forward(self, X):
        """forward function"""
        Z = X
        for c in self.convpool_layers:
            # Create convpool layer and return tanh(Wx+b)
            Z = c.forward(Z)
        # Flatten Z so that it can be used as an input to ANN
        # N x d dimensional matrix
        Z = Z.flatten(ndim=2)
        # Loop through hidden layers
        for h in self.hidden_layers:
            # Calculate output of each hidden layer
            Z = h.forward(Z)
        # Return the softmax of the ANN output
        return T.nnet.softmax(Z.dot(self.W) + self.b)

    def th_predict(self, X):
        """predict function"""
        pY = self.forward(X)
        return T.argmax(pY, axis=1)

def main():
    """main"""
    # Obtains 4D data for CNN
    X, Y = getImageData()
    model = CNN(
                convpool_layer_sizes=[(32, 5, 5), (64, 5, 5), (128, 5, 5)],
                hidden_layer_sizes=[500, 300],
                pool_sz=[(2, 2), (2, 2), (2, 2)]
            )
    model.fit(X, Y, show_fig=False)


if __name__ == '__main__':
    main()
