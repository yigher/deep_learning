"""CNN Tensorflow"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from util import getImageData, error_rate, init_weight_and_bias, init_tf_filter, y2indicator
from ann_tf import HiddenLayer




class ConvPoolLayer(object):
    """Conv. Pool class"""
    # mi = number of input feature maps
    # mo = number of input feature maps
    # fw = feature width
    # fh = feature height
    # poolsz = pool size
    def __init__(
            self, mi, mo, fw=5, fh=5,
            strides=[1, 1, 1, 1],
            poolsz=[1, 2, 2, 1],
            pool_strides=[1, 2, 2, 1]
    ):
        # tensorflow format
        # filter width and height, followed by input and output feature maps
        sz = (fw, fh, mi, mo)
        # Weight matrix
        W0 = init_tf_filter(sz, poolsz)
        self.W = tf.Variable(W0)
        # bias is just the feature outputs
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        # Keep track of the pool size
        self.poolsz = poolsz
        # Set strides
        self.strides = strides
        # Pool strides
        self.pool_strides = pool_strides
        # Keep track of the params
        self.params = [self.W, self.b]

    def forward(self, X):
        # W matrix conv layer
        conv_out = tf.nn.conv2d(X, self.W, strides=self.strides, padding='SAME')
        # Automatically does broadcasting of bias
        conv_out = tf.nn.bias_add(conv_out, self.b)
        # Max pooling layer
        pool_out = tf.nn.max_pool(
            conv_out, ksize=self.poolsz, strides=self.pool_strides, padding='SAME'
        )
        # return layer with tanh activation function
        return tf.tanh(pool_out)


class CNN(object):
    """CNN class"""
    def __init__(
            self, convpool_layer_sizes,
            strides=[[1, 1, 1, 1]],
            pool_sz=[[1, 2, 2, 1]],
            pool_strides=[[1, 2, 2, 1]],
            hidden_layer_sizes=[500]
    ):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.strides = strides
        self.pool_sz = pool_sz
        self.pool_strides = pool_strides
        self.hidden_layer_sizes = hidden_layer_sizes
        if len(convpool_layer_sizes) != len(strides):
            raise ValueError(
                "Length not equal convpool_layer_sizes: ", len(convpool_layer_sizes),
                " strides: ", len(strides)
            )
        if len(convpool_layer_sizes) != len(pool_sz):
            raise ValueError(
                "Length not equal convpool_layer_sizes: ", len(convpool_layer_sizes),
                " pool_sz: ", len(pool_sz)
            )
        if len(convpool_layer_sizes) != len(pool_strides):
            raise ValueError(
                "Length not equal convpool_layer_sizes: ", len(convpool_layer_sizes),
                " pool_strides: ", len(pool_strides)
            )

    def fit(
            self, X, Y, lr=10e-4, mu=0.99, reg=10e-4, decay=0.99999,
            eps=10e-3, batch_sz=30, epochs=100, show_fig=True
    ):
        # convert all of the params to float32
        lr = np.float32(lr)
        mu = np.float32(mu)
        reg = np.float32(reg)
        decay = np.float32(decay)
        eps = np.float32(eps)
        # K are the unique values of Y (number of classes)
        K = len(set(Y))
        # ============= Prep Data =============
        # Validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = y2indicator(Y).astype(np.float32)
        # Valid set - last 1000 entries
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        # Training set - Everything except last 1000 entries
        X, Y = X[:-1000], Y[:-1000]
        # Flat version required, so that error can be calculated.
        Yvalid_flat = np.argmax(Yvalid, axis=1)
        # ============= Prep ConvPool layers =============
        # initialise convpool layers
        N, width, height, c_number = X.shape
        # input feature maps
        mi = c_number
        outw = width
        outh = height
        self.convpool_layers = []
        convpool_layer_count = 0
        # create convpool layers
        for mo, fw, fh in self.convpool_layer_sizes:
            layer = ConvPoolLayer(
                mi, mo, fw, fh,
                self.strides[convpool_layer_count],
                self.pool_sz[convpool_layer_count],
                self.pool_strides[convpool_layer_count]
            )
            self.convpool_layers.append(layer)
            outw = outw // self.pool_sz[convpool_layer_count][1]
            outh = outh // self.pool_sz[convpool_layer_count][2]
            mi = mo
            convpool_layer_count += 1
        # ============= Prep ANN layers =============
        # Hidden layers
        self.hidden_layers = []
        M1 = self.convpool_layer_sizes[-1][0]*outw*outh
        hidden_layer_count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, hidden_layer_count)
            self.hidden_layers.append(h)
            M1 = M2
            hidden_layer_count += 1
        # ============= prep log regression layer =============
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W, 'W_logreg')
        self.b = tf.Variable(b, 'b_logreg')
        # ============= collect params =============
        self.params = [self.W, self.b]
        # collect convpool
        for h in self.convpool_layers:
            self.params += h.params
        # collect hidden
        for h in self.hidden_layers:
            self.params += h.params
        # ============= init tensorflow variables =============
        tfX = tf.placeholder(tf.float32, shape=(None, width, height, c_number), name='X')
        tfY = tf.placeholder(tf.float32, shape=(None, K), name='Y')
        # not doing softmax, calculating our own activation function
        act = self.forward(tfX)
        # reg cost - regularisation*sum of L2 loss for every parameter
        rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=act,
                labels=tfY
            )
        ) + rcost
        prediction = self.predict(tfX)
        # ============= init train function =============
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(cost)
        # calculate number of batches
        n_batches = N // batch_sz
        # initialise costs array
        costs = []
        init = tf.global_variables_initializer()
        # ============= init tf session =============
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y[j*batch_sz:(j*batch_sz+batch_sz)]

                    session.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})

                    if j % 20 == 0:
                        # calculate costs
                        c_out = session.run(cost, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        costs.append(c_out)
                        # calculate prediction
                        p = session.run(prediction, feed_dict={tfX: Xvalid, tfY: Yvalid})
                        # calculcate error rate
                        e = error_rate(Yvalid_flat, p)
                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", c_out, "error rate:", e)
        if show_fig:
            plt.plot(costs)
            plt.savefig("cost.png")

    def forward(self, X):
        Z = X
        for c in self.convpool_layers:
            Z = c.forward(Z)

        Z_shape = Z.get_shape().as_list()
        print("Z_shape: ", Z_shape)
        # flatten Z. -1 (as the shape is unknown)
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
        # pass onto hidden layers
        for h in self.hidden_layers:
            Z = h.forward(Z)

        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, 1)


def main():
    """main"""
    # Obtains 4D data for CNN
    X, Y = getImageData()

    # convert to tf [N,w,h,c]
    X = X.transpose((0, 2, 3, 1))
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    model = CNN(
        convpool_layer_sizes=[(32, 5, 5), (64, 5, 5), (128, 5, 5)],
        strides=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        pool_sz=[[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
        pool_strides=[[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1]],
        hidden_layer_sizes=[500, 300]
            )
    model.fit(X, Y, show_fig=False)

if __name__ == '__main__':
    main()
