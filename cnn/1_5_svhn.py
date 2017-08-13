import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle

image_vector_size = 3072
train_file = "svhn/train_32x32.mat"
test_file = "svhn/test_32x32.mat"

# One hot encoding i.e. 235 => 
# [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def error_rate(p, t):
    return np.mean(p != t)

def flatten(X):
    # Matlab files the rows is in the last dimension
    N = X.shape[-1]
    # Flatten the dimensions to a vector of size 3072
    flat = np.zeros((N, image_vector_size))
    for i in range(N):
        # Return everything except the last column
        flat[i] = X[:,:,:,i].reshape(image_vector_size)
    return flat

def main():
    train = loadmat(train_file)
    test = loadmat(test_file)
    # Flatten the image matrix, with normalised values (divide by 255)
    Xtrain = flatten(train['X'].astype(np.float32)/255)
    # Flatten y labels, and -1, as matlab indices start at 1.
    Ytrain = train['y'].flatten() -1
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    # Flatten the image matrix, with normalised values (divide by 255)
    Xtest = flatten(test['X'].astype(np.float32)/255)
    # Flatten y labels, and -1, as matlab indices start at 1.
    Ytest = test['y'].flatten() -1
    Xtest, Ytest = shuffle(Xtest, Ytest)
    Ytest_ind = y2indicator(Ytest)

    max_iter = 20
    print_period = 10
    N, D = Xtrain.shape
    batch_size = 500
    n_batch = int(N / batch_size)
    M1 = 1000
    M2 = 500
    K = 10
    # Weights and biases
    W1_init = np.random.randn(D, M1)/np.sqrt(D + M1)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2)/np.sqrt(D + M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K)/np.sqrt(D + K)
    b3_init = np.zeros(K)

    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Yish = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(Yish, T))

    train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

    #Calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_size:(j*batch_size + batch_size),]
                Ybatch = Ytrain_ind[j*batch_size:(j*batch_size + batch_size),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    LL.append(test_cost)
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(LL)
    plt.show()

if __name__ == '__main__':
    main()
