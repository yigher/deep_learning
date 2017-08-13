import numpy as np
import theano.tensor as T
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import theano

from datetime import datetime
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from scipy.io import loadmat
from sklearn.utils import shuffle

image_vector_size = 3072
train_file = "svhn/train_32x32.mat"
test_file = "svhn/test_32x32.mat"

def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a > 0)

# One hot encoding of 0-9 labels
def y2indicator(y):
    N = len(y)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

# Convolution and Pooling
def convpool(X, w, b, poolsize=(2, 2)):
    conv_out = conv2d(input=X, filters=w)
    # Downsample
    pooled_out = pool.pool_2d(input=conv_out, ws=poolsize, ignore_border=True)
    # hyperbolic tangent where dimshuffle is broadcasting bias
    # return T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
    return relu(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))

# initialise filter with random weights values
def init_filter(shape, poolsize):
    # Fan in plus Fan out
    w = np.random.randn(*shape)/np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:]/np.prod(poolsize)))
    return w.astype(np.float32)

# Convert from matlab to theano
def rearrange(X):
    # Theano expects a (N, C, W, H), but matlab produces (W, H, C, N)
    # Get number of rows from the last matlab shape element 
    N = X.shape[-1]
    out = np.zeros((N, X.shape[-2], X.shape[-3], X.shape[-4]), dtype=np.float32)
    # Each row
    for i in range(N):
        # Each channel
        for j in range(X.shape[-2]):
            # Convert from matlab to theano
            out[i, j, :, :] = X[:, :, j, i]
    return out / 255

def main():
    train = loadmat(train_file)
    test = loadmat(test_file)
    # Need to scale! don't leave as 0..255
    # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calculation
    Xtrain = rearrange(train['X'])
    Ytrain = train['y'].flatten() - 1
    del train
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Ytrain_ind = y2indicator(Ytrain)

    Xtest  = rearrange(test['X'])
    Ytest  = test['y'].flatten() - 1
    del test
    Ytest_ind  = y2indicator(Ytest)

    max_iter = 20
    print_period = 10
    # Learning rate
    lr = np.float32(0.00001)
    # Regulariser
    reg = np.float32(0.01)
    # Momentum
    mu = np.float32(0.99)
    # Number of images
    N = Xtrain.shape[0]
    # Batch size
    batch_size = 500
    # Number of batches
    n_batches = N // batch_size
    # 500 hidden neurons on FF network
    M = 500
    # 10 outputs (one hot encoding of labels)
    K = 10
    # Poolsize
    poolsize = (2, 2)
    
    # Initialise filters
    
    # Layer 1 - CNN
    # after conv will be of dimension 32 - 5 + 1 = 28
    # after downsample 28 / 2 = 14
    W1_shape = (20, 3, 5, 5)
    # Weights 1
    W1_init = init_filter(W1_shape, poolsize)
    # Biase 1
    b1_init = np.zeros(W1_shape[0], dtype=np.float32)
    
    # Layer 2 - CNN
    # after conv will be of dimension 14 - 5 + 1 = 10
    # after downsample 10 / 2 = 5
    W2_shape = (50, W1_shape[0], 5, 5)
    # Weights 2
    W2_init = init_filter(W2_shape, poolsize)
    # Biase 2
    b2_init = np.zeros(W2_shape[0], dtype=np.float32)
    
    # Layer 3 - FF Network. 
    # Flattened W2 layer x Number of Neurons
    W3_init = np.random.randn(W2_shape[0]*W2_shape[2]*W2_shape[3], M) / np.sqrt(W2_shape[0]*W2_shape[2]*W2_shape[3] + M)
    b3_init = np.zeros(M, dtype=np.float32)
    
    # Layer 4
    # 500x10 matrix of weights
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)

    # Theano variables
    # X 4D tensor
    X = T.tensor4('X', dtype='float32')
    Y = T.matrix('T')

    # Shared weights
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
    W3 = theano.shared(W3_init.astype(np.float32), 'W3')
    b3 = theano.shared(b3_init, 'b3')
    W4 = theano.shared(W4_init.astype(np.float32), 'W4')
    b4 = theano.shared(b4_init, 'b4')

    # Momentum variables need to be captured
    dW1 = theano.shared(np.zeros(W1_init.shape, dtype=np.float32), 'dW1')
    db1 = theano.shared(np.zeros(b1_init.shape, dtype=np.float32), 'dW1')
    dW2 = theano.shared(np.zeros(W2_init.shape, dtype=np.float32), 'dW1')
    db2 = theano.shared(np.zeros(b2_init.shape, dtype=np.float32), 'dW1')
    dW3 = theano.shared(np.zeros(W3_init.shape, dtype=np.float32), 'dW1')
    db3 = theano.shared(np.zeros(b3_init.shape, dtype=np.float32), 'dW1')
    dW4 = theano.shared(np.zeros(W4_init.shape, dtype=np.float32), 'dW1')
    db4 = theano.shared(np.zeros(b4_init.shape, dtype=np.float32), 'dW1')

    # Forward pass
    Z1 = convpool(X, W1, b1)
    Z2 = convpool(Z1, W2, b2)
    # FF network. Flattten 4d to 2d as input to NN
    Z3 = relu(Z2.flatten(ndim=2).dot(W3) + b3)
    # Output, with a softmax of FF network output
    pY = T.nnet.softmax(Z3.dot(W4) + b4)

    params = (W1, b1, W2, b2, W3, b3, W4, b4)
    # Regularisation
    reg_cost = reg*np.sum((param*param).sum() for param in params)
    # Cost = softmax + cross entropy + regularisation
    cost = -(Y * T.log(pY)).sum() + reg_cost
    # Prediction
    prediction = T.argmax(pY, axis=1)

    # Weight updates
    update_W1 = W1 + mu*dW1 - lr*T.grad(cost, W1)
    update_b1 = b1 + mu*db1 - lr*T.grad(cost, b1)
    update_W2 = W2 + mu*dW2 - lr*T.grad(cost, W2)
    update_b2 = b2 + mu*db2 - lr*T.grad(cost, b2)
    update_W3 = W3 + mu*dW3 - lr*T.grad(cost, W3)
    update_b3 = b3 + mu*db3 - lr*T.grad(cost, b3)
    update_W4 = W4 + mu*dW4 - lr*T.grad(cost, W4)
    update_b4 = b4 + mu*db4 - lr*T.grad(cost, b4)

    # Update the weight changes
    update_dw1 = mu*dW1 - lr*T.grad(cost, W1)
    update_db1 = mu*db1 - lr*T.grad(cost, b1)
    update_dw2 = mu*dW2 - lr*T.grad(cost, W2)
    update_db2 = mu*db2 - lr*T.grad(cost, b2)
    update_dw3 = mu*dW3 - lr*T.grad(cost, W3)
    update_db3 = mu*db3 - lr*T.grad(cost, b3)
    update_dw4 = mu*dW4 - lr*T.grad(cost, W4)
    update_db4 = mu*db4 - lr*T.grad(cost, b4)

    # Define train function
    train = theano.function(
        inputs=[X, Y],
        updates=[
            (W1, update_W1),
            (b1, update_b1),
            (W2, update_W2),
            (b2, update_b2),
            (W3, update_W3),
            (b3, update_b3),
            (W4, update_W4),
            (b4, update_b4),
            (dW1, update_dw1),
            (db1, update_db1),
            (dW2, update_dw2),
            (db2, update_db2),
            (dW3, update_dw3),
            (db3, update_db3),
            (dW4, update_dw4),
            (db4, update_db4)
        ]
    )
    # Function to get prediction
    get_prediction = theano.function(
        inputs=[X,Y],
        outputs=[cost, prediction]
    )

    t0 = datetime.now()
    LL = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_size:(j*batch_size + batch_size),]
            Ybatch = Ytrain_ind[j*batch_size:(j*batch_size + batch_size),]
            train(Xbatch, Ybatch)
            if j % print_period == 0:
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
                LL.append(cost_val)
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(LL)
    plt.show()

    # Visualising filters
    # visualize W1 (20, 3, 5, 5)
    # 60 5x5 filters in total, as there are 3 input channels
    # 8 by 8 so that the drawing is square
    # m and n determines what we want to draw
    # m and n from 0-8, use them to index the selected square
    W1_val = W1.get_value()
    grid = np.zeros((8*5, 8*5))
    m = 0
    n = 0
    # For each row
    for i in range(W1_shape.shape[0]):
        # And each colour channel
        for i in range(W1_shape.shape[1]):
            filt = W1_val[i, j]
            grid[m*5:(m+1)+5, n*5:(n+1)*5] = filt
            m += 1
            if m >= 8:
                m = 0
                n += 1
    plt.imshow(grod, cmap="gray")
    plt.title("W1")
    plt.show()

    # visualize W2 (50, 20, 5, 5)
    W2_val = W2.get_value()
    grid = np.zeros((32*5, 32*5))
    m = 0
    n = 0
    for i in range(W2_shape.shape[0]):
        for j in range(W2_shape.shape[1]):
            filt = W2_val[i,j]
            grid[m*5:(m+1)*5,n*5:(n+1)*5] = filt
            m += 1
            if m >= 32:
                m = 0
                n += 1
    plt.imshow(grid, cmap='gray')
    plt.title("W2")
    plt.show()

if __name__ == '__main__':
    main()
