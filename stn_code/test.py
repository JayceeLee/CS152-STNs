from scipy import ndimage
import tensorflow as tf
from spatial_transformer import transformer
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import numpy as np
import matplotlib.pyplot as plt
import os

print("Using tensorflow version", tf.__version__)

# Load MNIST Data
mnist_cluttered = np.load(os.path.join('stn_code','data','mnist_sequence1_sample_5distortions5x5.npz'))

# Get Train and Test Sets from MNIST array
# Change labels from dense to one hot encoding
X_train = mnist_cluttered['X_train']
y_train = dense_to_one_hot(mnist_cluttered['y_train'], n_classes=10)
X_valid = mnist_cluttered['X_valid']
y_valid = dense_to_one_hot(mnist_cluttered['y_valid'], n_classes=10)
X_test = mnist_cluttered['X_test']
y_test = dense_to_one_hot(mnist_cluttered['y_test'], n_classes=10)

# Show an image to make sure it is working
plt.imshow(np.reshape(X_train[0], (40,40)))
plt.show()

# Create Graph representation of network

# Placeholders for 40x40 resolution image
x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 10])



# Create a batch of three images (1600 x 1200)
# Image retrieved from:
#  https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread(os.path.join('stn_code','data','cat.jpg'))
im = im / 255.
im = im.reshape(1, 1200, 1600, 3)
im = im.astype('float32')

#  Let the output size of the transformer be half the image size.
out_size = (600, 800)

#  Simulate batch
batch = np.append(im, im, axis=0)
batch = np.append(batch, im, axis=0)
num_batch = 3

x = tf.placeholder(tf.float32, [None, 1200, 1600, 3])
x = tf.cast(batch, 'float32')

#  Create localisation network and convolutional layer
with tf.variable_scope('spatial_transformer_0'):

    #  Create a fully-connected layer with 6 output nodes
    n_fc = 6
    W_fc1 = tf.Variable(tf.zeros([1200 * 1600 * 3, n_fc]), name='W_fc1')

    #  Zoom into the image
    initial = np.array([[0.5, 0, 0], [0, 0.5, 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()

    b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
    h_fc1 = tf.matmul(tf.zeros([num_batch, 1200 * 1600 * 3]), W_fc1) + b_fc1
    h_trans = transformer(x, h_fc1, out_size)

#  Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
y = sess.run(h_trans, feed_dict={x: batch})

plt.imshow(y[0])
plt.show()