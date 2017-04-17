from scipy import ndimage
import tensorflow as tf
from spatial_transformer import transformer
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import numpy as np
import matplotlib.pyplot as plt
import os

print("Using tensorflow version", tf.__version__)

#### Load MNIST Data ####
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
plt.imshow(np.reshape(X_train[15], (40,40)))
plt.show()

#### Create Graph representation of network ####

# Placeholders for 40x40 resolution image
x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 1600])

# Reshape input image into 4D tensor in order to get it to work
# Not sure why it has to be 4D...?
x_tensor = tf.reshape(x, [-1, 40, 40, 1])

### Localization Network ###

# Set up 2-layer localization network to find parameters 
# for affine transformation of the input.

# Start with identify transformation - cast to float32 type and flatten
initial = np.array([[1., 0, 0], [0, 1., 0]]).astype('float32').flatten()

# Create variables for Layer 1
W_fc_loc1 = tf.Variable(tf.zeros([1600, 20])) # weight variable
b_fc_loc1 = tf.Variable(tf.random_normal([20], mean=0.0, stddev=0.01)) # bias variable

# Create variables for Layer 2
W_fc_loc2 = tf.Variable(tf.zeros([20, 6]))
b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

# Create the localization network (Using fully-connect network)

# Output of layer 1: h1 = tanh(X*w1 + b1)
h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)

# Apply dropout layer to reduce overfitting
# h1_drop = dropout(h)
keep_prob = tf.placeholder(tf.float32)
h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)

# Output of layer 2: h = tanh(h1_drop * w2 + b2)
h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

# Create Spatial Transformer Module to identify discriminative patches
# transformer() takes care of Parameterized Sampling Grid and Differentiable Image Sampling
out_size = (40, 40)
h_trans = transformer(x_tensor, h_fc_loc2, out_size)

### ConvNet for Image Classification ###

# We'll setup the first convolutional layer
# Weight matrix is [height x width x input_channels x output_channels]
filter_size = 3
n_filters_1 = 16
W_conv1 = weight_variable([filter_size, filter_size, 1, n_filters_1])

# Bias is [output_channels]
b_conv1 = bias_variable([n_filters_1])

# Now we can build a graph which does the first layer of convolution:
# we define our stride as batch x height x width x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(input=h_trans,
                 filter=W_conv1,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv1)

# And just like the first layer, add additional layers to create
# a deep net
n_filters_2 = 16
W_conv2 = weight_variable([filter_size, filter_size, n_filters_1, n_filters_2])
b_conv2 = bias_variable([n_filters_2])
h_conv2 = tf.nn.relu(
    tf.nn.conv2d(input=h_conv1,
                 filter=W_conv2,
                 strides=[1, 2, 2, 1],
                 padding='SAME') +
    b_conv2)

# We'll now reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 10 * 10 * n_filters_2])

# Create a fully-connected layer:
n_fc = 1024
W_fc1 = weight_variable([10 * 10 * n_filters_2, n_fc])
b_fc1 = bias_variable([n_fc])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# And finally our softmax layer:
W_fc2 = weight_variable([n_fc, 10])
b_fc2 = bias_variable([10])
y_logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



#  Simulate batch
batch = X_train[15]
batch = np.reshape(batch, (1,1600))
# batch = np.append(batch, X_train[0], axis=0)
# batch = np.append(batch, X_train[0], axis=0)
batch2 = X_train[15]
batch2 = np.reshape(batch2, (1,1600))

num_batch = 1

#  Run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
z = sess.run(h_trans, feed_dict={x: batch, y: batch2, keep_prob: 1.0})

plt.imshow(np.reshape(z[0], (40,40)))
plt.show()