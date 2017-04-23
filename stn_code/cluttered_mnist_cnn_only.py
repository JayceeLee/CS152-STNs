# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from spatial_transformer import transformer
import numpy as np
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
import os
import matplotlib.pyplot as plt

# Training Parameters
iter_per_epoch = 50
n_epochs = 2

mainOutdir = os.path.join('output', 'cnn_run1_test')
modelPath = os.path.join(mainOutdir, 'cnn_model')

if not os.path.exists(modelPath):
    os.makedirs(modelPath)

outdirOriginal = os.path.join(mainOutdir, 'originalImages')
outdirModified = os.path.join(mainOutdir, 'modifiedImages')

if not os.path.exists(outdirOriginal):
   os.makedirs(outdirOriginal)

if not os.path.exists(outdirModified):
    os.makedirs(outdirModified)

# Suppress the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load data
# May need to modify this path depending on which file you are running this file from

# Below path works if you are in CS152-STN folder
mnist_cluttered = np.load(os.path.join('stn_code','data','mnist_sequence1_sample_5distortions5x5.npz'))

# Below path works if you are in stn_code folder
# mnist_cluttered = np.load(os.path.join('.','data','mnist_sequence1_sample_5distortions5x5.npz'))


X_train = mnist_cluttered['X_train']
y_train = mnist_cluttered['y_train']
X_valid = mnist_cluttered['X_valid']
y_valid = mnist_cluttered['y_valid']
X_test = mnist_cluttered['X_test']
y_test = mnist_cluttered['y_test']

# % turn from dense to one hot representation
Y_train = dense_to_one_hot(y_train, n_classes=10)
Y_valid = dense_to_one_hot(y_valid, n_classes=10)
Y_test = dense_to_one_hot(y_test, n_classes=10)

# Graph representation of our network

# Placeholders for 40x40 resolution
x = tf.placeholder(tf.float32, [None, 1600])
y = tf.placeholder(tf.float32, [None, 10])

# Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph.  If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant.  Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
x_tensor = tf.reshape(x, [-1, 40, 40, 1])

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
    tf.nn.conv2d(
                #  input=x_tensor,
                 input=x_tensor,
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

# Define loss/eval/training functions
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_logits, labels=y))
opt = tf.train.AdamOptimizer()
optimizer = opt.minimize(cross_entropy)
grads = opt.compute_gradients(cross_entropy, [b_fc_loc2])

# Monitor accuracy
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# We now create a new session to actually perform the initialization the
# variables:
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# We'll now train in minibatches and report accuracy, loss:
train_size = len(X_train)

indices = np.linspace(0, train_size - 1, iter_per_epoch)
indices = indices.astype('int')

for epoch_i in range(n_epochs):
    for iter_i in range(iter_per_epoch - 1):
        batch_xs = X_train[indices[iter_i]:indices[iter_i+1]]
        batch_ys = Y_train[indices[iter_i]:indices[iter_i+1]]

        if iter_i % 10 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict={
                                x: batch_xs,
                                y: batch_ys,
                                keep_prob: 1.0
                            })
            # print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss))

        sess.run(optimizer, feed_dict={
            x: batch_xs, y: batch_ys, keep_prob: 0.8})

    print('Accuracy (%d): ' % epoch_i + str(sess.run(accuracy,
                                                     feed_dict={
                                                         x: X_valid,
                                                         y: Y_valid,
                                                         keep_prob: 1.0
                                                     })))
    theta = sess.run(h_fc_loc2, feed_dict={
           x: batch_xs, keep_prob: 1.0})
print("\n\n\n Final Theta Values for Layer 1:")
print(theta[0])

batch = X_test

batch2 = Y_test

y_actual = y_test.flatten()

# Predictions in the form of [[score_class1 score_class2 ... score_classn] for each sample]
y_pred_1h = sess.run(y_logits, feed_dict={x: batch, y: batch2, keep_prob: 1.0})
index = np.where(y_pred_1h)
y_predict = np.argmax(y_pred_1h, axis=1)

y_actu = pd.Series(y_actual, name='Actual')
y_pred = pd.Series(y_predict, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

print(df_confusion)

print("\ncorrectly predicted %d" % (y_actual == y_predict).sum())
print("out of %d" % len(X_test))
print("\nIndices of wrong images:", str(np.where(y_actual != y_predict)[0].tolist()))

# Save Model
saver.save(sess, os.path.join(modelPath,"model.chk"))

# Original
# make original images (do this once and comment out to save time)
# for i in range(len(X_test)):
#     plt.imshow(np.reshape(X_test[i], (40,40)))
#     plt.savefig(os.path.join(outdirOriginal, str(i) + '.png'), format = "png")


