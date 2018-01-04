import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Create input object from MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])


#Reshape the images into 28x28 pixel images
#-1: flag saying "make this a list of the other dimensions"
x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1), shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#FIRST CONVOLUTION LAYER
#32 FEATURES FOR EACH 5X5 PATCH OF THE IMAGE
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#do convolution on images, add bias and push through RELU activation

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#take results and run through max_pool_2x2

h_pool1 = max_pool_2x2(h_conv1)

#SECOND CONVOLUTION LAYER
# PROCESS THE 32 FEATURES FROM CONVOLUTION LAYER 1 IN 5X5 PATCH. RETURN 64 FEATURES WEIGHTS AND BIASES

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

#convolution of the output of the first convolution LAYER
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#connect output of pooling layer 2 as input of fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

#define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#loss measurement
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))

#loss Optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
