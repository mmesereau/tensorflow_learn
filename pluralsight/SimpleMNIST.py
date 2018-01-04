import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#one_hot=True means only return highest probability digit

x = tf.placeholder(tf.float32, shape=[None, 784])
#placeholder for 28x28px image
#none: dimension exists, but we don't know how many items are in the dimension

y_ = tf.placeholder(tf.float32, shape=[None, 10])
#y_ is a 10 element vector representing the probability that the output is each digit i.e. [0.0, 0.0, 0.1, 0.4, 0.1, 0.0, 0.3, 0.0, 0.1, 0.0]

W = tf.Variable(tf.zeros([784, 10]))
#weight: initialized at a 784 by 10 tensor filled with zeros

b = tf.Variable(tf.zeros([10]))
#bias: initialized at a 10 by 1 tensor filled with zeros

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test Accuracy: {0}%".format(test_accuracy * 100.0))

sess.close()
