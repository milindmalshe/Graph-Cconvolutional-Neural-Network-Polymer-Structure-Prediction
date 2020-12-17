import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print "numpy version: ", np.__version__
print "Tensorflow version: ", tf.__version__

#most common - cuda libraries
sess = tf.Session()

w_np = [[1.0, 2.1, 3.1], [4.0, 1.2, 6.9]]
x_np = [[0.1, 0.2], [0.3, 0.4]]

w = tf.convert_to_tensor(w_np)
x = tf.convert_to_tensor(x_np)

op = tf.matmul(x, w)

print sess.run(op)



def net(x):
    x_shape = x.get_shape().as_list()
    w1 = tf.get_variable("w1", initializer=tf.truncated_normal([x_shape[1], 10], stddev=2.0/float(x_shape[1])))
    b1 = tf.get_variable("b1", initializer=tf.constant(0.0, shape=[1, 10]))
    fc1 = tf.matmul(x, w1) + b1
    fc1 = tf.nn.relu(fc1)

    fc1_shape = fc1.get_shape().as_list
    w2 = tf.get_variable("w2", initializer=tf.truncated_normal([fc1_shape[1], 1], stddev=2.0/float(x_shape[1])))
    b2 = tf.get_variable()