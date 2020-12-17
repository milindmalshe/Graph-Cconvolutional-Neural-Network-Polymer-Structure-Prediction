from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np


#tf.logging.set_verbosity(tf.logging.INFO)



def cnn_model_fn(features, labels, mode):

    #input layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) #change tensor shape to arrange

    #convolutional layer
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

    #maxpool layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #the combination of pool size and strides

    #convolution layer 2 and pooling layer 2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    #Dense Layer
    pools2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense1 = tf.layers.dense(inputs=pools2_flat, units=1024, activation=tf.nn.relu) #units -> number of units in the dense layer

    #dropout regularization
    dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    #final layer:
    logits = tf.layers.dense(inputs=dropout, units=10)
    #the logits generates 10 values that indicate which among the 10 digits it is

    #compile predictions in a dict
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #calculating loss

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) #need to change this


    #configure the training op
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)


    #add evaluation metrics
    eval_metrics_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)



def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    print(train_data.shape)
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    print(train_labels.shape)
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    #create the estimator
    #the estimator is a tensorflow class for performing high-level model training, evaluation and intference
    mnist_classifer = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    #setup logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    #train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":train_data}, y=train_labels, batch_size=100,
                                                        num_epochs=None, shuffle=True)
    mnist_classifer.train(input_fn=train_input_fn, steps=1000, hooks=[logging_hook])

    #Evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifer.evaluate(input_fn=eval_input_fn)
    test_results = mnist_classifer.predict(input_fn=eval_input_fn)
    print(list(test_results))



if __name__ == "__main__":
  tf.app.run()
