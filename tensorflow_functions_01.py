import numpy as np
import tensorflow as tf

def cnn_model_01(features, labels, mode, params):


    #X is the input feature sized num_conv N*num_conv* |r| * |atom type|
    batch_num = features["x"].shape[0]
    num_conv = features["x"].shape[1]
    size_r = features["x"].shape[2]
    size_m = features["x"].shape[3] #atom type

    #extract parameters to reshape array
    conv_params = params['conv_params']
    num_x = conv_params[0]
    num_y = conv_params[1]
    num_z = conv_params[2]

    input_layer = tf.reshape(features["x"], [-1, num_conv, size_r, size_m])  # change tensor shape to arrange

    print size_r
    print size_m
    #define weights
    w_0 = tf.Variable(tf.random_normal([int(size_r), int(size_m)], dtype=tf.float64), name='w_0')
    l1 = tf.tensordot(input_layer, w_0, 2)


    #set predictions
    y_pred = l1

    ##Add more to the convnet here

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y_pred)

    print labels.shape
    print y_pred.shape
    #define loss function
    loss = tf.losses.mean_squared_error(labels=labels, predictions=y_pred)  # need to change this

    print y_pred

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=training_op)


    eval_metrics_ops = {"accuracy": tf.metrics.mean_squared_error(labels=labels, predictions=y_pred)}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)




###call a function to run the tensorflow session

def train_cnn(X_t, Y_t):

    #the estimator is a tensorflow class for performing high-level model training, evaluation and intference
    cnn_model = tf.estimator.Estimator(model_fn=cnn_model_01, params={'conv_params': [1, 3, 16]})


    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer) #estimator already initializes variables without explicitly calling for it
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":X_t}, y=Y_t, batch_size=1,
                                                     num_epochs=100, shuffle=True)
        cnn_model.train(input_fn=train_input_fn)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":X_t}, y=Y_t, num_epochs=1,
                                                           shuffle=False)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":X_t}, num_epochs=1, shuffle=False)

        eval_results = cnn_model.evaluate(input_fn=eval_input_fn)
        pred_results = cnn_model.predict(input_fn=predict_input_fn)
        print(eval_results)
        #print(list(pred_results))

    return eval_results