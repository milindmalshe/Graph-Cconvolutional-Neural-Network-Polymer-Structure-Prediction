import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Conv2D, Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Input, BatchNormalization, TimeDistributed, Reshape, Lambda
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import concatenate
from keras import losses


from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.constraints import NonNeg

#cross-validation liibrary
from sklearn.model_selection import KFold
from sklearn import preprocessing

import feature_extract_01
import tensorflow as tf

import graph_features

def gcn_01(row_max, ng, feature_max=10):

    input_1 = Input(shape=(row_max, feature_max))
    l_1 = TimeDistributed(Dense(5, activation='relu'))(input_1)
    f_1 = Flatten()(l_1)

    input_2 = Input(shape=(ng,))
    merge_layer = concatenate([f_1,input_2])
    #batch_norm_layer = BatchNormalization()(merge_layer)
    #drop_l = Dropout(rate=0.25)

    #drop_1 = drop_l(merge_layer)
    dense_1 = Dense(10, activation='relu')(merge_layer)
    #dense_1 = Dense(10, activation='relu')(drop_1)
    #dense_1 = Dense(10, activation='relu')(batch_norm_layer)

    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2], outputs=out_1)


    return model_cnn



def gcn_013(row_max, ng, feature_max=10):

    input_1 = Input(shape=(row_max, feature_max))
    l_1 = TimeDistributed(Dense(5, activation='relu'))(input_1)

    print( "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    print( "l_1 shape: ", l_1.shape)
    ##1D convolution:
    l_2 = Conv1D(filters=1, kernel_size=4)(l_1)
    p_2 = MaxPooling1D(pool_size=2)(l_2)

    l_3 = Conv1D(filters=1, kernel_size=3)(p_2)
    p_3 = MaxPooling1D(pool_size=2)(l_3)

    print( "l_3 shape: ", p_3.shape)
    f_1 = Flatten()(p_3)

    input_2 = Input(shape=(ng,))
    merge_layer = concatenate([f_1,input_2])
    #batch_norm_layer = BatchNormalization()(merge_layer)
    #drop_l = Dropout(rate=0.25)

    #drop_1 = drop_l(merge_layer)
    dense_1 = Dense(10, activation='relu')(merge_layer)
    #dense_1 = Dense(10, activation='relu')(drop_1)
    #dense_1 = Dense(10, activation='relu')(batch_norm_layer)

    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2], outputs=out_1)


    return model_cnn



def gcn_02(row_max, ng, h1=10, intv=5, feature_max=10):

    input_1 = Input(shape=(row_max, feature_max))

    l_1 = TimeDistributed(Dense(h1, activation='relu'))(input_1)
    #l_1 = BatchNormalization()(l_1)


    #permutation invariance block
    #m_0 = l_1[:, 0, :]
    m_0 = Lambda(lambda xin: xin[:, 0, :], name='m_0')(l_1)
    m_1 = Lambda(lambda xin: xin[:, 1, :], name='m_1')(l_1)
    m_2 = Lambda(lambda xin: xin[:, 2, :], name='m_2')(l_1)

    m_3 = Lambda(lambda xin: xin[:, 3:6, :], name='m_3')(l_1)
    s_3 = Lambda(lambda xin: K.max(xin, axis=1, keepdims=False))(m_3)
    s_3B = Lambda(lambda xin: K.mean(xin, axis=1, keepdims=False))(m_3)
    m_4 = Lambda(lambda xin: xin[:, 6:, :], name='m_4')(l_1)

    num_i = int((row_max - 6) / intv)
    print( "m_4.shape: ", m_4.shape)
    m_4a = Reshape((num_i, intv, h1))(m_4)
    s_4a = Lambda(lambda xin: K.max(xin, axis=2, keepdims=False), name='s_4a')(m_4a)
    s_4B = Lambda(lambda xin: K.mean(xin, axis=2, keepdims=False), name='s_4B')(m_4a)

    s_4 = Flatten()(s_4a)
    s_4B = Flatten()(s_4B)
    print( "l1 shape: ", l_1.shape)
    print( "m_0 shape:", m_0.shape)

    print( "row_max: ", row_max)
    print( "m_4a shape: ", m_4a.shape)
    print( "s_4 shape: ", s_4.shape)


    print( "***********")
    print( "get shapes of all")
    print( "s_3 shape: ", s_3.shape)

    #f_1 = Flatten()(l_1)

    input_2 = Input(shape=(ng,))
    #merge_layer = concatenate([f_1,input_2])
    merge_layer = concatenate([m_0, m_1, m_2, s_3B, s_4B, input_2])
    #merge_layer = concatenate([m_0, m_1, m_2, s_3, s_3B, s_4, s_4B, input_2])
    #batch_norm_layer = BatchNormalization()(merge_layer)
    drop_l = Dropout(rate=0.25)

    drop_1 = drop_l(merge_layer)
    dense_1 = Dense(10, activation='relu')(merge_layer)
    #dense_1 = BatchNormalization()(dense_1)
    #dense_1 = Dense(10, activation='relu')(batch_norm_layer)
    #dense_1 = Dense(10, activation='relu')(drop_1)
    #dense_1 = Dense(10, activation='relu')(batch_norm_layer)

    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2], outputs=out_1)


    return model_cnn


def gcn_021(path_max, row_max, ng, h1=10, feature_max=10):

    #inputs
    input_1 = Input(shape=(path_max, row_max, feature_max))
    print( "input_1 shape", input_1.shape)
    shared_w = TimeDistributed(Dense(h1, activation='relu'))

    for i in range(0, 100):
        input_x = Lambda(lambda xin: xin[:, i, :, :])(input_1)
        print( "input_x shape: ", input_x.shape)
        l_1 = shared_w(input_x)
        print( "l_1 shape: ", l_1.shape)

        if i == 0:
            l_11 = Reshape((1, row_max, h1))(l_1)
        else:
            l_12 = Reshape((1, row_max, h1))(l_1)
            l_11 = concatenate([l_11, l_12], axis=1)


    print( "l_11 shape: ", l_11.shape)

    l_2 = Lambda(lambda xin: K.mean(xin, axis=1, keepdims=False), name='l_2')(l_11)

    ###downsample along the node axis
    print( "l_2 shape: ", l_2.shape)
    f_1 = Flatten()(l_2)

    print( "f_1 shape: ", f_1.shape)

    input_g = Input(shape=(ng,))
    merge_layer = concatenate([f_1, input_g])
    #batch_norm_layer = BatchNormalization()(merge_layer)
    #drop_l = Dropout(rate=0.25)

    #drop_1 = drop_l(merge_layer)
    dense_1 = Dense(10, activation='relu')(merge_layer)
    #dense_1 = Dense(10, activation='relu')(drop_1)
    #dense_1 = Dense(10, activation='relu')(batch_norm_layer)

    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_g], outputs=out_1)


    return model_cnn



def gcn_022(path_max, row_max, ng, h1=10, intv=5, feature_max=10):

    #inputs
    #inputs
    input_1 = Input(shape=(path_max, row_max, feature_max))
    print( "input_1 shape", input_1.shape)
    shared_w = TimeDistributed(Dense(h1, activation='relu'))

    for i in range(0, 100):
        input_x = Lambda(lambda xin: xin[:, i, :, :])(input_1)
        print( "input_x shape: ", input_x.shape)
        l_1 = shared_w(input_x)
        print( "l_1 shape: ", l_1.shape)

        if i == 0:
            l_11 = Reshape((1, row_max, h1))(l_1)
        else:
            l_12 = Reshape((1, row_max, h1))(l_1)
            l_11 = concatenate([l_11, l_12], axis=1)


    print( "l_11 shape: ", l_11.shape)

    l_2 = Lambda(lambda xin: K.mean(xin, axis=1, keepdims=False), name='l_2')(l_11)

    ###downsample along the node axis
    print( "l_2 shape: ", l_2.shape)

    m_0 = Lambda(lambda xin: xin[:, 0, :], name='m_0')(l_2)
    m_1 = Lambda(lambda xin: xin[:, 1, :], name='m_1')(l_2)
    m_2 = Lambda(lambda xin: xin[:, 2, :], name='m_2')(l_2)

    m_3 = Lambda(lambda xin: xin[:, 3:6, :], name='m_3')(l_2)
    s_3 = Lambda(lambda xin: K.max(xin, axis=1, keepdims=False))(m_3)
    s_3B = Lambda(lambda xin: K.mean(xin, axis=1, keepdims=False))(m_3)
    m_4 = Lambda(lambda xin: xin[:, 6:, :], name='m_4')(l_2)

    num_i = int((row_max - 6) / intv)
    print( "m_4.shape: ", m_4.shape)
    m_4a = Reshape((num_i, intv, h1))(m_4)
    s_4a = Lambda(lambda xin: K.max(xin, axis=2, keepdims=False), name='s_4a')(m_4a)
    s_4B = Lambda(lambda xin: K.mean(xin, axis=2, keepdims=False), name='s_4B')(m_4a)

    s_4 = Flatten()(s_4a)
    s_4B = Flatten()(s_4B)


    input_g = Input(shape=(ng,))
    #merge_layer = concatenate([f_1, input_g])
    merge_layer = concatenate([m_0, m_1, m_2, s_3B, s_4B, input_g])
    #batch_norm_layer = BatchNormalization()(merge_layer)
    #drop_l = Dropout(rate=0.25)

    #drop_1 = drop_l(merge_layer)
    dense_1 = Dense(10, activation='relu')(merge_layer)
    #dense_1 = Dense(10, activation='relu')(drop_1)
    #dense_1 = Dense(10, activation='relu')(batch_norm_layer)

    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_g], outputs=out_1)


    return model_cnn




def gcn_011(row_max, ng, feature_max=10):

    input_1 = Input(shape=(row_max, feature_max))
    input_2 = Input(shape=(row_max, feature_max))


    ###this block of code is purely for the tensorflow instantiation of the same model


    x = tf.placeholder(tf.float32, shape=[None, 2], name='inputs')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='targets')

    net = tf.layers.dense(x, 10, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=tf.nn.relu)




    ###timedistributed add
    I = K.eye(size=row_max)

    z_1 = TimeDistributed(Dense(1, activation='relu'))(input_1)
    z_2 = TimeDistributed(Dense(1, activation='relu'))


    l_1 = TimeDistributed(Dense(5, activation='relu'))(input_1)
    f_1 = Flatten()(l_1)

    input_3 = Input(shape=(ng,))
    merge_layer = concatenate([f_1,input_2])
    #batch_norm_layer = BatchNormalization()(merge_layer)
    drop_l = Dropout(rate=0.25)

    drop_1 = drop_l(merge_layer)
    #dense_1 = Dense(10, activation='relu')(merge_layer)
    dense_1 = Dense(10, activation='relu')(drop_1)
    #dense_1 = Dense(10, activation='relu')(batch_norm_layer)

    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2], outputs=out_1)


    return model_cnn


def gcn_012(Z_1, Z_2, x_g, y_e, feature_max=10, filter_num=2, batch_size=8):



    if len(y_e.shape)<2:
        y_e = y_e.reshape((y_e.shape[0], 1))

    # split the data into training and test set
    x_t1, x_t2, x_gt, y_t, x_v1, x_v2, x_gv, y_v, x_e1, x_e2, x_ge, y_e = graph_features.split_data02(x_1=Z_1,
                                                                                                          x_2=Z_2,
                                                                                                          x_g=x_g,
                                                                                                          y=y_e)
    ######-------------------------
    #train_dataset = zip_data01(Z1=x_t1, Z2=x_t2, Zg=x_gt, Y=y_t)
    #val_dataset = zip_data01(Z1=x_v1, Z2=x_v2, Zg=x_gv, Y=y_v)
    #test_dataset = zip_data01(Z1=x_e1, Z2=x_e2, Zg=x_ge, Y=y_e)

    #need to generalize train_dataset -> validation set
    #iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    #next_element = iterator.get_next()


    ###create graph

    row_max = Z_1.shape[1]
    ng = x_g.shape[1]


    z_1 = tf.placeholder(tf.float32, [None, row_max, feature_max]) #sum
    z_2 = tf.placeholder(tf.float32, [None, row_max, feature_max]) #max
    z_g = tf.placeholder(tf.float32, [None, ng])
    y_label = tf.placeholder(tf.float32, [None, 1])

    gamma_0 = 0.75*np.ones(feature_max, ).astype(np.float32)
    #gamma_0 = np.diag(gamma_0).astype(np.float32)
    #gamma = tf.get_variable("gamma", dtype=tf.float32, shape=[feature_max, feature_max], initializer=tf.random_normal_initializer(mean=0.75, stddev=0.1))
    gamma = tf.get_variable("gamma", dtype=tf.float32, initializer=gamma_0, constraint=lambda t: tf.clip_by_value(t, 0, 1))
    gamma = tf.diag(gamma)

    I = tf.constant(np.eye(feature_max, dtype=np.float32))


    print( "I shape: ", I.shape)
    print( "gamma shape: ",gamma.shape)

    m_1 = tf.matmul(I, gamma)
    m_2 = tf.matmul(I, (1-gamma))

    print( "m_1 shape: ", m_1.shape)
    print( "m_2 shape: ", m_2.shape)
    print( "z_1 shape: ", z_1.shape)
    print( "z_2 shape: ", z_2.shape)

    z = tf.tensordot(z_1, m_1, axes=[[2], [0]]) + tf.tensordot(z_2, m_2, axes=[[2], [0]])
    #z = tf.tensordot(next_element[0], m_1, axes=[[2], [0]]) + tf.tensordot(next_element[1], m_2, axes=[[2], [0]])

    #W0 = tf.get_variable("W0", shape=[feature_max, filter_num], initializer=tf.random_normal_initializer(), dtype=tf.float64)
    #b0 = tf.get_variable("b1", shape=[row_max], initializer=tf.random_normal_initializer())

    #z = tf.cast(z, dtype=tf.float32)

    #H1 = tf.tensordot(z, W0, axes=([2], [0]))
    H1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(filter_num, activation='relu'))(z)

    print( "H1 shape: ", H1.shape)
    H1 = tf.reshape(H1, [-1, H1.shape[1]*H1.shape[2]])


    print( "H1 shape: ", H1.shape)
    print( "z_g shape: ", z_g.shape)
    print( type(Z_1), type(Z_2), type(x_g), type(y_e))

    #H1 = tf.cast(H1, dtype=tf.float64)
    H_g = tf.concat([H1, z_g], axis=1)

    H2 = tf.layers.dense(H_g, 10, activation=tf.nn.relu)
    out = tf.layers.dense(H2, 1, activation=None)

    #out = out([next_element[0], next_element[1], next_element[2]])

    #loss_1 = tf.losses.mean_squared_error(labels=next_element[3], predictions=out)
    loss_1 = tf.losses.mean_squared_error(labels=y_label, predictions=out)
    optimizer= tf.train.AdamOptimizer().minimize(loss_1)

    init = tf.global_variables_initializer()

    ###execute the tensorflow computational graph

    tf.summary.scalar("loss", loss_1)
    merged_summary_op = tf.summary.merge_all()


    #training_init_op = iterator.make_initializer(train_dataset)
    #validation_init_op = iterator.make_initializer(val_dataset)




    with tf.Session() as sess:
        sess.run(init)
        #sess.run(training_init_op)
        file_id = "/tmp/tensorboard-layers-api/gcn_012"
        summary = tf.summary.FileWriter(file_id, graph=tf.get_default_graph())


        for epoch in range(2000):

            print( x_t1.shape, x_t2.shape, x_gt.shape, y_t.shape)

            feed_dict_tr = {z_1: x_t1, z_2: x_t2, z_g:x_gt, y_label:y_t}
            __, loss_val = sess.run([optimizer, loss_1], feed_dict=feed_dict_tr)

            print( "loss_val: ", loss_val)

            feed_dict_val = {z_1: x_v1, z_2: x_v2, z_g: x_gv, y_label: y_v}
            y_val = sess.run([out], feed_dict=feed_dict_val)

            val_err = np.sqrt(((y_val[0].flatten() - y_v.flatten()) ** 2).mean()) / np.sqrt(
                ((y_v.flatten()) ** 2).mean())


            feed_dict_test = {z_1: x_e1, z_2: x_e2, z_g: x_ge, y_label: y_e}
            y_out = sess.run([out], feed_dict=feed_dict_test)

            test_err = np.sqrt(((y_out[0].flatten() - y_e.flatten()) ** 2).mean()) / np.sqrt(
                ((y_e.flatten()) ** 2).mean())


            #dz_1 = tf.data.Dataset.from_tensor_slices(Z_1).batch(batch_size=batch_size)
            #iterator = dz_1.make_one_shot_iterator()
            #next_element = iterator.get_next()
            #val = sess.run(next_element)



            print( "y_out: ", y_out[0].flatten())
            print( "y_e: ", y_e.flatten())
            print( "error: ", np.abs(y_out[0].flatten() - y_e.flatten()))
            print( "val error: ", val_err)
            print( "test error: ", test_err)
            print( sess.run(gamma))









    return None


def fit_model01(X1, Xg, Y, iter_max=3):

    val_thres = 1.00

    save_model = []

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        #model_cnn = CNN_model02B(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn = gcn_01(row_max=X1.shape[1], ng=Xg.shape[1], feature_max=X1.shape[2])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        train_history = model_cnn.fit([X1, Xg], [Y], epochs=50, batch_size=16, callbacks=callbacks, validation_split=0.1)
        #train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=10, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]


        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model



def fit_model012(X1, Xg, Y, iter_max=3):

    val_thres = 1.00

    save_model = []

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        #model_cnn = CNN_model02B(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn = gcn_013(row_max=X1.shape[1], ng=Xg.shape[1], feature_max=X1.shape[2])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]
        train_history = model_cnn.fit([X1, Xg], [Y], epochs=50, batch_size=16, callbacks=callbacks, validation_split=0.1)
        #train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=10, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]


        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model




def fit_model01B(X1, Xg, Y, iter_max=3):

    val_thres = 1.00

    save_model = []

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        #model_cnn = CNN_model02B(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn = gcn_02(row_max=X1.shape[1], ng=Xg.shape[1], feature_max=X1.shape[2])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
        train_history = model_cnn.fit([X1, Xg], [Y], epochs=75, batch_size=16, callbacks=callbacks, validation_split=0.1)
        #train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=10, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]


        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model


def fit_model021(X1, Xg, Y, iter_max=3):

    val_thres = 1.00

    save_model = []

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        #model_cnn = CNN_model02B(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn = gcn_021(path_max=X1.shape[1], row_max=X1.shape[2], ng=Xg.shape[1], feature_max=X1.shape[3])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
        train_history = model_cnn.fit([X1, Xg], [Y], epochs=75, batch_size=16, callbacks=callbacks, validation_split=0.1)
        #train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=10, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]


        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model



def fit_model022(X1, Xg, Y, iter_max=3):

    val_thres = 1.00

    save_model = []

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        #model_cnn = CNN_model02B(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn = gcn_022(path_max=X1.shape[1], row_max=X1.shape[2], ng=Xg.shape[1], feature_max=X1.shape[3])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
        train_history = model_cnn.fit([X1, Xg], [Y], epochs=75, batch_size=16, callbacks=callbacks, validation_split=0.1)
        #train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=10, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]


        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model


###---------------------------------------
def zip_data01(Z1, Z2, Zg, Y, batch_size=8):


    dz_1 = tf.data.Dataset.from_tensor_slices(Z1)
    dz_2 = tf.data.Dataset.from_tensor_slices(Z2)
    dz_g = tf.data.Dataset.from_tensor_slices(Zg)
    dy = tf.data.Dataset.from_tensor_slices(Y)

    zip_data = tf.data.Dataset.zip((dz_1, dz_2, dz_g, dy)).batch(batch_size=batch_size)





    return zip_data
