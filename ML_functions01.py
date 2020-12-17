import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling3D,Input, BatchNormalization
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import concatenate
from keras import losses

from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

#cross-validation liibrary
from sklearn.model_selection import KFold


import feature_extract_01




def CNN_model01(nz, nx, channels):

    #shared CNN weights
    conv_1 = Conv2D(4, kernel_size=1, activation='relu')
    conv_2 = Conv2D(4, kernel_size=2, activation='relu')

    #conv_1 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')
    #conv_2 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')


    #first input model
    input_1 = Input(shape=(nz, nx, channels))
    conv_11 = conv_1(input_1)
    pool_11 = MaxPool2D(pool_size=(2, 2))(conv_11)
    conv_12 = conv_2(pool_11)
    #pool_12 = MaxPool2D(pool_size=(1, 1))(conv_12)
    #f1 = Flatten()(pool_12)
    f1 = Flatten()(conv_12)

    #second input model
    input_2 = Input(shape=(nz, nx, channels))
    conv_21 = conv_1(input_2)
    pool_21 = MaxPool2D(pool_size=(2,2))(conv_21)
    conv_22 = conv_2(pool_21)
    #pool_22 = MaxPool2D(pool_size=(1, 1))(conv_22)
    #f2 = Flatten()(pool_22)
    f2 = Flatten()(conv_22)

    #merge two layers and add dense
    merge_layer = concatenate([f1, f2])
    dense_1 = Dense(40, activation='relu')(merge_layer)
    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2], outputs=out_1)

    print( model_cnn.summary())

    return model_cnn



def CNN_model02(nz, nx, channels):

    #shared CNN weights
    conv_1 = Conv2D(10, kernel_size=1, activation='relu')
    conv_2 = Conv2D(4, kernel_size=2, activation='relu')

    #conv_1 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')
    #conv_2 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')


    #first input model
    input_1 = Input(shape=(nz, nx, channels))
    conv_11 = conv_1(input_1)
    pool_11 = MaxPool2D(pool_size=(2, 2))(conv_11)
    conv_12 = conv_2(pool_11)
    #pool_12 = MaxPool2D(pool_size=(1, 1))(conv_12)
    #f1 = Flatten()(pool_12)
    f1 = Flatten()(conv_12)

    #second input model
    input_2 = Input(shape=(nz, nx, channels))
    conv_21 = conv_1(input_2)
    pool_21 = MaxPool2D(pool_size=(2,2))(conv_21)
    conv_22 = conv_2(pool_21)
    #pool_22 = MaxPool2D(pool_size=(1, 1))(conv_22)
    #f2 = Flatten()(pool_22)
    f2 = Flatten()(conv_22)

    #merge two layers and add dense
    merge_layer = concatenate([f1, f2])
    dense_1 = Dense(20, activation='relu')(merge_layer)
    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2], outputs=out_1)

    print( model_cnn.summary())

    return model_cnn


def CNN_model02B(nz, nx, channels, ng):

    #CNN to concatenate global features
    #ng is the number of global features

    #shared CNN weights
    conv_1 = Conv2D(10, kernel_size=1, activation='relu')
    conv_2 = Conv2D(5, kernel_size=2, activation='relu')

    batch_norm_layer = BatchNormalization()

    #conv_1 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')
    #conv_2 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')


    #first input model
    input_1 = Input(shape=(nz, nx, channels))
    conv_11 = conv_1(input_1)
    pool_11 = MaxPool2D(pool_size=(2, 2))(conv_11)
    conv_12 = conv_2(pool_11)
    #pool_12 = MaxPool2D(pool_size=(1, 1))(conv_12)
    #f1 = Flatten()(pool_12)
    f1 = Flatten()(conv_12)

    #second input model
    input_2 = Input(shape=(nz, nx, channels))
    conv_21 = conv_1(input_2)
    pool_21 = MaxPool2D(pool_size=(2,2))(conv_21)
    conv_22 = conv_2(pool_21)

    pool_22 = MaxPool2D(pool_size=(2, 1))(conv_22)
    f2 = Flatten()(pool_22)

 #   f2 = Flatten()(conv_22)


    #add global features
    input_3 = Input(shape=(ng, ))

    #merge two layers and add dense
    merge_layer = concatenate([f1, f2, input_3])

    #add batch norm here
    #batch_norm = batch_norm_layer(merge_layer)

    dense_1 = Dense(20, activation='relu')(merge_layer)
    #dense_1 = Dense(20, activation='relu')(batch_norm)
    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2, input_3], outputs=out_1)

    print( model_cnn.summary())

    return model_cnn




def CNN_model02C(nz, nx, channels, ng):

    #CNN to concatenate global features
    #ng is the number of global features

    #shared CNN weights
    conv_1 = Conv2D(4, kernel_size=1, activation='relu', kernel_initializer='glorot_normal')
    conv_2 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')

    drop_l = Dropout(rate=0.25)

    #batch_norm_layer = BatchNormalization()

    #conv_1 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')
    #conv_2 = Conv2D(4, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')


    #first input model
    input_1 = Input(shape=(nz, nx, channels))
    drop_11 = drop_l(input_1)
    conv_11 = conv_1(input_1)
    #conv_11  = conv_1(drop_11)
    pool_11 = MaxPool2D(pool_size=(2, 2))(conv_11)
    conv_12 = conv_2(pool_11)
    #pool_12 = MaxPool2D(pool_size=(2, 1))(conv_12)
    f1 = Flatten()(conv_12)
    #f1 = Flatten()(pool_12)

    #second input model
    input_2 = Input(shape=(nz, nx, channels))
    drop_21 = drop_l(input_2)
    conv_21 = conv_1(input_2)
    #conv_21 = conv_1(drop_21)
    pool_21 = MaxPool2D(pool_size=(2, 2))(conv_21)
    conv_22 = conv_2(pool_21)

    #pool_22 = MaxPool2D(pool_size=(2, 1))(conv_22)
    #f2 = Flatten()(pool_22)

    f2 = Flatten()(conv_22)


    #add global features
    input_3 = Input(shape=(ng, ))

    #merge two layers and add dense
    merge_layer = concatenate([f1, f2, input_3])
    merge_layer = drop_l(merge_layer)

    #add batch norm here
    #batch_norm = batch_norm_layer(merge_layer)

    dense_1 = Dense(16, activation='relu', kernel_initializer='glorot_normal')(merge_layer)
    #dense_1 = Dense(20, activation='relu')(batch_norm)
    out_1 = Dense(1, activation='linear')(dense_1)

    model_cnn = Model(inputs=[input_1, input_2, input_3], outputs=out_1)

    print( model_cnn.summary())

    return model_cnn


def fit_model(model, X1, X2, Y, split_val=0.1):

    #compile model with mse and adam
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit([X1, X2], [Y], epochs=10, batch_size=16, validation_split=0.1)


    return model


def fit_model02(model, X1, X2, Y, split_val=0.1, attempt_max=3):

    val_thres = 1.00

    for i in range(attempt_max):

        init_model = model
        #compile model with mse and adam
        model.compile(loss='mean_squared_error', optimizer='adam')
        train_history = init_model.fit([X1, X2], [Y], epochs=50, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]

        if val_loss < val_thres:
            save_model = init_model
            val_thres = val_loss




    return save_model





def fit_model03(RBF, X1, X2, Y, iter_max=3):

    val_thres = 1.00

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        model_cnn = CNN_model02(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        train_history = model_cnn.fit([X1, X2], [Y], epochs=40, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]




        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model




###-----concatenated Model--------

def fit_model02B(RBF, X1, X2, Xg, Y, iter_max=3):

    val_thres = 1.00

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        #model_cnn = CNN_model02B(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn = CNN_model02C(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
        train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=5, batch_size=16, callbacks=callbacks, validation_split=0.2)
        #train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=10, batch_size=16, validation_split=0.1)
        val_loss = train_history.history['val_loss'][-1]




        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model



####---fit model with validation---------

def fit_model02C(RBF, X1, X2, Xg, Y, X1_v, X2_v, Xg_v, Y_v, iter_max=3):

    val_thres = 1.00


    for i in range(iter_max):

        # model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        callbacks = [EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)]
        #model_cnn = CNN_model02B(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn = CNN_model02C(RBF.shape[1], RBF.shape[2], RBF.shape[3], Xg.shape[1])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        train_history = model_cnn.fit([X1, X2, Xg], [Y], epochs=3, batch_size=16, callbacks=callbacks, validation_data=([X1_v, X2_v, Xg_v], Y_v))
        val_loss = train_history.history['val_loss'][-1]

        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model



#####fit_model04 -> contains

def fit_model04(RBF, X1, X2, Y, X_v1, X_v2, Y_v, iter_max=5):

    val_thres = 1.00

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        model_cnn = CNN_model02(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        model_cnn.compile(loss='mean_squared_error', optimizer='adam')
        train_history = model_cnn.fit([X1, X2], [Y], epochs=40, batch_size=16, validation_data=([X_v1, X_v2], [Y_v]))
        val_loss = train_history.history['val_loss'][-1]




        if val_loss < val_thres:
            print( val_loss)
            save_model = model_cnn
            val_thres = val_loss


    return save_model



def predict_model(model, X):

    Y_p = model.predict(X)


    return Y_p



def get_model_weights(model):


    W = model.get_weights()


    return W




######cross-validation split
def kfold_index(X, Y):

    kf = KFold(n_splits=10)
    #train+

    return None

####

####---------------------------------------------
##--Functions to define space action space


def global_NN01(S_l, A_l, S_g, A_g):

    #S_g -> Global Encoding, into a flattened array
    #S_l -> S is a flattened array containing the local information
    #A_l -> Local action
    #A_g -> global action

    #global actions

    input_Sg = Input(shape=(S_g.shape[0]))
    input_Ag =Input(shape=(A_g.shape[0]))
    merge_global = concatenate([input_Sg, input_Ag])
    dense_global = Dense(40, activation='relu')(merge_global)
    #lets start with the local actions

    input_Sl = Input(shape=(S_l.shape[0]))
    input_Al = Input(shape=(A_l.shape[0]))

    merge_local = concatenate([input_Sl, input_Al, dense_global])

    dense_local = Dense(40, activation='relu')(merge_local)
    out_1 = Dense(S_l.shape[0], activation='linear')(dense_local)

    model_out = Model(inputs=[input_Sg, input_Ag, input_Sl, input_Al], outputs=out_1)





    return model_out



def local_NN01(S_l, A_l):

    input_Sl = Input(shape=(S_l.shape[0]))
    input_Al = Input(shape=(A_l.shape[0]))

    merge_local = concatenate([input_Sl, input_Al])

    dense_local = Dense(40, activation='relu')(merge_local)
    out_1 = Dense(S_l.shape[0], activation='linear')(dense_local)




    return None


def autoencoder_01(X_t, encoding_dim=20, dim_1=64):

    input_x = Input(shape=(X_t.shape[1],))
    # "encoded" is the encoded representation of the input
    int_layer = Dense(dim_1, activation='relu')(input_x)
    encoded = Dense(encoding_dim, activation='linear')(int_layer)

    layer_2 = Dense(dim_1, activation='relu')(encoded)
    # "decoded" is the lossy reconstruction of the input
    #decoded = Dense(X_t.shape[1], activation='linear')(encoded)
    decoded = Dense(X_t.shape[1], activation='linear')(layer_2)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_x, decoded)

    print( autoencoder.summary())


    return autoencoder



def fit_AE01(X_t, iter_max=1):

    val_thres = 1.00

    for i in range(iter_max):

        #model_cnn = CNN_model01(RBF.shape[1], RBF.shape[2], RBF.shape[3])
        model_AE = autoencoder_01(X_t=X_t)
        model_AE.compile(loss='mean_squared_error', optimizer='adam')
        train_history = model_AE.fit(X_t, X_t, epochs=5, batch_size=16, validation_split=0.2)
        val_loss = train_history.history['val_loss'][-1]




        if val_loss < val_thres:
            print( val_loss)
            save_model = model_AE
            val_thres = val_loss


    return save_model



#####make functions to extract the intermediate features

def extract_AE_features(model, RBF_1, RBF_2, encoded_dim=20):

    print( "RBF shape")
    print( RBF_1.shape)

    X1 = np.zeros((RBF_1.shape[0], RBF_1.shape[1], RBF_1.shape[2], encoded_dim))
    X2 = np.zeros_like(X1)

    encoder_layer = K.function([model.layers[0].input], [model.layers[2].output])

    for m in range(0, RBF_1.shape[0]): #sample size
        for i in range(0, RBF_1.shape[1]): #z-dir
            for j in range(RBF_1.shape[2]):
                z_1 = RBF_1[m, i, j, :]
                z_1 = z_1[None, :]

                z_2 = RBF_2[m, i, j, :]
                z_2 = z_2[None, :]

                #print "encoded layer: "
                #print encoder_layer([z_1])[0]

                X1[m, i, j, :] = encoder_layer([z_1])[0]
                X2[m, i, j, :] = encoder_layer([z_2])[0]


    return X1, X2


def load_NN(filepath):

    model = load_model(filepath=filepath)



    return model




def AE_features02(model, RBF_1, RBF_2, encoded_dim=20):

    encoder_layer = K.function([model.layers[0].input], [model.layers[2].output])

    z1 = RBF_1.copy()
    z2 = RBF_2.copy()

    #reshape input arrays
    Z1 = z1.reshape(-1, RBF_1.shape[3])
    Z2 = z2.reshape(-1, RBF_2.shape[3])

    x1 = (encoder_layer([Z1])[0]).reshape(RBF_1.shape[0], RBF_1.shape[1], RBF_1.shape[2], encoded_dim)
    x2 = (encoder_layer([Z2])[0]).reshape(RBF_2.shape[0], RBF_2.shape[1], RBF_2.shape[2], encoded_dim)

    return x1, x2




####------------
###cross-validation to make sure that
def val_check01(x1, x2, crosslink_mat, y_out, trial_max=5):

    val_thres = 1.00

    for trial_num in range(trial_max):

        X_t1, X_t2, Xg_t, Y_t, X_v1, X_v2, Xg_v, Y_v, X_e1, X_e2, Xg_e, Y_e = feature_extract_01.partition_data03(
            X=x1, X2=x2, Xg=crosslink_mat, Y=y_out, n=77, n2=87)

        X_t1, X_t2, Xg_t, Y_t = feature_extract_01.shuffle_data02B(X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t)

        model_2, val_loss = fit_model02C(RBF=x1, X1=X_t1, X2=X_t2, Xg=Xg_t, Y=Y_t, X1_v=X_v1, X2_v=X_v2, Xg_v=Xg_v, Y_v=Y_v)


        if val_loss < val_thres:
            save_model = model_2
            val_thres = val_loss





    return save_model


def predict_with_uncertainty(f, X, num_out=1, n_iter=500):
    result = np.zeros((n_iter, num_out))
    x1 = X[0]
    x2 = X[1]
    x3 = X[2]

    for i in range(n_iter):

        result[i, :] = f([x1, x2, x3, 1])[0]



    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0)

    return prediction, uncertainty

