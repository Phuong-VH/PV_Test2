# from __future__ import print_function, division
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from keras.layers import Input, ZeroPadding3D, concatenate
from tensorflow.keras import regularizers
from keras.optimizers import Adam
from keras.utils import to_categorical
from qpso import QPSO
import datetime
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

data_directory = './PV_Test/3D_data_total.npz'
data = np.load(data_directory)

X = data['X_all']
Y = data['Y_all']

X_all = X/255

X_train, X_test, y_train, y_test = train_test_split(X_all, Y, test_size=0.20, random_state=42)

# One-hot encode the target labels
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Convert the target labels to dense tensors
y_train_dense = tf.convert_to_tensor(y_train_onehot, dtype=tf.float32)

def first_model(zeropad_1, filt1, filtlen_1,dropout1,pol1):
    nb_filter1  = filt1;
    filter1_size = filtlen_1;
    poling1 = pol1;
    dropout1 = dropout1;
    
    conv1 = Conv3D(nb_filter1, (filter1_size, filter1_size, filter1_size), activation='relu', padding="same")(zeropad_1)
    max1 = MaxPooling3D(pool_size=(poling1, poling1, poling1), padding="same")(conv1)
    drop1 = Dropout(dropout1)(max1)
    flat1 = Flatten()(drop1)
    return flat1

def second_model(zeropad_1, filt1, filt2, filtlen_1, filtlen_2, dropout1, dropout2, pol1, pol2):
    nb_filter1  = filt1;
    nb_filter2  = filt2;
    filter1_size = filtlen_1;
    filter2_size = filtlen_2;
    poling1 = pol1;
    poling2 = pol2;
    dropout1 = dropout1;
    dropout2 = dropout2;
    
    conv1 = Conv3D(nb_filter1, (filter1_size, filter1_size, filter1_size), activation='relu', padding="same")(zeropad_1)
    max1 = MaxPooling3D(pool_size=(poling1, poling1, poling1), padding="same")(conv1)
    drop1 = Dropout(dropout1)(max1)
    conv2 = Conv3D(nb_filter2, (filter2_size, filter2_size, filter2_size), activation='relu', padding="same")(drop1)
    max2 = MaxPooling3D(pool_size=(poling2, poling2, poling2), padding="same")(conv2)
    drop2 = Dropout(dropout2)(max2)
    flat2 = Flatten()(drop2)
    return flat2

def thrid_model(zeropad_1, filt1, filt2, filt3, filtlen_1, filtlen_2, filtlen_3, dropout1, dropout2, dropout3, pol1, pol2, pol3):
    nb_filter1  = filt1;
    nb_filter2  = filt2;
    nb_filter3  = filt3;
    filter1_size = filtlen_1;
    filter2_size = filtlen_2;
    filter3_size = filtlen_3;
    poling1 = pol1;
    poling2 = pol2;
    poling3 = pol3;
    dropout1 = dropout1;
    dropout2 = dropout2;
    dropout3 = dropout3;

    
    conv1 = Conv3D(nb_filter1, (filter1_size, filter1_size, filter1_size), activation='relu', padding="same")(zeropad_1)
    max1 = MaxPooling3D(pool_size=(poling1, poling1, poling1), padding="same")(conv1)
    drop1 = Dropout(dropout1)(max1)
    conv2 = Conv3D(nb_filter2, (filter2_size, filter2_size, filter2_size), activation='relu', padding="same")(drop1)
    max2 = MaxPooling3D(pool_size=(poling2, poling2, poling2), padding="same")(conv2)
    drop2 = Dropout(dropout2)(max2)
    conv3 = Conv3D(nb_filter3, (filter3_size, filter3_size, filter3_size), activation='relu', padding="same")(drop2)
    max3 = MaxPooling3D(pool_size=(poling3, poling3, poling3), padding="same")(conv3)
    drop3 = Dropout(dropout3)(max3)    
    flat3 = Flatten()(drop3)
    return flat3

def fourth_model(zeropad_1, filt1, filt2, filtlen_1, filtlen_2, dropout1, dropout2, pol1, pol2):
    #2CNN parallel
    nb_filter1  = filt1;
    nb_filter2  = filt2;
    filter1_size = filtlen_1;
    filter2_size = filtlen_2;
    poling1 = pol1;
    poling2 = pol2;
    dropout1 = dropout1;
    dropout2 = dropout2;

    conv1 = Conv3D(nb_filter1, (filter1_size, filter1_size, filter1_size), activation='relu', padding="same")(zeropad_1)
    max1 = MaxPooling3D(pool_size=(poling1, poling1, poling1), padding="same")(conv1)
    drop1 = Dropout(dropout1)(max1)
    flat1 = Flatten()(drop1)
    
    conv2 = Conv3D(nb_filter2, (filter2_size, filter2_size, filter2_size), activation='relu', padding="same")(zeropad_1)
    max2 = MaxPooling3D(pool_size=(poling2, poling2, poling2), padding="same")(conv2)
    drop2 = Dropout(dropout2)(max2)
    flat2 = Flatten()(drop2)

    comb = concatenate([flat1,flat2])

    return comb

def fifth_model(zeropad_1, filt1, filt2, filt3, filtlen_1, filtlen_2, filtlen_3, dropout1, dropout2, dropout3, pol1, pol2, pol3):
    #3CNNparallel
    nb_filter1  = filt1;
    nb_filter2  = filt2;
    nb_filter3  = filt3;
    filter1_size = filtlen_1;
    filter2_size = filtlen_2;
    filter3_size = filtlen_3;
    poling1 = pol1;
    poling2 = pol2;
    poling3 = pol3;
    dropout1 = dropout1;
    dropout2 = dropout2;
    dropout3 = dropout3;

    conv1 = Conv3D(nb_filter1, (filter1_size, filter1_size, filter1_size), activation='relu', padding="same")(zeropad_1)
    max1 = MaxPooling3D(pool_size=(poling1, poling1, poling1), padding="same")(conv1)
    drop1 = Dropout(dropout1)(max1)
    flat1 = Flatten()(drop1)
    
    conv2 = Conv3D(nb_filter2, (filter2_size, filter2_size, filter2_size), activation='relu', padding="same")(zeropad_1)
    max2 = MaxPooling3D(pool_size=(poling2, poling2, poling2), padding="same")(conv2)
    drop2 = Dropout(dropout2)(max2)
    flat2 = Flatten()(drop2)

    conv3 = Conv3D(nb_filter3, (filter3_size, filter3_size, filter3_size), activation='relu', padding="same")(zeropad_1)
    max3 = MaxPooling3D(pool_size=(poling3, poling3, poling3), padding="same")(conv3)
    drop3 = Dropout(dropout3)(max3)
    flat3 = Flatten()(drop3)

    comb = concatenate([flat1,flat2,flat3])

    return comb

def cnn(x):
    t1_0 = datetime.datetime.now()
    struct = int(x[0])
    filt1 = int(x[1])
    filt2 = int(x[2])
    filt3 = int(x[3])
    filtlen_1 = int(x[4])
    filtlen_2 = int(x[5])
    filtlen_3 = int(x[6])
    dropout1 = x[7]
    dropout2 = x[8]
    dropout3 = x[9]
    pol1 = int(x[10])
    pol2 = int(x[11])
    pol3 = int(x[12])
    dense1 = int(x[13])
    dropout4 = x[14]

    inp_1 = Input(shape=(256, 256, 5, 1))
    zeropad_1 = ZeroPadding3D(padding=(1, 1, 1))(inp_1)

    if struct == 1:
        nn = first_model(zeropad_1, filt1, filtlen_1,dropout1,pol1)
    elif struct == 2:
        nn = second_model(zeropad_1, filt1, filt2, filtlen_1, filtlen_2, dropout1, dropout2, pol1, pol2)
    elif struct == 3:
        nn = thrid_model(zeropad_1, filt1, filt2, filt3, filtlen_1, filtlen_2, filtlen_3, dropout1, dropout2, dropout3, pol1, pol2, pol3)
    elif struct == 4:
        nn = fourth_model(zeropad_1, filt1, filt2, filtlen_1, filtlen_2, dropout1, dropout2, pol1, pol2)
    elif struct == 5:
        nn = fifth_model(zeropad_1, filt1, filt2, filt3, filtlen_1, filtlen_2, filtlen_3, dropout1, dropout2, dropout3, pol1, pol2, pol3)
    else:
        raise ValueError("Invalid struct value")

    dropout_output = Dropout(dropout4) (nn)
    hid = Dense(dense1, activation='relu')(dropout_output)
    output = Dense(10, activation='softmax')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model = model.fit(X_train, y_train_dense, epochs=100, batch_size=16, verbose=0, validation_data=(X_test, y_test_dense))
    cnn_features = model.predict(X_train)
    y_train_dense_np = np.array(y_train_dense)
    y_train_catboost = np.argmax(y_train_dense_np, axis=1)
    
    catboost_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth, learning_rate=catboost_learning_rate, verbose=0)
    catboost_model.fit(cnn_features, y_train_catboost)
    
    score = catboost_model.score(cnn_features, y_train_catboost)

    t2_0 = datetime.datetime.now()
    
    print('score: ', -score, 'values: ', func, 'time: ', t2_0-t1_0)
    return -score 

bounds = [(1,5),(16,64),(16,64),(16,64),(2,5),(2,5),(2,5),(0.2,0.5),(0.2,0.5),(0.2,0.5),(2,5),(2,5),(2,5),(32,128),(0.2,0.5)]
NDim = 15;
s = QPSO(sphere, NParticle, NDim, bounds, MaxIters)
s.update(callback=log, interval=100)
print("Found best position: {0}".format(s.gbest))

