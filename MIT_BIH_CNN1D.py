import keras
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import optimizers, losses, activations, models
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPool1D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

df_train = pd.read_csv("database/mitbih_train.csv", header=None)
df_test = pd.read_csv("database/mitbih_test.csv", header=None)

X_train = np.array(df_train[list(range(187))].values)[..., np.newaxis]
y_train = np.array(df_train[187].values).astype(np.int8)

X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
y_test = np.array(df_test[187].values).astype(np.int8)

#cnn
nclass = 5
classificador = Sequential()
classificador.add(Convolution1D(16, kernel_size=5, padding='valid', activation='relu'))
classificador.add(Convolution1D(16, kernel_size=5, padding='valid', activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPool1D(pool_length=(2)))

classificador.add(Convolution1D(32, kernel_size=3, padding='valid', activation='relu'))
classificador.add(Convolution1D(32, kernel_size=3, padding='valid', activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPool1D(pool_length=(2)))

classificador.add(Convolution1D(256, kernel_size=3, padding='valid', activation='relu'))
classificador.add(Convolution1D(256, kernel_size=3, padding='valid', activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPool1D(pool_length=(2)))

classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.3))

classificador.add(Dense(units=64, activation ='relu'))
classificador.add(Dropout(0.3))

classificador.add(Dense(units = nclass, activation = 'softmax'))

otimizador = optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
classificador.compile(optimizer= otimizador, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

es = EarlyStopping(monitor='acc', mode="max", patience=5, verbose=1)
rlr = ReduceLROnPlateau(monitor='acc', mode="max", patience=3, verbose=2)
checkpoint = ModelCheckpoint(filepath ='pesos.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, es, rlr]

classificador.fit(X_train, y_train, batch_size = 128, epochs = 1000, callbacks=callbacks_list)
history = classificador.fit(X_train, y_train, batch_size = 128, epochs = 1000, callbacks=callbacks_list)
classificador.load_weights('pesos.h5')

preds = classificador.predict(X_test)
preds_final = np.argmax(preds, axis = -1)

matriz = confusion_matrix(y_test, preds_final)

f1 = f1_score(y_test, preds_final, average="macro") #média ponderada da precisão e recordação
acc = accuracy_score(y_test, preds_final)

print("accuracy score : %s "% acc)
#accuracy score : 0.9866161154759729 

print("f1 score : %s "% f1)
#f1 score : 0.919695350993216