import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import optimizers, losses, activations, models
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPool1D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score

classificador = Sequential()
classificador.add(Convolution1D(32, kernel_size=5, padding='valid', activation='relu')) #trocar 32 por 64
classificador.add(BatchNormalization())
classificador.add(MaxPool1D(pool_size=(2)))

classificador.add(Convolution1D(32, kernel_size=5, padding='valid', activation='relu')) #trocar 32 por 64
classificador.add(BatchNormalization())
classificador.add(MaxPool1D(pool_size=(2)))

classificador.add(Flatten())

classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units=128, activation ='relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 6, activation = 'softmax'))

classificador.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])

df_train = pd.read_csv("database/mitbih_train.csv", header=None)
df_test = pd.read_csv("database/mitbih_test.csv", header=None)

X_train = np.array(df_train[list(range(187))].values)[..., np.newaxis]
y_train = np.array(df_train[187].values).astype(np.int8)

X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
y_test = np.array(df_test[187].values).astype(np.int8)

previsores_treinamento = X_train.reshape(X_train.shape[0],187,1)
previsores_teste = X_test.reshape(X_test.shape[0],187,1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

classe_treinamento = np_utils.to_categorical(y_train, 6)
classe_teste = np_utils.to_categorical(y_train, 6)

es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=0, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
ncp = ModelCheckpoint(filepath ='pesos.h5', monitor='loss', save_best_only=True)

classificador.fit(X_train, y_train, batch_size = 128, epochs = 100, callbacks =[es, rlr,ncp])


preds = classificador.predict(X_test)
preds_final = np.argmax(preds, axis = -1)

acc = accuracy_score(y_test, preds_final)

print("Accuracy score : {}".format(acc))