import keras
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import optimizers, losses, activations, models
from keras.models import Sequential
from keras.layers import Convolution1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score,cohen_kappa_score


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
classificador.add(MaxPooling1D(pool_size=(2)))

classificador.add(Convolution1D(32, kernel_size=3, padding='valid', activation='relu'))
classificador.add(Convolution1D(32, kernel_size=3, padding='valid', activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling1D(pool_size=(2)))

classificador.add(Convolution1D(256, kernel_size=3, padding='valid', activation='relu'))
classificador.add(Convolution1D(256, kernel_size=3, padding='valid', activation='relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling1D(pool_size=(2)))

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

history = classificador.fit(X_train, y_train, batch_size = 128, epochs = 1000, callbacks=callbacks_list)
classificador.load_weights('pesos.h5')

preds = classificador.predict(X_test)
preds_final = np.argmax(preds, axis = -1)

matriz = confusion_matrix(y_test, preds_final)

def metrics(y_test, preds_final):
    acc = accuracy_score(y_test, preds_final)
    prec = precision_score(y_test, preds_final, average='macro')
    recall = recall_score(y_test, preds_final, average='macro')
    f1 = f1_score(y_test, preds_final, average="macro")    #média ponderada da precisão e recordação
    kappa = cohen_kappa_score(y_test, preds_final)
    return acc, prec, recall, F1, kappa
acc, prec, recall, F1, kappa = metrics(y_test, preds_final)

print('Acc = ' + str(acc))
print('Prec = ' + str(prec))
print('Recall = ' + str(recall))
print('F1 = ' + str(F1))
print('Kappa = ' + str(kappa))
print(matriz)

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

df_cm = pd.DataFrame(matriz, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size


#Acc = 0.9882605517997441
#Prec = 0.9474507309123782
#Recall = 0.9108242994263904
#F1 = 0.9287765633561444 
#Kappa = 0.9609927014311314
#[[18067    28    14     3     6]
#[   90   452    10     2     2]
#[   30     5  1392    18     3]
#[   20     0    14   128     0]
#[   11     0     1     0  1596]]