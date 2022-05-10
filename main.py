import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

csv = pd.read_csv('dados.csv', sep=',')
csv = csv.drop(columns=['lote'])
labelEnc = LabelEncoder()
csv['fruta'] = labelEnc.fit_transform(csv['fruta'])
dados = csv.values

classificators = dados[:,0]

atributes = dados[:,1:]

model = Sequential()
model.add(Dense(units=1, activation='sigmoid'))

model.add(Dense(units=5, activation='relu'))

model.fit(atributes, classificators, batch_size=10, epochs=100)

model.compile(optimizer='adam',  loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

model.save('modelo.h5')