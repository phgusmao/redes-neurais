from keras.models import load_model
import numpy as np


newDatas = np.array([
    [3.1, 122],
    [4.1, 146],
    [2.2, 86]
])

model = load_model('modelo.h5')

result = model.predict(newDatas)
for i in range(len(result)):
    if(result[i]>=0.5):
        print('Limão')
    elif(result[i]<0.5):
        print('Laranja')