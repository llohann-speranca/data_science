import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score



df = pd.read_csv('admissions_data.csv')


X = df.iloc[:,1:-1]
y = df.iloc[:,-1]




X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)




def model_fn():
    model = Sequential( \
            [layers.InputLayer(input_shape=(X.shape[1])),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1,activation='sigmoid')
            ])
    model.compile(loss='mse', metrics='mae', optimizer=Adam(learning_rate=0.01))
    return model

model = model_fn()

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=0)
hist = keras.callbacks.History()


scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train,\
                epochs = 150,\
                batch_size = 32,\
                callbacks = [es, hist],\
                validation_split=0.2,
                verbose=0
)




print(f'R2 score: {r2_score(y_test, model.predict(X_test_scaled))}')

# Do extensions code below
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below

fig, ax = plt.subplots(1,1)
ax.plot(hist.history['loss'], label='train')
ax.plot(hist.history['val_loss'], label='test')
ax.legend()
# fig.savefig('static/images/my_plots.png')
plt.show()
