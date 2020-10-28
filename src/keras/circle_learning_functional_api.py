from sklearn.datasets import make_circles
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

def plot_data(pl,X,y):
    # plot class where y==0
    pl.plot(X[y==0,0], X[y==0,1],'ob',alpha=0.5)

    #Plot class where y==1
    pl.plot(X[y==1,0], X[y==1,1],'xr',alpha=0.5)
    pl.legend(['0','1'])
    return pl


#plot contour instead of boundary
def plot_decision_boundary(model,X,y):
    amin, bmin=X.min(axis=0) - 0.1
    amax, bmax=X.max(axis=0) + 0.1

    hticks= np.linspace(amin,amax,101)
    vticks= np.linspace(bmin,bmax,101)

    aa,bb = np.meshgrid(hticks,vticks)
    ab = np.c_[aa.ravel(),bb.ravel()]

    #make predictioniwth model and reshape the output so contourf can plot
    c=model.predict(ab)
    Z = c.reshape(aa.shape)

    plt.figure(figsize=(12,8))

    #plot the contour
    plt.contourf(aa,bb,Z,cmap='bwr',alpha=0.2)

    #plot the moons of data
    plot_data(plt,X,y)
    return plt


# Generate data from blobs, with 2 centers

X,y = make_circles(n_samples=1000,factor=0.6,noise=0.1,random_state=42)

pl=plot_data(plt,X,y)
pl.show()

# as data is linearly separable, we can classify using single neuron ( without any hidden layer)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Using Functional API

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


# creating a  model
inputs = Input(shape=(2,)) # input layer

x1 = Dense(4, activation='tanh', name='Hidden1')(inputs) # hidden 1
x2 = Dense(4, activation='tanh', name='Hidden2')(x1) # hidden 2 ( get input from previous layer)

o = Dense(1, activation='sigmoid',name="Output")(x2)

# lets define our model ( we can easily change order of layers if we wanted to using functional APIs)

model = Model(inputs=inputs, outputs=o) # only 1 input layer and 1 output layer, can also specify as lists

model.summary()
#compile
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

# from keras.utils import plot_model
# plot_model(model,to_file='model.png',show_shapes=True, show_layer_names=True)

#fit the model, with 100 cycles
model.fit(X_train,y_train,epochs=100,verbose=1)


# get loss and accuracy on test_data

eval_result = model.evaluate(X_test,y_test)

#print accuracy
print("\n\n TestLoss: ",eval_result[0]," test Accuracy: ",eval_result[1])

#plot contour boundary

plot_decision_boundary(model,X,y).show()
