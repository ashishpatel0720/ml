from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import os

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

X,y = make_blobs(n_samples=1000,centers=2,random_state=42)

pl=plot_data(plt,X,y)
pl.show()

# as data is linearly separable, we can classify using single neuron ( without any hidden layer)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# creating a Sequential model

model = Sequential()

# add only neuron required
model.add(Dense(1,input_shape=(2,), activation='sigmoid'))

#compile
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])

#fit the model, with 100 cycles
model.fit(X_train,y_train,epochs=100,verbose=0)


# get loss and accuracy on test_data

eval_result = model.evaluate(X_test,y_test)

#print accuracy
print("\n\n TestLoss: ",eval_result[0]," test Accuracy: ",eval_result[1])

#plot contour boundary

plot_decision_boundary(model,X,y).show()
