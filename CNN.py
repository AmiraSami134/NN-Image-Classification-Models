import tensorflow
from tensorflow import keras
from tensorflow.keras import datasets

from keras.layers import Dense,Conv2D,MaxPool2D,Flatten 

from keras import Sequential

(x_train,y_train),(x_test,y_test)=datasets.mnist.load_data()

print(x_train[0].shape)
print(x_train[0])

print(x_train.shape)

print(x_test.shape)

print(y_train[0])

%pylab inline
import matplotlib.pyplot as plt
plt.imshow(x_train[0])

# to get the value of the label in a vector
Y_train=keras.utils.to_categorical(y_train,10)
Y_test=keras.utils.to_categorical(y_test,10)

print(Y_train[0])

# CNN 
model=Sequential()

model.add(Conv2D(filters=32, kernel_size=5, strides=(1,1), padding='same', activation=tensorflow.nn.relu, input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=64, kernel_size=5, strides=(1,1), padding='same', activation=tensorflow.nn.relu))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Flatten())

model.add(Dense(units=128, activation=tensorflow.nn.relu))

model.add(Dense(units=10, activation=tensorflow.nn.softmax))

model.summary()

x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=x_train,y=Y_train, epochs=1, batch_size=128)

test_loss, test_accuracy=model.evaluate(x=x_test,y=Y_test)
print("The test accuracy= ", test_accuracy)

model.save('CNN_MODEL.h5')