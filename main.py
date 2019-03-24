from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.models import load_model
import numpy as np
import cv2


def reverse(arg):
    for x in range(len(arg)):
        for y in range(len(arg[x])):
            if arg[x][y] == 255:
                arg[x][y] = 0
            elif arg[x][y] == 0:
                arg[x][y] = 254
    return arg


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # plt.imshow(x_train[1])
    # plt.show()
    # print(x_train[0].shape)

    # reshape data to fit model
    x_train_reshaped = x_train.reshape(60000, 28, 28, 1)
    x_test_reshaped = x_test.reshape(10000, 28, 28, 1)

    # one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # create model
    # model = Sequential()

    # add model layers
    # model.add(Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)))
    # model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(10, activation='softmax'))
    #
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(x_train_reshaped, y_train, validation_data=(x_test_reshaped, y_test), epochs=3)

    # model.save('models/my_first_model.h5')

    model = load_model('models/my_first_model.h5')

    # singe_train_image = np.expand_dims(x_train_reshaped[0], axis=0))
    # model.predict(single_train_image)

    input = cv2.imread("data/one.png",0)
    input = reverse(input)
    input = input.reshape((-1, 28, 28, 1))

    model.predict(input)




