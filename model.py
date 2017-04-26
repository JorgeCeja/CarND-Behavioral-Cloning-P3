
from __future__ import division

from keras.preprocessing.image import flip_axis, random_shift
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from keras.models import Sequential

import numpy as np
import sklearn
import random
import csv
import cv2


def getData(path):
    X, y = [], []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # non representative data
            if float(line[6]) < 20:
                continue

            centerImgPath = './data/IMG/'+line[0].split('/')[-1]
            leftImgPath = './data/IMG/'+line[1].split('/')[-1]
            rightImgPath = './data/IMG/'+line[2].split('/')[-1]

            X += [centerImgPath, leftImgPath, rightImgPath]
            y += [float(line[3]), float(line[3]) + 0.25, float(line[3]) - 0.25]

    return X, y


def getImage(path):
    # get image in BGR color space
    img = cv2.imread(path)
    # Crop top 40px and bottom 20px
    img = img[40:140,:,:]
    # Resize cropped image to 200x66
    img = cv2.resize(img,(200, 66), interpolation=cv2.INTER_AREA)
    # convert to RGB for further augmentation
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def changeBrightness(img):
    brightness = np.random.uniform(0.3, 1.2)
    newImg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    newImg[:,:,2] = newImg[:,:,2]*brightness
    newImg = cv2.cvtColor(newImg, cv2.COLOR_HSV2RGB)
    return newImg


def generator(X, y, validation, batch_size):
    while 1: # Loop forever so the generator never terminates
        images, angles = [], []

        for i in range(batch_size - 1):
            sampleIndex = random.randint(0, len(X) - 1)
            image = getImage(X[sampleIndex])
            steeringAngle = y[sampleIndex]

            if validation == False:
                if random.random() < 0.5:
                    image = changeBrightness(image)

                # shift vertically +-15%
                image = random_shift(image, 0, 0.15, 0, 1, 2)

                if random.random() < 0.5:
                    image = flip_axis(image, 1)
                    steeringAngle = -steeringAngle

            # convert to BGR color space from RGB
            image = image[...,::-1]
            # Convert to YUV color space from BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

            images.append(image)
            angles.append(steeringAngle)

        X_train, y_train = np.array(images), np.array(angles)

        yield X_train, y_train


def model():
    ch, row, col = 3, 66, 200 # COLOR , HEIGHT, WIDTH

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', activation='elu'))

    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu'))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='elu'))

    model.add(Flatten())

    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.50))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.50))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(0.50))

    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

if __name__ == '__main__':
    X, y = getData('./data/driving_log.csv')
    sklearn.utils.shuffle(X, y)
    
    # Split our data into trainig and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    trainGenerator = generator(X_train, y_train, False, 256)
    validationGenerator = generator(X_test, y_test, True, 256)

    network = model()
    network.fit_generator(trainGenerator, validation_data=validationGenerator, nb_val_samples=len(X_test), samples_per_epoch=len(X_train), nb_epoch=5)
    print network.summary()
    network.save('model.h5')
