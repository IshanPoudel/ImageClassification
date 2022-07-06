#use the MINST fashion project for classification of projects

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
# import certifi
from tensorflow.keras import layers as l

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
actual_test_image = test_images
#all the images are pizel values ,
#convert the pixel value between 0 TO 1
#stored in numpy array
train_images = train_images/255
test_images = test_images/255

#define different layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128 , activation ="relu"),
    keras.layers.Dense(10 , activation="softmax")
])

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy' ,
              metrics=['accuracy'])

model.fit(train_images , train_labels , epochs=5 )

#evaluate the model
test_loss , test_acc = model.evaluate(test_images , test_labels)
print("Tested Acc: " , test_acc)

#make predictions based on that model
prediction = model.predict(test_images)
#put the input ,
# produces prediction value for each

# for clothes in prediction:
#     print(class_names[np.argmax(clothes)])

for i in range(10):
    plt.grid(False)
    plt.imshow(actual_test_image[i] )
    plt.xlabel("Actual: " +class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()


