import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#Convolutional neural network
#Images data generation

train_datagen=ImageDataGenerator(zoom_range=0.2,
                                 shear_range=0.2,
                                 horizontal_flip=True,
                                 rotation_range=0.2,
                                 rescale=1/255)

test_datagen=ImageDataGenerator(rescale=1/255)


train_dataset=train_datagen.flow_from_directory(r"D:/projects/FaceMaskDetector-master-20220525T071512Z-001/FaceMaskDetector-master/train",
                                                class_mode="binary",
                                                target_size=(150,150),
                                                batch_size=16)

test_dataset=test_datagen.flow_from_directory(r"D:/projects/FaceMaskDetector-master-20220525T071512Z-001/FaceMaskDetector-master/test",
                                                class_mode="binary",
                                                target_size=(150,150),
                                                batch_size=16)



#CNN
cnn=tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,
                               input_shape=(150,150,3),activation="relu"))

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation="relu"))

#Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2))

cnn.add(tf.keras.layers.Flatten())

#Hidden layer
cnn.add(tf.keras.layers.Dense(units=120,activation="relu"))

cnn.add(tf.keras.layers.Dense(units=1,activation="sigmoid"))


cnn.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

cnn.fit(train_dataset,validation_data=test_dataset,epochs=100)



cnn.save("mymodel.h5")






# Steps
# 1->Convolution layer(May be multiple)
# 2->Pooling layer(May be multiple)
# 3->Flatten
# 4->Hidden layer(May be multiple)
# 5->Output







