import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

train_dir = 'dataset/train'
test_dir = 'dataset/test'

img_width, img_height = 48, 48
batch_size = 64
epochs = 30

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(img_width, img_height), color_mode='grayscale', batch_size=batch_size, class_mode='categorical')
test_data  = test_datagen.flow_from_directory(test_dir, target_size=(img_width, img_height), color_mode='grayscale', batch_size=batch_size, class_mode='categorical')

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_data, validation_data=test_data, epochs=epochs)

if not os.path.exists("model"):
    os.makedirs("model")
model.save("model/emotion_model.h5")