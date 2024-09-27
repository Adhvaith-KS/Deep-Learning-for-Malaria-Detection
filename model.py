import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import sys

sys.stdout.reconfigure(encoding='utf-8') #for encoding symbols in output 


# here, I am defining the path to the dataset
base_dir = r'D:\\Downloads\\malariaimages\\cell_images\\cell_images'

# Image Dataset modifications
train_datagen = ImageDataGenerator(
    rescale=1./255,         # To Normalise pixel values
    shear_range=0.2,        # Random changing of image to train the model for adverse cases
    zoom_range=0.2,         # Random zooming of dataset images to train the model on a more diverse dataset
    horizontal_flip=True,   # To flip images for the above reason
    validation_split=0.2    # I am using 20% of the dataset for validation (to see how good the model is with data it has not seen before)
)

# Training and Validation 
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),  # Resizing all images to 150x150
    batch_size=32,           # 32 photos at a time used for training
    class_mode='binary',     # Binary classification since there is only Infected and Uninfected for now
    subset='training'        # Set for training
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'      # Set for validation images
)

# Building the CNN Model with the below layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification (since there are only 2 types in this case)
])

# Compiling the Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Training the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Saving  the Model
model.save('malaria_cnn_model.h5')

# Evaluating and displaying results of the Model on Validation Data
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
