#digit recognition model
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    # keras.layers.Dense(10, activation21='softmax')
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model with appropriate optimizer and loss function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation for training
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2, 
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

# Reshape the data for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Fit the model with data augmentation
batch_size = 32
epochs = 5

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size,
    epochs=epochs,
    validation_data=(x_test, y_test),
)
## vis
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the training and validation loss over epochs
plt.figure(figsize=(10, 5)) 
plt.plot(range(1, epochs + 1), train_loss, label='Training Loss', marker='o')
plt.plot(range(1, epochs + 1), val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# Visualize model predictions on test images
num_images_to_visualize = 10

# Randomly select some test images
random_indices = np.random.choice(x_test.shape[0], num_images_to_visualize, replace=False)
sample_images = x_test[random_indices]
sample_labels = y_test[random_indices]

# Make predictions on the sample images
predictions = model.predict(sample_images)

# Create a plot to display the sample images and their predictions
plt.figure(figsize=(12, 8))
for i in range(num_images_to_visualize):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"True: {sample_labels[i]}\nPredicted: {np.argmax(predictions[i])}")
    plt.axis('off')

plt.show()
