# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist  # Example dataset, replace with your own

# Load and preprocess the dataset (replace with your dataset loading code)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

# Build the neural network model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))  # Flatten 28x28 images to a 1D array
model.add(layers.Dense(128, activation='relu'))  # Fully connected layer with 128 neurons
model.add(layers.Dropout(0.2))  # Dropout layer to reduce overfitting
model.add(layers.Dense(10, activation='softmax'))  # Output layer with 10 neurons (for 10 classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (replace with your own training data)
model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')

# Make predictions (replace with your own input data)
predictions = model.predict(test_images[:5])
print('Predictions:', predictions)
