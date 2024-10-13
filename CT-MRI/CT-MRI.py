import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# Suppress TensorFlow logging (1 = WARNING, 2 = ERROR, 3 = FATAL)

# Define constants
IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 400
BATCH_SIZE = 1

# Function to load and preprocess images
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    return image

# Load CT dataset
def load_ct_dataset(path):
    dataset = tf.data.Dataset.list_files(path + '/*.jpg')
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset

ct_dataset = load_ct_dataset('Images')

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3]))
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(128, (4, 4), strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2D(256, (4, 4), strides=2, padding='same'))
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=2, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2DTranspose(64, (4, 4), strides=2, padding='same'))
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2DTranspose(3, (4, 4), strides=2, padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))

    return model

# Instantiate the generator model
generator_g = build_generator()  # CT to MRI

# Function to generate images
def generate_images(model, test_input):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image (CT)', 'Translated Image (MRI)']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        plt.imshow((display_list[i] * 0.5) + 0.5)
        plt.axis('off')
    plt.show()

# Visualize some results
for sample_ct in ct_dataset.take(1):
    generate_images(generator_g, sample_ct)