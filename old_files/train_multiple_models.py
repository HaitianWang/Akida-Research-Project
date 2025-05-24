import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os, shutil, random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cnn2snn
import akida as ak
from keras import Model
from keras.layers import Activation, Dropout, Reshape
from akida_models.layer_blocks import dense_block
from akida_models import fetch_file, akidanet_imagenet, mobilenet_imagenet
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Change Akida version
os.environ["CNN2SNN_TARGET_AKIDA_VERSION"] = "v1"
os.chdir('notebooks/Documents/GitHub/GENG5551-Akida-Chip')

metadata = pd.read_csv('archive/HAM10000_metadata.csv')

# Define paths for train and test datasets
train_dir = 'archive/data/train'
test_dir = 'archive/data/test'

# Define the target directories for cancerous and benign images
train_cancerous_dir = 'archive/data/train/Cancerous'
train_benign_dir = 'archive/data/train/Benign'
test_cancerous_dir = 'archive/data/test/Cancerous'
test_benign_dir = 'archive/data/test/Benign'

image_data_generators = []

train_datagen1 = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    channel_shift_range=50.0
)

image_data_generators.append(train_datagen1)

def random_cutout(img, mask_size):
    h, w, c = img.shape
    y = np.random.randint(0, h)
    x = np.random.randint(0, w)
    
    y1 = np.clip(y - mask_size // 2, 0, h)
    y2 = np.clip(y + mask_size // 2, 0, h)
    x1 = np.clip(x - mask_size // 2, 0, w)
    x2 = np.clip(x + mask_size // 2, 0, w)
    
    img[y1:y2, x1:x2, :] = 0  # Apply cutout (set pixel values to 0)
    return img

# Create a custom function to apply cutout during augmentation
def custom_augmentation(img):
    img = random_cutout(img, mask_size=30)  # Experiment with mask size
    return img

# Updated ImageDataGenerator for training set with brightness and channel shift range
train_datagen2 = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Brightness range (adjust as needed)
    channel_shift_range=50.0,  # Shift color channels to simulate lighting variation
    preprocessing_function=custom_augmentation  # Apply cutout augmentation
)



test_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=100,
    class_mode='binary',
)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=100,
    class_mode='binary'
)


IMG_SIZE = 224
CLASSES = 2


# Create a base model without top layers
base_model = akidanet_imagenet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                               classes=CLASSES,
                               alpha=0.5,
                               include_top=False,
                               pooling='avg')


# Get pretrained quantized weights and load them into the base model
pretrained_weights = fetch_file(

    origin='https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_alpha_50.h5',
    fname="akidanet_imagenet_224_alpha_50.h5"
    )

base_model.load_weights(pretrained_weights, by_name=True)
base_model.summary()


# layer freezing
for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = dense_block(x,
                units=512,
                name='fc1',
                add_batchnorm=True,
                relu_activation='ReLU6')
x = Dropout(0.5, name='dropout_1')(x)
x = dense_block(x,
                units=CLASSES,
                name='predictions',
                add_batchnorm=False,
                relu_activation=False)


x = Reshape((CLASSES,), name='reshape1')(x)

# Build the model
model_keras = Model(base_model.input, x, name='akidanet_derma')

model_keras.summary()


# Setting up the learning rate schedule
initial_learning_rate = 1e-3
final_learning_rate = 1e-5
decay_steps = 10
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps,
    decay_rate=(final_learning_rate / initial_learning_rate) ** (1 / decay_steps),
    staircase=True)

# Configuring the optimizer
# optimizer = RAdam(learning_rate=lr_schedule)

optimizer = RMSprop(
    learning_rate=1e-3,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-7
)

# Compiling the model
model_keras.compile(
    optimizer=optimizer,
    loss=BinaryCrossentropy(),
    metrics=['accuracy'])

# Setting up callbacks for saving the model and early stopping
checkpoint_cb = ModelCheckpoint(
    'akidanet_derma_best.h5', save_best_only=True)
early_stopping_cb = EarlyStopping(
    patience=10, restore_best_weights=True)

# GB: To change nb of epochs 
# (on CPU, 1 epoch lasts 22 min, so 10 is 4 hours long)
EPOCHS = 15 # Initial value: 10

# Training the model
history = model_keras.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=test_gen,
    callbacks=[checkpoint_cb, early_stopping_cb])