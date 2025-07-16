# === Imports and Initial Setup ===
import itertools
import os
import shutil
from glob import glob
from timeit import default_timer as timer
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from IPython.display import clear_output
from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense, Dropout,
                                     Flatten, MaxPooling2D)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import akida as ak
import cnn2snn
from quantizeml.models.transforms import sanitize

# Change Akida version
os.environ["CNN2SNN_TARGET_AKIDA_VERSION"] = "v1"

# Double-check Avida version
print(' Akida version: ', cnn2snn.get_akida_version())
seed = (4,2)
os.chdir('notebooks/Documents/GitHub/GENG5551-Akida-Chip')
os.getcwd()

# === Dataset Preparation and Preprocessing ===
df = pd.read_csv('archive/HAM10000_metadata.csv')
size = (64, 64)
base_dir = './archive'
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(base_dir, '*', '*.jpg'))}

df['path'] = df['image_id'].map(imageid_path_dict.get)
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize(size)))
df = df[['dx', 'image']]
df.head()
df['dx'].value_counts()
arr = df.iloc[1]['image']
plt.imshow(arr)
plt.axis('off')
plt.show()

n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (12,12))
for n_axs, (type_name, type_rows) in zip(m_axs,
                                         df.sort_values(['dx']).groupby('dx')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=42).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')

counts = pd.DataFrame(df['dx'].value_counts()).reset_index()
df_nv = df[df['dx'] == 'nv']
df_df = df[df['dx'] == 'df']
vasc_df = df[df['dx'] == 'vasc']
akiec_df = df[df['dx'] == 'akiec']
bcc_df = df[df['dx'] == 'bcc']
bkl_df = df[df['dx'] == 'bkl']
mel_df = df[df['dx'] == 'mel']


# === Data Augmentation ===
# Function: aug_brightness — defines core processing logic
def aug_brightness(img_arr):
    aug_arr = tf.image.stateless_random_brightness(img_arr, 0.7, seed)
    aur_arr = np.array(aug_arr)
    return aug_arr

# Function: aug_contrast — defines core processing logic
def aug_contrast(img_arr):
    aug_arr = tf.image.stateless_random_contrast(img_arr, 0.2, 0.8, seed)
    aur_arr = np.array(aug_arr)
    return aug_arr

# Function: aug_flip_left_right — defines core processing logic
def aug_flip_left_right(img_arr):
    aug_arr = tf.image.stateless_random_flip_left_right(img_arr, seed).numpy().tolist()
    aur_arr = np.array(aug_arr)
    return aug_arr

# Function: aug_flip_up_down — defines core processing logic
def aug_flip_up_down(img_arr):
    aug_arr = tf.image.stateless_random_flip_up_down(img_arr, seed).numpy().tolist()
    aur_arr = np.array(aug_arr)
    return aug_arr

# Function: aug_hue — defines core processing logic
def aug_hue(img_arr):
    aug_arr = tf.image.stateless_random_hue(img_arr, 0.5, seed)
    aur_arr = np.array(aug_arr)
    return aug_arr

# Function: aug_saturation — defines core processing logic
def aug_saturation(img_arr):
    aug_arr = tf.image.stateless_random_saturation(img_arr, 0.1, 1, seed)
    aur_arr = np.array(aug_arr)
    return aug_arr

func_list = [aug_brightness, aug_contrast, aug_flip_left_right, aug_flip_up_down, aug_hue, aug_saturation]

# Function: random_augmentation — defines core processing logic
def random_augmentation(images, dx, num_of_samples):
    new_images = []
    all_indices = []
    while (len(new_images) < num_of_samples):
        indices = np.random.randint(images.shape[0], size=6)
        r = list(map(lambda x, y: x(y), func_list, images.iloc[indices]))
        r = [np.array(tensor) for tensor in r]
        new_images.extend(r)
        all_indices.extend(indices)
        clear_output(wait=True)
        print(str(len(new_images)) + '/' + str(num_of_samples))
    new_images = pd.DataFrame({'dx': dx, 'image': new_images})
    return all_indices, new_images

df_images = df_df['image']
df_indices, df_new_images = random_augmentation(df_images, 'df', 700)

vasc_images = vasc_df['image']
vasc_indices, vasc_new_images = random_augmentation(vasc_images, 'vasc', 700)

akiec_images = akiec_df['image']
akiec_indices, akiec_new_images = random_augmentation(akiec_images, 'akiec', 1000)

bcc_images = bcc_df['image']
bcc_indices, bcc_new_images = random_augmentation(bcc_images, 'bcc', 1200)

bkl_images = bkl_df['image']
bkl_indices, bkl_new_images = random_augmentation(bkl_images, 'bkl', 1400)

mel_images = mel_df['image']
mel_indices, mel_new_images = random_augmentation(mel_images, 'mel', 1400)

new_images_df = pd.concat([df_new_images, vasc_new_images, akiec_new_images, bcc_new_images, bkl_new_images, mel_new_images], axis=0).reset_index(drop=True)
aug_df = pd.concat([df, new_images_df], axis=0).reset_index(drop=True)
aug_df = aug_df.sample(frac=1).reset_index(drop=True)

pd.DataFrame(aug_df['dx'].value_counts()).reset_index()

aug_df.shape

for i in range(12, 18):
    plt.subplot(1, 3, 1)
    plt.imshow(mel_df['image'].iloc[mel_indices[i]])
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mel_new_images['image'].iloc[i])
    plt.title('Augmented')
    plt.axis('off')

    plt.show()

labelEncoder = LabelEncoder()
aug_df['label'] = labelEncoder.fit_transform(aug_df['dx'])

x = np.asarray(aug_df['image'].to_list())
y = to_categorical(aug_df['label'], num_classes=7)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=42, shuffle=True)

y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Count the occurrences of each class in training and testing sets
train_counts = pd.Series(y_train_labels).value_counts().sort_index()
test_counts = pd.Series(y_test_labels).value_counts().sort_index()

# Display the counts
print("Training Set Class Distribution:")
print(train_counts)

print("\nTesting Set Class Distribution:")
print(test_counts)

import numpy as np
from sklearn.model_selection import train_test_split

# === SMOTE Oversampling ===
from imblearn.over_sampling import SMOTE

# Assuming x_train, y_train, x_test, y_test are already defined
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train_reshaped = x_train.reshape(x_train.shape[0], -1)

smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train_reshaped, y_train.argmax(axis=1))

x_train_smote = x_train_smote.reshape(x_train_smote.shape[0], 64, 64, 3)
y_train_smote = tf.keras.utils.to_categorical(y_train_smote, num_classes=7)

import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Count the occurrences of each class after SMOTE
train_counts_smote = pd.Series(y_train_smote.argmax(axis=1)).value_counts().sort_index()

# Display the counts
print("Training Set Class Distribution After SMOTE:")
print(train_counts_smote)

# Bar plot of class distribution after SMOTE
plt.bar(train_counts_smote.index, train_counts_smote.values)
plt.xlabel('Class Labels')
plt.ylabel('Count')
plt.title('Distribution of Classes After SMOTE')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# === Visualizing SMOTE Augmented Images ===

import matplotlib.pyplot as plt
# Display some SMOTE images and original images for comparison
n_samples = 5
fig, m_axs = plt.subplots(2, n_samples, figsize=(12, 6))

# SMOTE Images
for i in range(n_samples):
    m_axs[0, i].imshow(x_train_smote[i])
    m_axs[0, i].set_title('SMOTE Image')
    m_axs[0, i].axis('off')

# Original Images
for i in range(n_samples):
    m_axs[1, i].imshow(x_train[i])
    m_axs[1, i].set_title('Original Image')
    m_axs[1, i].axis('off')

plt.tight_layout()
plt.show()


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, SeparableConv2D, DepthwiseConv2D, BatchNormalization, ReLU,
    AveragePooling2D, Dropout, Add, Conv2D, Multiply, Reshape, Concatenate
)

# === Model Architecture: Ghost-SE-ECA-CNN ===
from akida_models.layer_blocks import dense_block

# Function: ecablocks — defines core processing logic
def ecablocks(input_tensor, k=3, name="eca"):
    x = DepthwiseConv2D((k, k), padding='same', use_bias=False, name=name+'_dwconv')(input_tensor)
    x = BatchNormalization(name=name+'_bn')(x)
    scale = Conv2D(input_tensor.shape[-1], (1, 1), activation='sigmoid', use_bias=False, name=name+'_conv')(x)
    return Multiply(name=name+'_scale')([input_tensor, scale])

# SE block without GlobalAveragePooling2D
# Function: seblock — defines core processing logic
def seblock(input_tensor, ratio=16, name="se"):
    pool_size = (input_tensor.shape[1], input_tensor.shape[2])
    squeeze = AveragePooling2D(pool_size, padding='valid', name=name+'_avgpool')(input_tensor)
    excitation = Conv2D(input_tensor.shape[-1] // ratio, (1, 1), activation='relu', use_bias=False, name=name+'_fc1')(squeeze)
    excitation = Conv2D(input_tensor.shape[-1], (1, 1), activation='sigmoid', use_bias=False, name=name+'_fc2')(excitation)
    return Multiply(name=name+'_scale')([input_tensor, excitation])

# Ghost-SE-ECA block
# Function: ghostblocks — defines core processing logic
def ghostblocks(x, filters, name_prefix):
    shortcut = x

    # Ghost Module
    x1 = SeparableConv2D(filters // 2, (1, 1), padding='same', use_bias=False, name=f"{name_prefix}_ghost1")(x)
    x2 = SeparableConv2D(filters // 2, (3, 3), padding='same', use_bias=False, name=f"{name_prefix}_ghost2")(x1)
    x = Concatenate(name=f"{name_prefix}_ghost_concat")([x1, x2])
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = ReLU(max_value=6, name=f"{name_prefix}_relu")(x)

    x = Dropout(0.25, name=f"{name_prefix}_dropout")(x)

    # ECA block
    x = ecablocks(x, name=f"{name_prefix}_eca")

    # Residual Shortcut
    if shortcut.shape[-1] != filters:
        shortcut = SeparableConv2D(filters, (1, 1), padding='same', use_bias=False, name=f"{name_prefix}_proj")(shortcut)
        shortcut = BatchNormalization(name=f"{name_prefix}_proj_bn")(shortcut)

    x = Add(name=f"{name_prefix}_add")([x, shortcut])
    x = AveragePooling2D((2, 2), padding='same', name=f"{name_prefix}_pool")(x)
    return x

# create models
inputs = Input(shape=(64, 64, 3), name='input_layer')
x = ghostblocks(inputs, 32, "block1")
x = ghostblocks(x, 64, "block2")
x = ghostblocks(x, 128, "block3")
x = ghostblocks(x, 256, "block4")

# Spike Generator Block (filters=256)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='spike_generator/conv')(x)
x = BatchNormalization(name='spike_generator/bn')(x)
x = ReLU(max_value=1, name='spike_generator/relu')(x)
x = seblock(x, ratio=16, name='spike_generator/se')

# Dense Block
x = dense_block(x, units=7, name='predictions', add_batchnorm=False, relu_activation=False)
x = Reshape((7,), name='reshape')(x)

model_keras = Model(inputs, x, name='akidanet_derma')

model_keras.summary()



# model = sanitize(model)
model_keras.compile(Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

print(x_train_smote.shape, y_train_smote.shape)
print(x_test.shape, y_test.shape)


# List all GPUs recognized by TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print("GPUs:", gpus)


# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
# Training the model using the defined architecture and training data
history = model_keras.fit(x_train_smote, y_train_smote,
                    batch_size=32,
                    epochs=60,
                    validation_data=(x_test, y_test),
                    callbacks=[reduce_lr, early_stopping],
                    verbose=1)


# Evaluating the model performance on test data
test_loss, test_accuracy = model_keras.evaluate(x_test, y_test)
initial_model = model_keras

# === Accuracy Curves and Visualization ===
print('Test Accuracy:', test_accuracy)
print('Test Loss:', test_loss)


# Get the number of images in the training dataset
num_images = len(x_train_smote)

# Measure the time taken for Keras inference
start = timer()
potentials_keras = model_keras.predict(x_train_smote, batch_size=100)
end = timer()
print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')


# === Inference Benchmark and Top-5 Visualization ===
# Get the predicted labels from the model output
preds_keras = np.argmax(potentials_keras, axis=1)

# Convert one-hot encoded y_train_smote back to class indices
y_train_smote_labels = np.argmax(y_train_smote, axis=1)

# Calculate the accuracy
accuracy_keras = np.sum(np.equal(preds_keras, y_train_smote_labels)) / num_images
print(f"Keras accuracy: {accuracy_keras*num_images:.0f}/{num_images}.")

# Generate the confusion matrix
# Computing and visualizing the confusion matrix
cm = confusion_matrix(y_train_smote_labels, preds_keras)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_train_smote_labels), 
            yticklabels=np.unique(y_train_smote_labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

model_keras.summary()

model_keras.save('initial_model83.h5')


import tensorflow as tf

# Assuming x_train is your image data and y_train is your labels
x_train = x_train_smote 
y_train = y_train_smote
IMG_SIZE = 64  

# Function: format_example — defines core processing logic
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# Create a Dataset from x_train and y_train
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Apply the format_example function
train_dataset = train_dataset.map(format_example, num_parallel_calls=tf.data.AUTOTUNE)



# === Model Quantization and SNN Conversion ===
model_quantized = cnn2snn.quantize(model_keras,
                                   weight_quantization=4,
                                   activ_quantization=4,
                                   input_weight_quantization=8)

model_quantized = cnn2snn.quantize_layer(model_quantized, 'spike_generator/relu', 1)

model_quantized.summary()

# Get the number of images in the training dataset
num_images = len(x_train_smote)

# Measure the time taken for Keras inference
start = timer()
potentials_keras = model_quantized.predict(x_train_smote, batch_size=100)
end = timer()
print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')

# Get the predicted labels from the model output
preds_keras = np.argmax(potentials_keras, axis=1)

# Convert one-hot encoded y_train_smote back to class indices
y_train_smote_labels = np.argmax(y_train_smote, axis=1)

# Calculate the accuracy
accuracy_keras = np.sum(np.equal(preds_keras, y_train_smote_labels)) / num_images
print(f"Keras accuracy: {accuracy_keras*num_images:.0f}/{num_images}.")

# Generate the confusion matrix
# Computing and visualizing the confusion matrix
cm = confusion_matrix(y_train_smote_labels, preds_keras)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_train_smote_labels), 
            yticklabels=np.unique(y_train_smote_labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Function: compile_evaluate — defines core processing logic
def compile_evaluate(model):
    """ Compiles and evaluates the model, then return accuracy score. """
    model.compile(metrics=['accuracy'])
# Evaluating the model performance on test data
    return model.evaluate(x_test, y_test, verbose=0)[1]


# Evaluating the model performance on test data
print('Test accuracy after 8-bit quantization:', compile_evaluate(model_quantized))

from timeit import default_timer as timer
import numpy as np

num_images = len(x_test)

# Start timer
start = timer()

# Perform inference
potentials_keras = model_quantized.predict(x_test, batch_size=100)

# End timer
end = timer()
print(f'Keras inference on {num_images} images took {end-start:.2f} s.\n')

# Get the predicted class indices
preds_keras = np.squeeze(np.argmax(potentials_keras, 1))

# Convert y_test from one-hot encoded format to categorical format
y_test_categorical = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy_keras = np.sum(np.equal(preds_keras, y_test_categorical)) / num_images
print(f"Keras accuracy: {accuracy_keras*num_images:.0f}/{num_images}. {accuracy_keras*num_images/num_images}%")

model_quantized.save('model_quantized.h5')

import numpy as np
from cnn2snn import convert

model_akida = convert(model_quantized)
model_akida.summary()
model_akida.save('model_akida.fbz')

# Check Model performance
start = timer()
y_test_categorical = np.argmax(y_test, axis=1)
# Evaluating the model performance on test data
accuracy_akida = model_akida.evaluate(x_test, y_test_categorical)
end = timer()
print(f'Inference on {num_images} images took {end-start:.2f} s.\n')
print(f"Accuracy: {accuracy_akida*num_images:.0f}/{num_images}.")

y_test_categorical = np.argmax(y_test, axis=1)
y_test_categorical

import numpy as np

# Convert y_test from one-hot encoded to categorical format
y_test_categorical = np.argmax(y_test, axis=1)

# Count the occurrences of each class in the categorical y_test
class_counts = np.bincount(y_test_categorical)
print(class_counts)

import matplotlib.pyplot as plt
import matplotlib.lines as lines
from akida_models.imagenet import preprocessing

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

IMAGE_SIZE = 224
NUM_CHANNELS = 3

# Functions used to display the top5 results
# Function: get_top5 — defines core processing logic
def get_top5(potentials, true_label):
    """
    Returns the top 5 classes from the output potentials
    """
    tmp_pots = potentials.copy()
    top5 = []
    min_val = np.min(tmp_pots)
    for ii in range(5):
        best = np.argmax(tmp_pots)
        top5.append(best)
        tmp_pots[best] = min_val
    vals = np.zeros((6,))
    vals[:5] = potentials[top5]
    if true_label not in top5:
        vals[5] = potentials[true_label]
    else:
        vals[5] = 0
    vals /= np.max(vals)
    class_name = []
    for ii in range(5):
        class_name.append(class_names[top5[ii]])
    if true_label in top5:
        class_name.append('')
    else:
        class_name.append(
            class_names[true_label])

    return top5, vals, class_name


# Function: adjust_spines — defines core processing logic
def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine
    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


# Function: prepare_plots — defines core processing logic
def prepare_plots():
    fig = plt.figure(figsize=(8, 4))
    # Image subplot
    ax0 = plt.subplot(1, 3, 1)
    imgobj = ax0.imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.uint8))
    ax0.set_axis_off()
    # Top 5 results subplot
    ax1 = plt.subplot(1, 2, 2)
    bar_positions = (0, 1, 2, 3, 4, 6)
    rects = ax1.barh(bar_positions, np.zeros((6,)), align='center', height=0.5)
    plt.xlim(-0.2, 1.01)
    ax1.set(xlim=(-0.2, 1.15), ylim=(-1.5, 12))
    ax1.set_yticks(bar_positions)
    ax1.invert_yaxis()
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks([])
    adjust_spines(ax1, 'left')
    ax1.add_line(lines.Line2D((0, 0), (-0.5, 6.5), color=(0.0, 0.0, 0.0)))
    # Adjust Plot Positions
    ax0.set_position([0.05, 0.055, 0.3, 0.9])
    l1, b1, w1, h1 = ax1.get_position().bounds
    ax1.set_position([l1 * 1.05, b1 + 0.09 * h1, w1, 0.8 * h1])
    # Add title box
    plt.figtext(0.5,
                0.9,
                "Imagenet Classification by Akida",
                size=20,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round",
                          ec=(0.5, 0.5, 0.5),
                          fc=(0.9, 0.9, 1.0)))

    return fig, imgobj, ax1, rects


# Function: update_bars_chart — defines core processing logic
def update_bars_chart(rects, vals, true_label):
    counter = 0
    for rect, h in zip(rects, yvals):
        rect.set_width(h)
        if counter < 5:
            if top5[counter] == true_label:
                if counter == 0:
                    rect.set_facecolor((0.0, 1.0, 0.0))
                else:
                    rect.set_facecolor((0.0, 0.5, 0.0))
            else:
                rect.set_facecolor('gray')
        elif counter == 5:
            rect.set_facecolor('red')
        counter += 1


# Prepare plots
fig, imgobj, ax1, rects = prepare_plots()

# Get a random image
img = np.random.randint(num_images)

# Predict image class
outputs_akida = model_akida.predict(np.expand_dims(x_test[img].astype(np.uint8), axis=0)).squeeze()

# Get top 5 prediction labels and associated names
true_label = y_test_categorical[img]
top5, yvals, class_name = get_top5(outputs_akida, true_label)

print("Akida outputs:", outputs_akida)
print("True label:", class_names[int(true_label)])
print("Predicted label:", class_names[np.argmax(outputs_akida)])

# Draw Plots
imgobj.set_data(x_test[img])
ax1.set_yticklabels(class_name, rotation='horizontal', size=9)
update_bars_chart(rects, yvals, true_label)
fig.canvas.draw()
plt.show()

tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(1, len(tr_acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, tr_loss, 'r', label='Train Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, tr_acc, 'r', label='Train Accuracy')
plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

tr_acc = history.history['accuracy']

epochs = range(1, len(tr_acc) + 1)

plt.figure(figsize=(10, 6))

plt.plot(epochs, tr_acc)
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

tr_acc = history.history['val_loss']

epochs = range(1, len(tr_acc) + 1)

plt.figure(figsize=(10, 6))

plt.plot(epochs, tr_acc)
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

predictions = model.predict(x_test)

# Computing and visualizing the confusion matrix
cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
cm

cm_plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']

# Computing and visualizing the confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm, class_names=cm_plot_labels, colorbar=True)
plt.show()

report = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=cm_plot_labels, output_dict=True)

df_classification_report = pd.DataFrame(report).transpose()
df_classification_report.drop(["accuracy", "macro avg", "weighted avg"], inplace=True)
df_classification_report = df_classification_report.sort_values(by=['f1-score'], ascending=False)
df_classification_report["precision"] = df_classification_report["precision"].map(lambda x: round(x, 3))
df_classification_report["recall"] = df_classification_report["recall"].map(lambda x: round(x, 3))
df_classification_report["f1-score"] = df_classification_report["f1-score"].map(lambda x: round(x, 3))
df_classification_report["support"] = df_classification_report["support"].map(int)
df_classification_report

# === Spike Estimation for Akida ===
model_akida.pop_layer()
num_samples_to_use = int(len(x_train) / 10)
spikes = model_akida.forward(x_train[:num_samples_to_use].astype(np.uint8))

# Compute the median of the number of output spikes

# === Spike Estimation for Akida ===
median_spikes = np.median(spikes.sum(axis=(1, 2, 3)))
print(f"Median of number of spikes: {median_spikes}")

# Set the number of weights to 1.2 x median
num_weights = int(1.2 * median_spikes)
print(f"The number of weights is then set to: {num_weights}")