# Akida-Research-Project

## Getting Started

1. Installing [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Creating conda environment
    - `conda env create -f environment.yml`
    - `conda config --add channels conda-forge`
3. Activating the environment
    - `conda activate akida`
4. Download the Skin Cancer MNIST: HAM10000 dataset from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download
5. Unzip it and move into root folder of repo

## Dataset Description

The HAM10000 dataset is a collection of dermascopic images of common pigmented skin lesions used for training of neural networks for automated diagnosis. It comprises of 10,015 dermatoscopic images sourced from various populations and captured through diverse modalities. It encompasses a comprehensive spectrum of diagnostic categories critical to the study of pigmented lesions. These categories include Actinic keratoses and intraepithelial carcinoma/Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (comprising solar lentigines, seborrheic keratoses, and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions (such as angiomas, angiokeratomas, pyogenic granulomas, and hemorrhage, vasc). This dataset is tailored to aid in the training and development of machine learning models in the field of dermatology.

## Model Training - working_CNN.ipynb

1. Run command: `sudo docker run --gpus all -it --rm -v $(pwd):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:2.12.0-gpu-jupyter`
2. Modify necessary hyperparameters (model, optimizer, learning rate etc)
3. Run notebook
4. Output will generate 3 files: 
    - `initial_model83.h5`: Custom keras model with 32-bit floating point weights
    - `model_quanitzed.h5`: Quantized version of keras model 
    - `model_akida.fbz`: Quantized model that has undergone Akida conversion

## One-Shot Learning - edge_training.ipynb

1. Run Akida conda environment 
2. Modify necessary hyperparameters (num_neurons, num_weights, learning_competition)
3. Run notebook
4. Output will generate file: `model_edge_trained.fbz`
