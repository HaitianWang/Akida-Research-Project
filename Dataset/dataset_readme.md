# HAM10000 Dataset README

## Dataset Overview

The **HAM10000** dataset (Human Against Machine with 10000 training images) is a curated collection of dermatoscopic images representing the most common pigmented skin lesions encountered in clinical practice. It is widely used for benchmarking machine learning models for skin cancer classification tasks.

This dataset is designed to support the training, validation, and evaluation of neural networks for **automated dermatological diagnosis**.

## Contents

- Total images: **10,015 dermatoscopic RGB images**
- Image resolution: Varies, but standardized to 600×450 px in original
- Format: `.jpg` images with associated metadata in `.csv`
- Metadata includes fields like: image ID, lesion type, diagnosis, anatomical site, patient ID, and more

## Diagnosis Categories

Each image is labeled with one of seven diagnostic categories:

| Label  | Description                                                                                 |
|--------|---------------------------------------------------------------------------------------------|
| akiec | Actinic keratoses and intraepithelial carcinoma / Bowen's disease                           |
| bcc   | Basal cell carcinoma                                                                         |
| bkl   | Benign keratosis-like lesions (solar lentigines, seborrheic keratoses, lichen-planus-like)  |
| df    | Dermatofibroma                                                                               |
| mel   | Melanoma                                                                                     |
| nv    | Melanocytic nevi                                                                            |
| vasc  | Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)                |


### Download
Download the Skin Cancer MNIST: HAM10000 dataset from Kaggle:
[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download)

## Recommended Structure

To use this dataset with our pipeline, place files in the following structure:

```
archive/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

## License

The dataset is publicly available for research and non-commercial use. Refer to the [Kaggle dataset page](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) for full licensing terms.
