# Methodology

This document gives a short technical explanation of the pipeline. The goal is to
make the implementation easy to inspect and reproduce.

## 1. Image preparation

Images are loaded in BGR format with OpenCV. CLAHE improves local contrast on the
LAB lightness channel. A black-hat operation finds dark hair, and Telea inpainting
reduces it. Large images are resized while keeping their aspect ratio.

## 2. Lesion segmentation

K-means divides the LAB pixels into four color groups. The darker valid groups are
used as an initial lesion region. A separate mask removes dark circular borders
that may be present in dermoscopic images. GrabCut refines the initial region, and
morphological operations remove noise and fill small gaps.

The generated masks are cached inside `HAM10000_organizado/segmentado`. A cached
mask is reused only when its dimensions match the processed image.

## 3. Feature extraction

The classifier receives 46 numerical features in six groups:

| Group | Count | Description |
| --- | ---: | --- |
| Geometry | 7 | Area, perimeter, circularity, solidity, aspect ratio, extent, and border irregularity |
| Hu moments | 7 | Shape values that are mostly invariant to rotation and scale |
| HSV color | 12 | Mean, standard deviation, skewness, and kurtosis for three channels |
| LAB color | 12 | The same statistics in the LAB color space |
| Texture | 6 | Five GLCM properties and Shannon entropy |
| Asymmetry | 2 | Difference after horizontal and vertical reflection |

Invalid numeric values are replaced with zero before training or prediction.

## 4. Training and threshold selection

The CSV is divided into 60% training, 20% validation, and 20% test splits. Each
split is stratified and uses a fixed random seed. A standard scaler is fitted only
on the training data, followed by a 300-tree Random Forest with balanced class
weights.

Several probability thresholds are evaluated on the validation split. The selected
threshold is the one with the highest F1 score among candidates with at least 90%
melanoma recall. If no candidate reaches that recall, the best F1 score is used.
Final metrics are calculated once on the test split. The trained model, scaler,
threshold, library version, and ordered feature names are saved together with
Joblib.

## 5. Evaluation limits

The reported metrics describe only one test split. They do not prove
clinical safety or performance on images from another camera, hospital, or patient
group. A stronger study should use patient-level splitting, cross-validation, an
external dataset, confidence intervals, and review by medical specialists.
