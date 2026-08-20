# Melanoma Detection with Classical Computer Vision

An experimental Python project that segments skin lesions and classifies them as
melanoma or melanocytic nevus. It uses OpenCV, K-means, GrabCut, handcrafted image
features, and a Random Forest classifier.

> This project is for study and research only. It is not a medical device and must
> not be used for diagnosis or treatment decisions.

## How it works

1. Improves image contrast and reduces visible hair.
2. Uses K-means and GrabCut to create a lesion mask.
3. Extracts 46 shape, color, texture, and asymmetry features.
4. Trains a Random Forest and selects a decision threshold on a validation split.
5. Applies the same pipeline to a new image.

More details are available in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Dataset

The images are not included in this repository. Download HAM10000 from its
official source and organize the two classes used by this project:

```text
HAM10000_organizado/
├── melanoma/
└── nevus_melanocitico/
```

Keep the original dataset license and citation. Do not commit the image folders;
they are ignored by Git.

## Usage

Create masks only:

```bash
melanoma-detection segment
```

Create the feature table:

```bash
melanoma-detection build-dataset
```

Train and evaluate the model:

```bash
melanoma-detection train
```

Classify one image:

```bash
melanoma-detection predict path/to/image.jpg
```

Add `--show` to display the detected lesion boundary. Every command accepts
`--help`; dataset, CSV, and model paths can also be changed through command options.
The commands may be run without installation as `python3 main.py <command>`.

## Project structure

```text
src/melanoma_detection/
├── cli.py             # Command-line interface
├── preprocessing.py   # Contrast, hair reduction, and resize steps
├── segmentation.py    # K-means and GrabCut lesion mask
├── features.py        # Handcrafted feature extraction
├── dataset.py         # Mask cache and CSV generation
├── model.py           # Training and evaluation
└── prediction.py      # Single-image prediction
```

## Limitations

- Results depend on image quality and correct lesion segmentation.
- The current evaluation uses one stratified train/validation/test split.
- The model supports only melanoma and melanocytic nevus.
- A probability from this model is a software output, not a clinical probability.

## License

The source code is available under the [MIT License](LICENSE). The HAM10000 dataset
has its own terms and is not covered by this repository's license.
