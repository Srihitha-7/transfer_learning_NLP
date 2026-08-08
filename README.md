# TensorFlow Hub Text Classification

A TensorFlow/Keras text-classification experiment using pretrained text embedding models from TensorFlow Hub.

## Overview

This project:

1. Downloads a text-classification dataset from Archive.org.
2. Splits the data into training and validation subsets using stratified sampling.
3. Converts question text into fixed-size embeddings using TensorFlow Hub.
4. Trains a small feed-forward neural network classifier on top of the embeddings.
5. Compares multiple pretrained embedding models using accuracy and loss curves.
6. Visualizes the training results with TensorBoard and TensorFlow Docs plotting utilities.

## Models evaluated

The notebook evaluates these TensorFlow Hub modules:

- GNews Swivel 20-dimensional embeddings
- NNLM English 50-dimensional embeddings
- NNLM English 128-dimensional embeddings
- Universal Sentence Encoder
- Universal Sentence Encoder Large

The notebook also contains an experiment named `gnews-swivel-20dim-finetuned`.

> **Important:** In the current notebook, the `trainable` argument is passed to `train_and_evaluate_model()` but is not actually used when loading the embedding layer. Therefore, the `gnews-swivel-20dim-finetuned` experiment should not be described as true end-to-end embedding fine-tuning without modifying the model code.

## Dataset

The notebook downloads:

`https://archive.org/download/fine-tune-bert-tensorflow-train.csv/train.csv.zip`

The dataset contains approximately 1.31 million rows in the original CSV. The notebook uses a small stratified subset for training and validation:

- Training: 13,061 samples
- Validation: 1,293 samples
- Random state: 42

The dataset is downloaded automatically when the notebook is executed, so the full dataset is **not stored in this repository**.

## Model architecture

For each embedding model, the classifier is:

```text
Input embedding
      ↓
Dense(256, ReLU)
      ↓
Dense(64, ReLU)
      ↓
Dense(1, Sigmoid)
```

Training configuration:

- Optimizer: Adam
- Learning rate: 0.0001
- Loss: Binary Crossentropy
- Metric: Binary Accuracy
- Batch size: 32
- Maximum epochs: 100
- Early stopping: validation loss, patience 2
- Best weights restored after early stopping

## Repository structure

```text
text-classification-tensorflow-hub/
│
├── text_classification_tensorflow_hub.ipynb
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Installation

Python 3.12 is recommended for the environment used for this notebook.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the notebook

Start Jupyter:

```bash
jupyter notebook
```

Then open:

```text
text_classification_tensorflow_hub.ipynb
```

The notebook can also be run in Google Colab. A GPU is recommended for faster experimentation.

## Reproducibility

The notebook uses:

- TensorFlow
- TensorFlow Hub
- TensorFlow Datasets
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorFlow Docs

The dataset and TensorFlow Hub embedding modules are downloaded at runtime.

## Results

The notebook compares models using:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

The original notebook contains the experiment outputs and plots. The GitHub version intentionally removes stored cell outputs to keep the repository clean and lightweight; rerun the notebook to regenerate them.

## Notes

This repository represents the code as it currently exists in the supplied Colab notebook. In particular, the embedding modules are loaded through `hub.load()`, and the classifier is trained on the resulting embeddings.

If you want a **true fine-tuning version**, the embedding module should be incorporated into a Keras model as a trainable layer and its weights should be allowed to update during `model.fit()`.

## License

MIT License.
