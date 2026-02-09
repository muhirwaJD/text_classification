# Text Classification Project

## Overview

This project implements and compares various machine learning and deep learning models for text classification using an e-commerce dataset. The goal is to understand how different models and text embedding techniques perform on classifying product descriptions or reviews.

## Project Objectives

- Implement multiple text classification models (Logistic Regression, RNN, LSTM, GRU)
- Compare different text embedding methods (TF-IDF, Word2Vec, FastText)
- Evaluate and compare model performance
- Understand the strengths and weaknesses of each approach

## Dataset

The project uses an e-commerce dataset located in `data/e-commerce/ecommerceDataset.csv`. The data has been preprocessed and split into:
- Training set: `data/preprocessed_data/train.csv`
- Validation set: `data/preprocessed_data/validation.csv`
- Test set: `data/preprocessed_data/test.csv`

## Models Implemented

### 1. Logistic Regression
Traditional machine learning approach using:
- TF-IDF embeddings
- Word2Vec (CBOW and Skip-gram)
- FastText embeddings

Location: `models/logistic_regression/`

### 2. Recurrent Neural Network (RNN)
Basic sequence model for text classification.

Location: `models/rnn/`

### 3. Long Short-Term Memory (LSTM)
Advanced sequence model that handles long-term dependencies better than RNN.

Location: `models/lstm/`

### 4. Gated Recurrent Unit (GRU)
Simplified variant of LSTM with fewer parameters.

Location: `models/GRU/`

## Project Structure

```
text_classification/
├── data/                           # Dataset directory
│   ├── e-commerce/                 # Raw e-commerce data
│   └── preprocessed_data/          # Train/validation/test splits
├── models/                         # Model implementations
│   ├── logistic_regression/        # Traditional ML with various embeddings
│   ├── rnn/                        # RNN implementation
│   ├── lstm/                       # LSTM implementation
│   └── GRU/                        # GRU implementation
└── results/                        # Model evaluation results
    ├── logistic_regression/        # LR performance metrics
    ├── rnn/                        # RNN performance metrics
    ├── lstm/                       # LSTM performance metrics
    └── gru/                        # GRU performance metrics
```

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository or download the project files

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Mac/Linux
```

3. Install required packages:
```bash
pip install numpy pandas scikit-learn tensorflow keras gensim nltk matplotlib seaborn jupyter
```

## How to Run

Each model is implemented in a Jupyter notebook within its respective folder:

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Navigate to the model you want to run:
   - Logistic Regression: `models/logistic_regression/traditional_ml_classification.ipynb`
   - RNN: `models/rnn/rnn-implementation.ipynb`
   - LSTM: `models/lstm/lstm_ml_classification.ipynb`
   - GRU: `models/GRU/gru_implementation.ipynb`

3. Run the cells in order to train and evaluate the model

## Results

Model performance metrics are saved in the `results/` directory:
- `model_comparison.csv`: Overall model performance comparison
- `per_class_metrics.csv`: Detailed per-category performance metrics

Each model folder also contains its own README with specific implementation details and results.

## Key Concepts

### Text Embeddings
Text embeddings convert words into numerical vectors that machines can process:
- **TF-IDF**: Measures word importance based on frequency
- **Word2Vec**: Learns word representations based on context
- **FastText**: Similar to Word2Vec but handles rare words better

### Model Types
- **Logistic Regression**: Simple, interpretable, works well as a baseline
- **RNN**: Processes sequences but may struggle with long texts
- **LSTM**: Better at remembering long-term patterns in text
- **GRU**: Similar to LSTM but computationally more efficient

## Learning Resources

For detailed information about specific models and their implementations, check the README files in each model directory.

## License

See LICENSE file for details.
