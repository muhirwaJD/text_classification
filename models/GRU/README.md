# GRU Model Implementation with Multiple Embeddings
## E-commerce Product Classification

## Overview

This project implements a **Gated Recurrent Unit (GRU)** neural network for multi-class text classification on e-commerce product data. The primary objective is to compare the performance of different text embedding techniques when combined with GRU architecture.

### Assignment Goals

- Implement GRU model architecture for text classification
- Compare 5 different text embedding techniques
- Evaluate model performance with comprehensive metrics
- Visualize results with learning curves and confusion matrices
- Provide systematic documentation and analysis


##  Embeddings Implemented

This project compares **5 different embedding techniques**:

| # | Embedding Type | Architecture | Key Characteristics |
|---|----------------|--------------|---------------------|
| 1 | **TF-IDF** | Dense NN | Document-level features, sparse representation |
| 2 | **Word2Vec Skip-gram** | Bidirectional GRU | Predicts context from target word, better for rare words |
| 3 | **Word2Vec CBOW** | Bidirectional GRU | Predicts target from context, faster training |
| 4 | **GloVe** | Bidirectional GRU | Pre-trained, captures global co-occurrence statistics |
| 5 | **FastText** | Bidirectional GRU | Subword information, handles OOV words effectively |

### Why These Embeddings?

- **TF-IDF**: Baseline, traditional NLP approach
- **Word2Vec (Skip-gram & CBOW)**: Context-aware, domain-specific training
- **GloVe**: Leverages pre-trained knowledge from large corpus
- **FastText**: Robust to spelling variations and out-of-vocabulary words


## Requirements

### Software Requirements

- Python 3.8+
- TensorFlow 2.x
- Google Colab (optional, recommended for GPU)

### Python Libraries

```
tensorflow>=2.8.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
gensim>=4.1.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

##  Installation

### Local Setup

```bash
# Clone repository
git clone <your-repo-url>
cd gru-text-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download GloVe Embeddings

```bash
# Download GloVe 6B 100d embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
mv glove.6B.100d.txt embeddings/
```


## Dataset

### Dataset Description

- **Domain**: E-commerce product classification
- **Task**: Multi-class text classification
- **Classes**: 4 categories (Books, Clothing & Accessories, Electronics, Household)
- **Data Split**: Data was split into train, validation, and test sets

### Expected Data Format

CSV files with two columns:

| Column | Description | Example |
|--------|-------------|---------|
| `text` | Product description | "Wireless Bluetooth headphones with noise cancellation" |
| `label` | Product category | "Electronics" |

### Dataset Statistics

```
Training samples:   ~35,000
Validation samples: ~9,000
Test samples:       ~7,500
Total samples:      ~51,500
```

---

##  Usage

### Quick Start

```python
# 1. Open the notebook in Google Colab or Jupyter

# 2. Update file paths to your data
train_df = pd.read_csv('path/to/train.csv')
val_df = pd.read_csv('path/to/val.csv')
test_df = pd.read_csv('path/to/test.csv')

# 3. Run all cells in order

# 4. View results and visualizations
```

## Model Architecture

### GRU Model Structure

```
Input (Padded Sequences)
    ↓
Embedding Layer (pre-trained or trainable)
    ↓
Bidirectional GRU (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Bidirectional GRU (64 units)
    ↓
Dropout (0.3)
    ↓
Dense (64 units, ReLU activation)
    ↓
Dropout (0.5)
    ↓
Dense (4 units, Softmax activation)
    ↓
Output (Class Probabilities)
```

### Training Configuration

- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Batch Size**: 64
- **Max Epochs**: 20
- **Callbacks**: 
  - EarlyStopping (patience=5, monitor='val_loss')
  - ReduceLROnPlateau (factor=0.5, patience=3)

---

## Results

### Performance Summary

Path to the performance summary: D:\text_classification\results\gru\model_comparison.csv

### Best Model

**Model**: GRU with Word2Vec
- **Test Accuracy**: 0.9754
- **Test F1-Score**: 0.9754
- **Test Precision**: 0.9755
- **Test Recall**: 0.9754

---

##  Visualizations

The notebook generates the following visualizations for each experiment:

### 1. Learning Curves
- Training vs Validation Accuracy
- Training vs Validation Loss
- Helps identify overfitting/underfitting

### 2. Confusion Matrices
- Heatmap showing prediction distribution
- Per-class performance analysis
- Identifies common misclassifications

### 3. Comparative Analysis
- Bar charts comparing all embeddings
- Accuracy and F1-score side-by-side
- Final comparison visualization saved as `embedding_comparison.png`

---

##  Key Findings

### Embedding Performance Analysis

1. **TF-IDF + Dense NN**
   -  Fast training, interpretable features
   -  No sequential information, sparse representation

2. **Word2Vec Skip-gram + GRU**
   -  Good for rare words, captures context
   -  Slower training than CBOW
   - **Use Case**: Domain-specific vocabulary

3. **Word2Vec CBOW + GRU**
   -  Faster training, predicts target from context
   -  Less effective for rare words
   - **Use Case**: Large, frequent vocabulary

4. **GloVe + GRU**
   -  Pre-trained, no training time, general knowledge
   -  May not capture domain-specific terms
   - **Use Case**: Transfer learning, low-resource scenarios

5. **FastText + GRU**
   -  Handles OOV words, subword information
   -  Larger model size
   - **Use Case**: Noisy text, spelling variations


---

## Team member

**Nice Eva Karabaranga**