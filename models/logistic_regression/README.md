# Raissa's Part: Traditional Machine Learning for Text Classification

## Overview
This folder contains the implementation of **Traditional Machine Learning models** for BBC news article classification.

## Assigned Tasks

### Model Architecture
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**

### Required Embeddings (Minimum 3)
1. **TF-IDF** (Term Frequency-Inverse Document Frequency)
2. **GloVe** (Global Vectors for Word Representation)
3. **Word2Vec** (Word to Vector)

## Folder Structure
```
raissa_traditional_ml/
├── README.md                          # This file
├── traditional_ml_classification.ipynb # Main implementation notebook
├── models/                            # Saved trained models
└── results/                           # Performance metrics and visualizations
```

## Implementation Plan

### 1. Data Preparation
- Load BBC news dataset from `../data/bbc/`
- Preprocess text (tokenization, cleaning, stopword removal)
- Split data into train/validation/test sets

### 2. Feature Extraction (Embeddings)
- **TF-IDF**: Extract TF-IDF features using scikit-learn
- **GloVe**: Load pre-trained GloVe embeddings and average word vectors
- **Word2Vec**: Train Word2Vec model on corpus or use pre-trained embeddings

### 3. Model Training
- Train Random Forest classifier with each embedding type
- Train SVM classifier with each embedding type
- Hyperparameter tuning using GridSearchCV/RandomizedSearchCV

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Cross-validation scores

### 5. Comparison
- Compare performance across different embeddings
- Compare Random Forest vs SVM
- Visualize results

## Expected Deliverables
- [x] Jupyter notebook with complete implementation
- [ ] Trained models (Random Forest & SVM with 3 embeddings = 6 models total)
- [ ] Performance comparison report
- [ ] Visualizations (confusion matrices, performance charts)
- [ ] Analysis and conclusions

## Getting Started
1. Install required dependencies (see notebook for details)
2. Open `traditional_ml_classification.ipynb`
3. Run cells sequentially

## Required Libraries
- scikit-learn (Random Forest, SVM, TF-IDF)
- gensim (Word2Vec)
- numpy, pandas
- matplotlib, seaborn (visualization)
- nltk (text preprocessing)

## Team Member
**Raissa** - Traditional ML Implementation
