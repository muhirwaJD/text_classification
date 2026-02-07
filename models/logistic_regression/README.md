# Traditional Machine Learning for E-commerce Text Classification

## Overview

This module implements a comprehensive comparative analysis of traditional machine learning approaches for e-commerce product text classification. The implementation evaluates three different word embedding techniques across multiple classifier architectures, providing insights into the effectiveness of various feature representation strategies.

## Author Information
- **Module**: Traditional Machine Learning (Logistic Regression)
- **Embeddings**: Word2Vec Skip-gram, Word2Vec CBOW, FastText
- **Dataset**: E-commerce Product Descriptions
- **Date**: February 6, 2026

## Project Structure

```
logistic_regression/
├── README.md                                    # This file
├── traditional_ml_classification.py             # Main implementation (Part 1: EDA & Embeddings)
├── model_training.py                            # Model training and evaluation (Part 2)
├── requirements.txt                             # Python dependencies
├── fasttext_model.bin                           # Trained FastText model (generated)
└── *.pkl                                        # Trained classifier models (generated)
```

## Methodology

### 1. Exploratory Data Analysis
Comprehensive dataset exploration including:
- Class distribution analysis across train/validation/test splits
- Text length statistics (character count, word count, average word length)
- Vocabulary analysis and frequency distributions
- Category-specific vocabulary examination
- 4+ visualizations as per rubric requirements

### 2. Embedding Techniques

#### Word2Vec Skip-gram
- **Type**: Dense neural word embeddings
- **Approach**: Predicts context words from target word
- **Configuration**:
  - Vector size: 100
  - Window: 5
  - Min count: 2
  - Epochs: 20
  - Trained on domain data
  - Document-level averaging for classification
- **Advantages**: Better for rare words, captures semantic relationships
- **References**: Mikolov et al. (2013)

#### Word2Vec CBOW (Continuous Bag of Words)
- **Type**: Dense neural word embeddings
- **Approach**: Predicts target word from context words
- **Configuration**:
  - Vector size: 100
  - Window: 5
  - Min count: 2
  - Epochs: 20
  - Trained on domain data
  - Document-level averaging for classification
- **Advantages**: Faster training, better for frequent words
- **References**: Mikolov et al. (2013)

#### FastText
- **Type**: Character n-gram based embeddings
- **Approach**: Subword information for handling OOV words and morphology
- **Configuration**:
  - Vector size: 100
  - Window: 5
  - CBOW architecture
  - Character n-grams: 3-6
  - Epochs: 20
  - Min count: 2
  - Trained on domain data
  - Document-level averaging for classification
- **References**: Bojanowski et al. (2017)

### 3. Classification Model

#### Logistic Regression
- **Type**: Linear discriminative classifier
- **Hyperparameters**: C=[0.1, 1.0, 10.0, 100.0], penalty=['l2'], solver=['lbfgs', 'liblinear']
- **Optimization**: L-BFGS and LIBLINEAR solvers with GridSearchCV
- **Cross-validation**: 3-fold stratified cross-validation
- **Class weighting**: Balanced to handle class imbalance
- **Advantages**: Fast training, interpretable, effective with dense embeddings
- **References**: Fan et al. (2008)

### 4. Experimental Design
- **Total experiments**: 3 embedding combinations
- **Model**: Logistic Regression with each embedding type
- **Hyperparameter tuning**: GridSearchCV with 3-fold stratified cross-validation
- **Evaluation metric**: Weighted F1-score (primary), Accuracy, Precision, Recall
- **Class imbalance handling**: Balanced class weights
- **Consistency**: Uses same embeddings as team's RNN implementation for comparison

## Setup & Installation

### Prerequisites
```bash
Python 3.8+
```

### Installation
```bash
# Navigate to project root
cd c:\Users\pc\Desktop\text_classification

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn gensim nltk joblib

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Note: Word2Vec and FastText models are trained on the dataset
# No need to download pre-trained embeddings
```

## Usage

### Running the Complete Pipeline

```bash
# Step 1: Run EDA and create embeddings
python models/logistic_regression/traditional_ml_classification.py

# Step 2: Train models and evaluate
python models/logistic_regression/model_training.py
```

### Quick Start (All-in-One)
```bash
# Run both scripts sequentially
python models/logistic_regression/traditional_ml_classification.py && python models/logistic_regression/model_training.py
```

## Results & Deliverables

### Generated Outputs

#### Visualizations
All saved in `results/logistic_regression/`:
1. `class_distribution.png` - Class balance across datasets
2. `text_length_analysis.png` - Text statistics distributions
3. `vocabulary_analysis.png` - Word frequency patterns
4. `performance_comparison.png` - Model performance metrics
5. `f1_heatmap.png` - F1-score heatmap
6. `confusion_matrices.png` - Confusion matrices for best models

#### Data Files
1. `model_comparison.csv` - Complete results table with all metrics
2. Trained models: `*.pkl` files for each model-embedding combination
3. `fasttext_model.bin` - Trained FastText embeddings

#### Comparison Tables
The `model_comparison.csv` includes:
- Embedding type
- Model architecture
- Accuracy, Precision, Recall, F1-Score
- Training time, Inference time
- Best hyperparameters

### Performance Metrics
All models evaluated on:
- **Accuracy**: Overall classification correctness
- **Precision**: Weighted average across classes
- **Recall**: Weighted average across classes
- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **Training Time**: Model fitting duration
- **Inference Time**: Prediction speed on test set

## Key Findings

### Embedding-Model Synergy
Different Word2Vec-based embeddings show varying performance:
- Skip-gram excels at capturing semantic nuances for diverse vocabulary
- CBOW offers computational efficiency with competitive performance
- FastText's subword information provides robustness to vocabulary variations and OOV terms

### Computational Trade-offs
- **Training time**: All embeddings trained on domain data (~similar duration)
- **Feature extraction**: Document-level averaging is fast
- **Model training**: Logistic Regression trains quickly with all embeddings
- **Best performance**: Varies by embedding characteristics and vocabulary coverage

### Practical Considerations
- **Skip-gram**: Better for rare words, domain-specific vocabulary
- **CBOW**: Faster training, more stable with frequent words
- **FastText**: Handles OOV words, typos, and morphological variations
- **All**: Trained on domain data for better adaptation to e-commerce terminology

## Limitations

1. **Document-level averaging**: May lose important word order and context
2. **Static embeddings**: No contextualized representations (vs. BERT/RoBERTa)
3. **Computational constraints**: Limited hyperparameter search space
4. **Class imbalance**: Some categories may have lower performance
5. **Linear classifier**: Cannot capture complex non-linear relationships

## Future Work

1. Explore weighted averaging schemes for document embeddings
2. Implement attention mechanisms for word importance
3. Investigate ensemble methods combining multiple embeddings
4. Apply SMOTE or other techniques for severe class imbalance
5. Experiment with contextualized embeddings (BERT, RoBERTa)
6. Perform detailed error analysis and feature importance study

## References

### Primary Literature
1. **Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013)**. Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.

2. **Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013)**. Distributed representations of words and phrases and their compositionality. In *Advances in Neural Information Processing Systems* (pp. 3111-3119).

3. **Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017)**. Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics*, 5, 135-146.

4. **Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008)**. LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9, 1871-1874.

### Supporting References
- Arora, S., Liang, Y., & Ma, T. (2017). A simple but tough-to-beat baseline for sentence embeddings. In *ICLR*.
- Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-scale Bayesian logistic regression for text categorization. *Technometrics*, 49(3), 291-304.

## Troubleshooting

### Common Issues

**Issue**: Out of memory during Word2Vec training
```bash
# Solution: Reduce vector_size or use a smaller training corpus
# Increase min_count to reduce vocabulary size





