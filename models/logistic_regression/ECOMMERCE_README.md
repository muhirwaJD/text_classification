# Traditional Machine Learning for E-commerce Text Classification

## Overview

This module implements a comprehensive comparative analysis of traditional machine learning approaches for e-commerce product text classification. The implementation evaluates three different word embedding techniques across multiple classifier architectures, providing insights into the effectiveness of various feature representation strategies.

## Author Information
- **Module**: Traditional Machine Learning (Logistic Regression, SVM, Random Forest)
- **Embeddings**: TF-IDF, GloVe, FastText
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

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Type**: Sparse statistical representation
- **Approach**: Combines term frequency with inverse document frequency weighting
- **Configuration**:
  - Maximum features: 5,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - Sublinear TF scaling applied
  - L2 normalization
  - Min DF: 2, Max DF: 0.8
- **References**: Ramos (2003), Salton & Buckley (1988)

#### GloVe (Global Vectors for Word Representation)
- **Type**: Dense pre-trained embeddings
- **Approach**: Co-occurrence-based word vectors capturing semantic relationships
- **Configuration**:
  - Embedding dimension: 100
  - Pre-trained on large corpus
  - Document-level averaging for classification
- **Source**: Stanford NLP (glove.6B.100d.txt)
- **References**: Pennington et al. (2014)

#### FastText
- **Type**: Character n-gram based embeddings
- **Approach**: Subword information for handling OOV words and morphology
- **Configuration**:
  - Vector size: 100
  - Window: 5
  - Skip-gram architecture
  - Character n-grams: 3-6
  - Epochs: 10
  - Trained on domain data
- **References**: Bojanowski et al. (2017)

### 3. Classification Models

#### Logistic Regression
- **Type**: Linear discriminative classifier
- **Hyperparameters**: C=[0.1, 1.0, 10.0, 100.0], penalty=['l2'], solver=['lbfgs', 'liblinear']
- **Optimization**: L-BFGS and LIBLINEAR solvers
- **Advantages**: Fast training, interpretable, effective with high-dimensional sparse data
- **References**: Fan et al. (2008)

#### Support Vector Machine (SVM)
- **Type**: Kernel-based maximum margin classifier
- **Hyperparameters**: C=[0.1, 1.0, 10.0], kernel=['linear', 'rbf'], gamma=['scale', 'auto']
- **Kernels**: Linear and RBF for capturing non-linear relationships
- **Advantages**: Effective in high-dimensional spaces, memory efficient
- **References**: Joachims (1998)

#### Random Forest
- **Type**: Ensemble of decision trees
- **Hyperparameters**: n_estimators=[100, 200, 300], max_depth=[None, 20, 30], max_features=['sqrt', 'log2']
- **Advantages**: Handles non-linearity naturally, robust to overfitting
- **References**: Breiman (2001)

### 4. Experimental Design
- **Total experiments**: 9 model-embedding combinations
- **Hyperparameter tuning**: GridSearchCV with 3-fold stratified cross-validation
- **Evaluation metric**: Weighted F1-score (primary), Accuracy, Precision, Recall
- **Class imbalance handling**: Balanced class weights

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

# Download GloVe embeddings (optional but recommended)
# Visit: https://nlp.stanford.edu/projects/glove/
# Download glove.6B.zip, extract glove.6B.100d.txt
# Create embeddings/ folder and place file there
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
Different embeddings show varying effectiveness with different architectures:
- TF-IDF typically performs best with linear models (Logistic Regression, Linear SVM)
- Dense embeddings (GloVe, FastText) often work better with non-linear classifiers
- FastText's subword information helps with domain-specific vocabulary

### Computational Trade-offs
- **Fastest**: TF-IDF vectorization + Logistic Regression
- **Most accurate**: Varies by dataset characteristics
- **Best generalization**: Dense embeddings with ensemble methods

### Practical Considerations
- TF-IDF: No additional training needed, interpretable features
- GloVe: Requires pre-trained embeddings, good for general text
- FastText: Requires training time, best for domain-specific applications

## Limitations

1. **Document-level averaging**: May lose important word order and context
2. **Pre-trained embeddings**: GloVe may miss domain-specific terminology
3. **Computational constraints**: Limited hyperparameter search space
4. **Class imbalance**: Some categories may have lower performance
5. **Static embeddings**: No contextualized representations (vs. BERT/RoBERTa)

## Future Work

1. Explore weighted averaging schemes for document embeddings
2. Implement attention mechanisms for word importance
3. Investigate ensemble methods combining multiple embeddings
4. Apply SMOTE or other techniques for severe class imbalance
5. Experiment with contextualized embeddings (BERT, RoBERTa)
6. Perform detailed error analysis and feature importance study

## References

### Primary Literature
1. **Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017)**. Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics*, 5, 135-146.

2. **Breiman, L. (2001)**. Random forests. *Machine Learning*, 45(1), 5-32.

3. **Fan, R. E., Chang, K. W., Hsieh, C. J., Wang, X. R., & Lin, C. J. (2008)**. LIBLINEAR: A library for large linear classification. *Journal of Machine Learning Research*, 9, 1871-1874.

4. **Joachims, T. (1998)**. Text categorization with support vector machines: Learning with many relevant features. In *European Conference on Machine Learning* (pp. 137-142).

5. **Pennington, J., Socher, R., & Manning, C. D. (2014)**. GloVe: Global vectors for word representation. In *Proceedings of EMNLP* (pp. 1532-1543).

6. **Ramos, J. (2003)**. Using TF-IDF to determine word relevance in document queries. In *Proceedings of the First Instructional Conference on Machine Learning* (Vol. 242, pp. 133-142).

7. **Salton, G., & Buckley, C. (1988)**. Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.

### Supporting References
- Hsu, C. W., Chang, C. C., & Lin, C. J. (2003). A practical guide to support vector classification.
- Arora, S., Liang, Y., & Ma, T. (2017). A simple but tough-to-beat baseline for sentence embeddings. In *ICLR*.
- Genkin, A., Lewis, D. D., & Madigan, D. (2007). Large-scale Bayesian logistic regression for text categorization. *Technometrics*, 49(3), 291-304.

## Troubleshooting

### Common Issues

**Issue**: GloVe embeddings not found
```bash
# Solution: Download from Stanford NLP
# https://nlp.stanford.edu/projects/glove/
# Place glove.6B.100d.txt in embeddings/ folder
```

**Issue**: Out of memory during training
```bash
# Solution: Reduce max_features in TF-IDF vectorizer
# Use incremental learning for large datasets
```

**Issue**: Slow GridSearchCV
```bash
# Solution: Reduce parameter grid or use RandomizedSearchCV
# Decrease cv folds to 3 (already implemented)
```

## Contributing to Report

### Section Contributions
For the group report, this module provides:
1. **Methodology section**: Traditional ML approach with embeddings
2. **Results tables**: Complete performance metrics (Table 1, Table 2)
3. **Visualizations**: 6+ publication-quality figures
4. **Discussion points**: Embedding-model synergy analysis

### Integration with Team Results
Compare with:
- LSTM results (from lstm/ folder)
- RNN results (from rnn/ folder)
- Other team member implementations

---

**Last Updated**: February 6, 2026  
**Status**: Complete - Ready for Execution and Report Integration
