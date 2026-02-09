# E-commerce Text Classification using LSTM (Dense Network)

**Author**: Carine UMUGABEKAZI  
**Date**: February 2026  
**Best Performance**: 97.62% Test Accuracy

## 1. Problem Definition

This project implements a neural network classifier for multi-class e-commerce product categorization using TF-IDF features. The model achieves state-of-the-art performance (97.62% accuracy) among all team implementations, demonstrating that feature quality often matters more than architectural complexity.

**Research Question**: Can a dense neural network with TF-IDF features outperform traditional recurrent architectures (RNN, GRU) that use word embeddings?

**Answer**: Yes. Our "LSTM" implementation (actually a dense feedforward network) achieves 97.62% accuracy, outperforming both RNN (97.32%) and GRU (97.54%), confirming that TF-IDF's discriminative power is superior for document-level classification tasks.

## 2. Dataset

**Dataset**: E-commerce Product Classification  
**Source**: `data/preprocessed_data/`  
**Total Samples**: 50,425 products  
**Categories**: 4 classes
- Books
- Clothing & Accessories
- Electronics
- Household

**Split Ratio**: 70% train / 15% validation / 15% test

**Dataset Characteristics**:
- Balanced class distribution
- Product descriptions with varying lengths
- Domain-specific vocabulary (brands, technical specs, materials)

## 3. Preprocessing Pipeline

Robust text preprocessing ensures clean, normalized input for TF-IDF vectorization.

**Steps**:
1. **Contraction Expansion**: "don't" → "do not"
2. **Lowercase Conversion**: Normalize case
3. **HTML/Special Character Removal**: Clean markup
4. **Punctuation Removal**: Retain only alphanumeric content
5. **Tokenization**: NLTK word tokenizer
6. **Stopword Removal**: Remove common English stopwords
7. **Lemmatization**: WordNetLemmatizer for root forms

**NLTK Resources Required**:
```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 4. Feature Extraction: TF-IDF

**Why TF-IDF Over Word Embeddings?**

TF-IDF was chosen over Word2Vec/GloVe/FastText embeddings because:
- **Discriminative Power**: Captures term importance for classification
- **Document-Level Representation**: Natural fit for product descriptions
- **Efficiency**: No embedding training or averaging required
- **Interpretability**: Direct feature-to-class relationships
- **Performance**: Achieves higher accuracy than embedding-based approaches

**TF-IDF Configuration**:
```python
TfidfVectorizer(
    max_features=5000,      # Top 5000 most important terms
    ngram_range=(1, 2),     # Unigrams and bigrams
    min_df=2,               # Ignore very rare terms
    sublinear_tf=True       # Logarithmic term frequency scaling
)
```

**Output**: 5,000-dimensional sparse vectors converted to dense arrays

## 5. Model Architecture

**Important Note**: Despite being labeled "LSTM" in the codebase, this implementation uses a **dense feedforward neural network** without recurrent layers. The architecture is optimized for pre-aggregated TF-IDF features rather than sequential processing.

**Architecture**:
```
Input Layer (5000 features)
    ↓
Dense Layer (256 units, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Output Layer (4 units, Softmax activation)
```

**Key Components**:
- **Dense(256)**: Hidden layer for non-linear feature combinations
- **Dropout(0.3)**: Regularization to prevent overfitting
- **Softmax**: Multi-class probability distribution

**Training Configuration**:
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 6 (early stopping with patience=3)
- **Callbacks**: EarlyStopping (monitor='val_loss')

**Why Dense Network Works Better**:
1. TF-IDF produces document-level aggregated features
2. No sequential information to capture
3. Dense layers efficiently learn feature combinations
4. Faster training and inference than recurrent architectures

## 6. Training Process

**Training Details**:
- **Training Samples**: 35,297
- **Validation Samples**: 7,564
- **Test Samples**: 7,564
- **Training Time**: 2.7 minutes (6 epochs)
- **Hardware**: CPU execution
- **Convergence**: Early stopping at epoch 6

**Training Dynamics**:
- Rapid convergence in first 3 epochs
- Validation accuracy stabilized at 97.5%
- No significant overfitting observed
- Loss curves showed consistent improvement

## 7. Results

### Overall Performance

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 97.62%  |
| Precision | 97.62%  |
| Recall    | 97.62%  |
| F1-Score  | 97.62%  |

### Per-Category Performance

| Category              | Precision | Recall | F1-Score | Support |
|-----------------------|-----------|--------|----------|---------|
| Books                 | 96.85%    | 97.86% | 97.35%   | 1,868   |
| Clothing & Accessories| 98.86%    | 97.33% | 98.09%   | 1,873   |
| Electronics           | 97.88%    | 96.99% | 97.43%   | 1,895   |
| Household             | 96.83%    | 98.51% | 97.66%   | 1,928   |

**Key Observations**:
- All categories exceed 97% F1-score
- No weak category (minimal variance: 97.35% - 98.09%)
- Balanced performance across classes
- Clothing & Accessories achieves highest precision (98.86%)

### Confusion Matrix Insights

- Books: 1,828/1,868 correct (97.86%)
- Clothing: 1,823/1,873 correct (97.33%)
- Electronics: 1,838/1,895 correct (96.99%)
- Household: 1,899/1,928 correct (98.51%)

**Misclassification Patterns**:
- Minimal confusion between categories
- Most errors involve Books ↔ Electronics (shared technical vocabulary)
- Very few Clothing misclassifications (distinctive terminology)

## 8. Comparative Analysis

### Comparison with Other Models

| Model                    | Accuracy | Training Time | Epochs |
|--------------------------|----------|---------------|--------|
| **LSTM (Dense + TF-IDF)**| **97.62%** | **2.7 min**   | **6**  |
| GRU (Word Embeddings)    | 97.54%   | 18.2 min      | 19     |
| RNN (Word Embeddings)    | 97.32%   | 22.8 min      | 24     |
| Logistic Regression      | 94.10%   | 3.5 min       | N/A    |

**Key Insights**:
1. **Best Accuracy**: LSTM (dense) achieves highest accuracy
2. **85% Faster**: 2.7 min vs 18-22 min for RNN/GRU
3. **Faster Convergence**: 6 epochs vs 19-24 epochs
4. **Feature Quality > Architecture Complexity**: TF-IDF outperforms word embeddings

### Why LSTM (Dense) Outperforms RNN/GRU

**Advantages**:
- **TF-IDF Features**: Capture discriminative term importance
- **Document-Level Representation**: Natural fit for classification
- **No Sequential Bias**: Dense layers learn arbitrary feature combinations
- **Efficient Training**: Fewer parameters, faster convergence

**RNN/GRU Limitations**:
- **Embedding Averaging**: Loses sequential information
- **Over-parameterization**: Recurrent layers unnecessary for aggregated features
- **Training Complexity**: More epochs needed for convergence

## 9. Key Findings

**Main Conclusions**:
1. **Feature Quality Matters**: TF-IDF's discriminative power exceeds semantic embeddings
2. **Architecture Fit**: Dense networks optimal for document-level features
3. **Efficiency Wins**: Simpler models train faster without sacrificing accuracy
4. **Document vs. Sequential**: Product descriptions benefit from bag-of-words approaches

**Practical Implications**:
- For document classification, prioritize feature engineering over complex architectures
- TF-IDF remains highly competitive despite deep learning advances
- Dense networks can outperform recurrent architectures when input is pre-aggregated
- Training efficiency matters for production deployment

## 10. Limitations

1. **Architecture Naming**: Model labeled "LSTM" but uses dense layers
2. **Feature Sparsity**: 5,000 features may not capture all semantic nuances
3. **No Sequential Modeling**: Cannot capture word order or syntax
4. **Dataset Specificity**: Results apply to e-commerce domain
5. **Single Run**: No multiple seeds for statistical significance

## 11. Future Directions

**Potential Improvements**:
1. **Hybrid Features**: Combine TF-IDF with embeddings
2. **Attention Mechanisms**: Add attention over TF-IDF features
3. **Ensemble Methods**: Combine dense model with RNN/GRU
4. **Domain Adaptation**: Test on other e-commerce datasets
5. **Hyperparameter Optimization**: Grid search for optimal architecture

**Alternative Approaches**:
- Transformer-based models (BERT) for contextualized representations
- Graph neural networks for category hierarchies
- Multi-task learning with product attributes

## 12. Files and Outputs

**Notebook**: `models/lstm/lstm_ml_classification.ipynb`  
**Results Directory**: `results/lstm/`

**Generated Files**:
- `confusion_matrix.png` - Classification confusion matrix
- `performance_comparison.png` - Model comparison charts
- `text_length_analysis.png` - Dataset statistics
- `model_comparison.csv` - Quantitative metrics
- `per_class_metrics.csv` - Category-level performance

## 13. Environment Setup

**Dependencies**:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow nltk
```

**TensorFlow Configuration**:
- CPU execution mode
- TensorFlow 2.x
- Keras Sequential API

**NLTK Setup**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 14. Execution Instructions

1. **Activate Virtual Environment**:
```bash
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. **Navigate to Notebook**:
```bash
cd models/lstm
jupyter notebook lstm_ml_classification.ipynb
```

3. **Run All Cells**: Execute cells sequentially (22 cells total)

4. **Check Results**: View outputs in `results/lstm/`

**Estimated Runtime**: ~3 minutes on CPU

## 15. Author Contribution

**Author**: Carine UMUGABEKAZI  
**Contribution**: Complete LSTM (Dense Network) implementation, TF-IDF feature engineering, model training, evaluation, and documentation

**Key Achievements**:
- Achieved highest accuracy (97.62%) among all team models
- Demonstrated feature quality > architectural complexity
- Fastest training time with early convergence
- Comprehensive evaluation and comparative analysis

**Acknowledgments**: 
- Team collaboration on dataset preprocessing
- Comparative analysis with RNN (Raissa) and GRU (Carmel) implementations
- Traditional ML baseline (Raissa - Logistic Regression)
