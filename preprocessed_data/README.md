# Preprocessed BBC News Dataset

This folder contains preprocessed and split data ready for feature extraction and modeling.

## Dataset Information
- **Source**: BBC News Dataset
- **Total Samples**: 2,225 documents
- **Categories**: 5 (business, entertainment, politics, sport, tech)
- **Preprocessing Applied**:
  - Lowercasing
  - Special characters and digits removed
  - Tokenization
  - Stopwords removed
  - Lemmatization applied
  - Short words (< 3 characters) removed

## Files

### train.csv
- **Samples**: 1,335 (60%)
- **Purpose**: Use for model training
- **Columns**: 
  - `text`: Preprocessed text
  - `category`: Class label

### validation.csv
- **Samples**: 445 (20%)
- **Purpose**: Use for hyperparameter tuning and model selection
- **Columns**: Same as train.csv

### test.csv
- **Samples**: 445 (20%)
- **Purpose**: Use for final model evaluation (DO NOT use for training!)
- **Columns**: Same as train.csv

### full_preprocessed.csv
- **Samples**: 2,225 (100%)
- **Purpose**: Complete dataset for reference or alternative splitting
- **Columns**: Same as above

## Usage Example

```python
import pandas as pd

# Load the data
train_df = pd.read_csv('preprocessed_data/train.csv')
val_df = pd.read_csv('preprocessed_data/validation.csv')
test_df = pd.read_csv('preprocessed_data/test.csv')

# Extract features
X_train = train_df['text']
y_train = train_df['category']

# Your feature extraction code here...
# (e.g., TF-IDF, Word2Vec, BERT embeddings, etc.)
```

## Important Notes

1. **Data Split**: The train/val/test split is stratified to maintain class balance
2. **Random Seed**: Split was done with `random_state=42` for reproducibility
3. **Preprocessing**: Text is already cleaned - start directly with feature extraction
4. **Test Set**: Keep test set separate until final evaluation

## Class Distribution

| Split      | Business | Entertainment | Politics | Sport | Tech |
|------------|----------|---------------|----------|-------|------|
| Train      | 306      | 232           | 250      | 307   | 240  |
| Validation | 102      | 77            | 83       | 102   | 81   |
| Test       | 102      | 77            | 84       | 102   | 80   |

## Contact
For questions about preprocessing, contact: Raissa (Traditional ML team)

---
Generated: February 3, 2026
