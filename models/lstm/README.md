```markdown
# BBC News Text Classification using LSTM

## 1. Problem Definition

The goal of this project is to perform multi-class text classification on news articles and study how different text embedding techniques interact with a deep learning model.

Specifically, the project investigates whether an LSTM-based classifier benefits more from:
- frequency-based representations (TF-IDF), or
- semantic word embeddings (GloVe and FastText).

The central research question is:
How does the choice of text representation affect classification performance when using the same LSTM architecture?


## 2. Dataset and Justification

The BBC News dataset was used for this study.  
It contains news articles grouped into five topical categories:
business, entertainment, politics, sport, and technology.

This dataset is appropriate because:
- it is widely used for text classification benchmarking,
- it has clear topic boundaries,
- it allows comparison of keyword-based vs embedding-based representations.

The dataset is stored as folders, where each folder name corresponds to a class label.

## 3. Dataset Structure and Loading

The dataset is loaded manually from directory paths rather than using built-in loaders.  
This allows full control over preprocessing and labeling.

Expected dataset path in the code:
```

../data/bbc

```

Folder structure:
```

bbc/
├── business/
├── entertainment/
├── politics/
├── sport/
└── tech/

```

Each file is a single news article.

## 4. Preprocessing Pipeline

A shared preprocessing pipeline is applied across all experiments to ensure fairness.

Steps:
- lowercase conversion
- punctuation and number removal
- tokenization
- stopword removal
- lemmatization

NLTK resources are required for these steps, including:
punkt, stopwords, wordnet, and averaged_perceptron_tagger.

## 5. Embedding Strategies

Three text representation methods are compared.

### 5.1 TF-IDF
TF-IDF is used as a strong baseline.  
It captures word importance across documents but ignores word order.

### 5.2 GloVe
GloVe embeddings provide pretrained semantic representations.  
Document vectors are created by averaging word embeddings.

### 5.3 FastText
FastText embeddings incorporate subword information, which helps with rare or unseen words.  
The same averaging strategy is applied for consistency.

The GloVe file is expected at:
```

../glove/glove.6B.100d.txt

```

## 6. Model Architecture

An LSTM-based neural network is used for all experiments.

The same architecture and training setup are maintained across embeddings to isolate the effect of the text representation.

Although LSTMs are designed for sequential data, they are used here to examine whether sequence modeling provides benefits beyond simple frequency-based features.

## 7. Experimental Design

- One model (LSTM) is used
- Three embedding techniques are evaluated
- Same train-test split and preprocessing pipeline
- Accuracy is used as the primary evaluation metric

This design allows direct comparison between embeddings.


## 8. Results and Comparison

Observed results from the experiments:
- TF-IDF achieved the highest accuracy (~96%)
- GloVe performed slightly lower
- FastText had the lowest accuracy

Despite being simpler, TF-IDF performed best.  
This suggests that BBC News classification is strongly keyword-driven and does not heavily rely on word order or contextual semantics.


## 9. Discussion and Limitations

The results indicate that LSTM does not significantly improve performance when embeddings are averaged and sequence information is weak.

Limitations:
- embeddings are averaged rather than fed as sequences,
- limited hyperparameter tuning,
- evaluation focuses mainly on accuracy.

Future improvements could include:
- sequence-based embeddings,
- attention mechanisms,
- transformer-based models.


## 10. Environment Setup and Execution

### Required Libraries
```

pip install numpy pandas matplotlib scikit-learn tensorflow nltk gensim fasttext

```

### NLTK Downloads
```

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

```

### Execution
- Ensure dataset and GloVe paths are correct
- Run the notebook from top to bottom


## 11. Author Contribution

The LSTM model, embedding experiments, analysis, and documentation were implemented independently as part of the coursework requirements.
```
