# Topic Clustering Module

A modular Python library for clustering text data into topics using cosine similarity and K-Means clustering. This module is designed to be easily integrated into the `rdb_app` project.

## Overview

This module provides functionality to:
- **Vectorize** text using TF-IDF (Term Frequency-Inverse Document Frequency)
- **Cluster** text data into topics using K-Means algorithm
- **Analyze** cluster composition and identify representative terms
- **Calculate** cosine distances from documents to cluster centers
- **Process** and clean text data before clustering

## Module Structure

```
clustering/
├── __init__.py              # Package initialization
└── topic_clustering.py      # Main clustering classes and functions
```

## Installation

1. Ensure all required dependencies are installed:

```bash
pip install -r requirements.txt
```

2. Copy the `clustering` folder to your project's source directory (e.g., `src/clustering/`).

## Core Components

### 1. TopicClusterer Class

The main class for topic clustering with the following key methods:

```python
from clustering.topic_clustering import TopicClusterer

# Create a clusterer instance
clusterer = TopicClusterer(
    n_clusters=5,           # Number of clusters
    max_df=0.95,           # Max document frequency
    min_df=1,              # Min document frequency
    random_state=42,       # For reproducibility
)

# Fit to training data
clusterer.fit(texts)

# Predict clusters for new data
clusters = clusterer.predict(new_texts)

# Get top terms for each cluster
top_terms = clusterer.get_top_terms(n_terms=10)

# Calculate distances to cluster centers
distances = clusterer.get_cluster_distances(texts)
```

**Key Parameters:**
- `n_clusters`: Number of topic clusters (default: 5)
- `max_df`: Ignore terms that appear in more than this fraction of documents (default: 0.95)
- `min_df`: Ignore terms that appear in fewer documents than this (default: 1)
- `random_state`: Random seed for reproducibility (default: 42)

**Key Methods:**
- `fit(texts, show_progress=True)`: Fit the model to text data
- `predict(texts)`: Predict cluster assignments for new texts
- `fit_predict(texts, show_progress=True)`: Fit and predict in one step
- `get_top_terms(n_terms=10)`: Get representative terms for each cluster
- `get_cluster_distances(texts)`: Get cosine distances to all cluster centers

### 2. Convenience Function: perform_topic_clustering()

A high-level function for clustering text in a pandas DataFrame:

```python
from clustering.topic_clustering import perform_topic_clustering
import pandas as pd

df = pd.DataFrame({
    'text': ['text 1', 'text 2', ...],
    'other_column': [...]
})

result_df = perform_topic_clustering(
    dataframe=df,
    text_column='text',
    n_clusters=5,
    output_column='Cluster_Topic',
    show_progress=True,
)
```

### 3. Text Cleaning Function

The `clean_text()` function is available for optional text preprocessing:

```python
from clustering.topic_clustering import clean_text

# Clean text by removing URLs, punctuation, etc.
cleaned = clean_text(text, remove_punctuation=True)
```

## Usage Examples

### Basic Usage

```python
from clustering.topic_clustering import TopicClusterer

texts = [
    "Machine learning is fascinating",
    "Deep learning with neural networks",
    "Data science and statistics",
    "Python programming language",
    "Web development with JavaScript",
]

# Create and fit clusterer
clusterer = TopicClusterer(n_clusters=2)
clusters = clusterer.fit_predict(texts)

# Output: [0, 0, 0, 1, 1] (or similar cluster assignments)
print(clusters)

# Get top terms
top_terms = clusterer.get_top_terms(n_terms=5)
# Output: {0: ['machine', 'learning', 'deep', 'neural', 'networks'],
#          1: ['python', 'programming', 'web', 'javascript', 'development']}
```

### DataFrame Integration

```python
from clustering.topic_clustering import perform_topic_clustering

df = pd.read_csv('data.csv')

# Cluster the 'content' column
result = perform_topic_clustering(
    dataframe=df,
    text_column='content',
    n_clusters=5,
    output_column='topic_cluster'
)

# The result DataFrame now has a 'topic_cluster' column
print(result[['content', 'topic_cluster']])
```

### Distance Analysis

```python
clusterer = TopicClusterer(n_clusters=3)
clusterer.fit(texts)

# Get distances from each text to cluster centers
distances = clusterer.get_cluster_distances(texts)

# distances[i, j] is the distance from text i to cluster center j
for i, text in enumerate(texts):
    closest_cluster = distances[i].argmin()
    print(f"Text closer to Cluster {closest_cluster}")
```

## Integration with rdb_app

To integrate this module into the rdb_app project:

1. **Move the clustering folder** to `src/clustering/`:
   ```bash
   cp -r clustering src/
   ```

2. **Update the src/__init__.py** to export clustering functions:
   ```python
   from .clustering import (
       TopicClusterer,
       perform_topic_clustering,
   )
   ```

3. **Use in your application** (e.g., in `app.py` or `transformation/` modules):
   ```python
   from src.clustering import perform_topic_clustering
   
   def process_data(df):
       # ... existing processing ...
       df = perform_topic_clustering(
           dataframe=df,
           text_column='content',
           n_clusters=5,
           output_column='topics'
       )
       return df
   ```

4. **Add to requirements.txt** (if not already present):
   - pandas>=2.0.0
   - scikit-learn>=1.3.0
   - numpy>=1.24.0
   - tqdm>=4.65.0

## Parameters and Configuration

### TF-IDF Vectorizer Parameters

- `max_df` (float): Ignore terms that appear in more than this fraction of documents
  - Range: 0.0 to 1.0
  - Higher value = allow more common terms
  - Default: 0.95 (removes very common terms)

- `min_df` (int): Ignore terms that appear in fewer documents than this
  - Default: 1 (include all terms that appear at least once)
  - Increase to remove very rare terms

- `max_features` (int): Maximum number of features to extract
  - Default: 500 (reduces dimensionality and computation)

### K-Means Parameters

- `n_clusters` (int): Number of clusters to create
  - Adjust based on your dataset and desired granularity
  - No default "best" value; use domain knowledge or elbow method

- `random_state` (int): Seed for reproducibility
  - Default: 42
  - Set to None for random results each time

- `n_init` (int): Number of initializations for K-Means
  - Default: 10
  - Higher values = more reliable but slower

## Performance Considerations

- **Large datasets**: Consider reducing `max_features` or increasing `min_df`
- **Memory usage**: TF-IDF vectorization stores a sparse matrix; large vocabularies may consume significant memory
- **Computation time**: K-Means scales with number of samples and features
- **Quality**: More clusters = more specific topics but less statistical power per cluster

## Output

The module provides:

1. **Cluster Assignments**: Integer labels (0 to n_clusters-1) for each document
2. **Top Terms**: Representative words for each cluster
3. **Distances**: Cosine distances from documents to cluster centers
4. **Progress Information**: Optional progress bars and summary statistics

## Dependencies

- **pandas**: Data manipulation and DataFrames
- **numpy**: Numerical computations
- **scikit-learn**: TF-IDF, K-Means, and cosine similarity
- **tqdm**: Progress bars

## Error Handling

The module includes validation for:

- Empty or invalid input data
- Missing columns in DataFrames
- Unfitted models (attempting to predict without fitting first)
- Empty text data

## License

This module follows the same license as the rdb_app project.

## Notes

- The module uses English stopwords by default in TF-IDF vectorization
- All text is automatically converted to lowercase during processing
- The cosine similarity metric is used for distance calculations
- Results are reproducible when `random_state` is set
