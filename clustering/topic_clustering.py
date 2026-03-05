"""Topic Clustering using Cosine Similarity and K-Means.

This module provides functionality for clustering text data into topics
using TF-IDF vectorization, cosine similarity, and k-means clustering.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
import re
import string

warnings.filterwarnings('ignore')


def clean_text(text: str, remove_punctuation: bool = True, remove_numbers: bool = False) -> str:
    """
    Clean text by removing special characters, extra whitespace, etc.
    
    Args:
        text (str): Input text to clean
        remove_punctuation (bool): Whether to remove punctuation (default: True)
        remove_numbers (bool): Whether to remove numbers (default: False)
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers if specified
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Remove punctuation if specified
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class TopicClusterer:
    """
    A class for clustering text data into topics using cosine similarity and k-means.
    
    Attributes:
        n_clusters (int): Number of topic clusters to create
        max_df (float): Maximum document frequency for TF-IDF vectorizer
        min_df (int): Minimum document frequency for TF-IDF vectorizer
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer
        kmeans (KMeans): Fitted K-Means clustering model
        tfidf_matrix (sparse matrix): TF-IDF vectors for the text data
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        max_df: float = 0.95,
        min_df: int = 1,
        random_state: int = 42,
        n_init: int = 10,
    ):
        """
        Initialize the TopicClusterer.
        
        Args:
            n_clusters (int): Number of clusters to create (default: 5)
            max_df (float): Maximum document frequency (default: 0.95)
            min_df (int): Minimum document frequency (default: 1)
            random_state (int): Random state for reproducibility (default: 42)
            n_init (int): Number of initializations for KMeans (default: 10)
        """
        self.n_clusters = n_clusters
        self.max_df = max_df
        self.min_df = min_df
        self.random_state = random_state
        self.n_init = n_init
        
        self.vectorizer = None
        self.kmeans = None
        self.tfidf_matrix = None
        self.feature_names = None
        
    def _preprocess_text(self, texts: List[str]) -> List[str]:
        """
        Preprocess text data by converting to lowercase and handling empty values.
        
        Args:
            texts (List[str]): List of text strings to preprocess
            
        Returns:
            List[str]: Preprocessed text data
        """
        processed = []
        for text in texts:
            if pd.isna(text):
                processed.append("")
            else:
                processed.append(str(text).lower().strip())
        return processed
    
    def fit(self, texts: List[str], show_progress: bool = True) -> 'TopicClusterer':
        """
        Fit the clustering model to the text data.
        
        Args:
            texts (List[str]): List of text documents to cluster
            show_progress (bool): Whether to show progress bar (default: True)
            
        Returns:
            TopicClusterer: Self for method chaining
            
        Raises:
            ValueError: If texts list is empty or contains only empty strings
        """
        if not texts or len(texts) == 0:
            raise ValueError("texts list cannot be empty")
        
        # Preprocess texts
        processed_texts = self._preprocess_text(texts)
        
        # Filter out empty texts and track indices
        non_empty_texts = [t for t in processed_texts if t.strip()]
        if not non_empty_texts:
            raise ValueError("All texts are empty after preprocessing")
        
        # Create and fit TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words='english',
            lowercase=True,
            max_features=500,
        )
        
        if show_progress:
            print("Vectorizing text using TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Apply K-Means clustering
        if show_progress:
            print(f"Clustering into {self.n_clusters} topics...")
        self.kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(non_empty_texts)),
            random_state=self.random_state,
            n_init=self.n_init,
        )
        self.kmeans.fit(self.tfidf_matrix)
        
        if show_progress:
            print("Clustering complete!")
        
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict cluster assignments for new text data.
        
        Args:
            texts (List[str]): List of text documents to predict
            
        Returns:
            np.ndarray: Cluster assignments for each document
            
        Raises:
            ValueError: If the model has not been fitted yet
        """
        if self.vectorizer is None or self.kmeans is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        processed_texts = self._preprocess_text(texts)
        tfidf_new = self.vectorizer.transform(processed_texts)
        return self.kmeans.predict(tfidf_new)
    
    def fit_predict(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Fit the model and predict cluster assignments in one step.
        
        Args:
            texts (List[str]): List of text documents to cluster
            show_progress (bool): Whether to show progress bar (default: True)
            
        Returns:
            np.ndarray: Cluster assignments for each document
        """
        self.fit(texts, show_progress=show_progress)
        return self.kmeans.labels_
    
    def get_top_terms(self, n_terms: int = 10) -> Dict[int, List[str]]:
        """
        Get the top N terms for each cluster.
        
        Args:
            n_terms (int): Number of top terms to retrieve per cluster (default: 10)
            
        Returns:
            Dict[int, List[str]]: Dictionary mapping cluster ID to list of top terms
            
        Raises:
            ValueError: If the model has not been fitted yet
        """
        if self.kmeans is None or self.feature_names is None:
            raise ValueError("Model must be fitted first. Call fit() first.")
        
        cluster_centers = self.kmeans.cluster_centers_
        top_terms = {}
        
        for cluster_id in range(self.n_clusters):
            center = cluster_centers[cluster_id]
            top_indices = center.argsort()[-n_terms:][::-1]
            top_terms[cluster_id] = [
                self.feature_names[idx] for idx in top_indices
            ]
        
        return top_terms
    
    def get_cluster_distances(self, texts: List[str]) -> np.ndarray:
        """
        Calculate cosine distances from each text to all cluster centers.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            np.ndarray: Distance matrix (n_samples, n_clusters)
            
        Raises:
            ValueError: If the model has not been fitted yet
        """
        if self.vectorizer is None or self.kmeans is None:
            raise ValueError("Model must be fitted first. Call fit() first.")
        
        processed_texts = self._preprocess_text(texts)
        tfidf_vectors = self.vectorizer.transform(processed_texts)
        
        # Calculate cosine similarity to cluster centers
        similarities = cosine_similarity(tfidf_vectors, self.kmeans.cluster_centers_)
        
        # Convert to distances (1 - similarity)
        distances = 1 - similarities
        return distances
    
    def get_2d_coordinates(self, texts: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D coordinates using PCA for visualization.
        
        Args:
            texts (Optional[List[str]]): Text data for 2D projection. If None, uses training data.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 2D coordinates and cluster labels
            
        Raises:
            ValueError: If the model has not been fitted yet
        """
        if self.tfidf_matrix is None or self.kmeans is None:
            raise ValueError("Model must be fitted first. Call fit() first.")
        
        # Get TF-IDF vectors
        if texts is not None:
            processed_texts = self._preprocess_text(texts)
            tfidf_vectors = self.vectorizer.transform(processed_texts)
            clusters = self.kmeans.predict(tfidf_vectors)
        else:
            tfidf_vectors = self.tfidf_matrix
            clusters = self.kmeans.labels_
        
        # Apply PCA to reduce to 2D
        pca = PCA(n_components=2, random_state=self.random_state)
        coordinates_2d = pca.fit_transform(tfidf_vectors.toarray())
        
        return coordinates_2d, clusters
    
    def get_top_terms_matrix(self, n_terms: int = 10) -> pd.DataFrame:
        """
        Get top terms for each cluster as a DataFrame for heatmap visualization.
        
        Args:
            n_terms (int): Number of top terms per cluster (default: 10)
            
        Returns:
            pd.DataFrame: DataFrame with clusters as rows and terms as columns, values are term importance
            
        Raises:
            ValueError: If the model has not been fitted yet
        """
        if self.kmeans is None or self.feature_names is None:
            raise ValueError("Model must be fitted first. Call fit() first.")
        
        cluster_centers = self.kmeans.cluster_centers_
        
        # Collect all top terms and their importance scores
        term_scores = {}
        
        for cluster_id in range(self.n_clusters):
            center = cluster_centers[cluster_id]
            top_indices = center.argsort()[-n_terms:][::-1]
            
            for rank, idx in enumerate(top_indices):
                term = self.feature_names[idx]
                score = center[idx]
                
                if term not in term_scores:
                    term_scores[term] = {}
                
                term_scores[term][cluster_id] = score
        
        # Create DataFrame
        df_terms = pd.DataFrame(term_scores).fillna(0).T
        
        # Reorder columns to match cluster order
        columns_order = [col for col in range(self.n_clusters) if col in df_terms.columns]
        df_terms = df_terms[[col for col in columns_order if col in df_terms.columns]]
        
        return df_terms


def perform_topic_clustering(
    dataframe: pd.DataFrame,
    text_column: str,
    n_clusters: int = 5,
    max_df: float = 0.95,
    min_df: int = 1,
    output_column: str = 'Cluster_Topic',
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Perform topic clustering on a dataframe column.
    
    This function clusters text data in a specified column of a DataFrame
    using cosine similarity-based K-Means clustering.
    
    Args:
        dataframe (pd.DataFrame): Input dataframe with text data
        text_column (str): Name of the column containing text to cluster
        n_clusters (int): Number of clusters to create (default: 5)
        max_df (float): Maximum document frequency (default: 0.95)
        min_df (int): Minimum document frequency (default: 1)
        output_column (str): Name of output column for cluster assignments (default: 'Cluster_Topic')
        show_progress (bool): Whether to show progress bar (default: True)
        
    Returns:
        pd.DataFrame: DataFrame with new column containing cluster assignments
        
    Raises:
        ValueError: If text_column does not exist or dataframe is empty
        KeyError: If text_column name is invalid
    """
    if dataframe is None or dataframe.empty:
        raise ValueError("DataFrame cannot be empty")
    
    if text_column not in dataframe.columns:
        raise KeyError(f"Column '{text_column}' not found in dataframe. Available columns: {list(dataframe.columns)}")
    
    # Extract texts
    texts = dataframe[text_column].tolist()
    
    # Create and fit clusterer
    clusterer = TopicClusterer(
        n_clusters=n_clusters,
        max_df=max_df,
        min_df=min_df,
    )
    
    if show_progress:
        print(f"Performing topic clustering on '{text_column}' column...")
    
    clusters = clusterer.fit_predict(texts, show_progress=show_progress)
    
    # Add cluster assignments to dataframe
    result_df = dataframe.copy()
    result_df[output_column] = clusters
    
    if show_progress:
        print(f"Clustering results added to '{output_column}' column")
        print(f"\nCluster distribution:")
        print(result_df[output_column].value_counts().sort_index())
        
        print(f"\nTop terms per cluster:")
        top_terms = clusterer.get_top_terms(n_terms=5)
        for cluster_id, terms in top_terms.items():
            print(f"  Cluster {cluster_id}: {', '.join(terms)}")
    
    return result_df
