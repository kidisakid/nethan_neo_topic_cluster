"""Topic Clustering Module.

This module provides topic clustering functionality using cosine similarity
and k-means clustering algorithms.
"""

from .topic_clustering import (
    TopicClusterer,
    perform_topic_clustering,
)

__all__ = [
    'TopicClusterer',
    'perform_topic_clustering',
]
