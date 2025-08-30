from __future__ import annotations
import os, json
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_json(path: str, data: Dict) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def topic_diversity(topics_words: List[List[str]]) -> float:
    """
    Diversity = unique words / total words across all topics.
    """
    all_words = [w for topic in topics_words for w in topic]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)

def plot_coherence_diversity(metrics: Dict[str, Dict[str, float]], out_path: str) -> None:
    """
    Creates a simple grouped bar chart for coherence and diversity.
    """
    ensure_dir(out_path)
    models = list(metrics.keys())
    coherence = [metrics[m]["coherence_c_v"] for m in models]
    diversity = [metrics[m]["topic_diversity"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, coherence, width, label="Coherence (c_v)")
    plt.bar(x + width/2, diversity, width, label="Topic Diversity")
    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.title("Coherence vs Diversity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def top_words_from_matrix(components, feature_names: List[str], top_n: int) -> List[List[str]]:
    """
    Given LDA components_ and vocab, return top-N words per topic.
    """
    topics = []
    for comp in components:
        indices = comp.argsort()[::-1][:top_n]
        topics.append([feature_names[i] for i in indices])
    return topics
