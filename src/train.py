from __future__ import annotations
import argparse, os, time, yaml
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from data import load_20newsgroups_docs
from utils import ensure_dir, save_json, plot_coherence_diversity, topic_diversity, top_words_from_matrix

# LDA (classic)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA


# BERTopic (contextual) + KMeans (no HDBSCAN/UMAP)
from sklearn.cluster import KMeans
from bertopic import BERTopic

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def coherence_npmi(topics_words: List[List[str]], tokens_list: List[List[str]]) -> float:
    """
    Lightweight NPMI coherence at document level.
    Returns the average NPMI across all topic word-pairs.
    """
    if len(tokens_list) < 5 or len(topics_words) == 0:
        return 0.0

    N = len(tokens_list)
    vocab = sorted({w for tw in topics_words for w in tw})
    if not vocab:
        return 0.0

    idx = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(len(vocab), dtype=int)
    co = np.zeros((len(vocab), len(vocab)), dtype=int)

    # Document frequencies & co-occurrences
    for tokens in tokens_list:
        present = set(t for t in tokens if t in idx)
        for w in present:
            df[idx[w]] += 1
        present_list = list(present)
        for i in range(len(present_list)):
            for j in range(i + 1, len(present_list)):
                a, b = idx[present_list[i]], idx[present_list[j]]
                co[a, b] += 1
                co[b, a] += 1

    P = df / N
    eps = 1e-12
    scores = []

    for words in topics_words:
        pairs = [(words[i], words[j]) for i in range(len(words)) for j in range(i + 1, len(words))]
        if not pairs:
            continue

        pair_scores = []
        for w1, w2 in pairs:
            i, j = idx.get(w1), idx.get(w2)
            if i is None or j is None:
                continue
            p12 = co[i, j] / N
            if p12 <= 0:
                continue
            p1, p2 = P[i], P[j]
            pmi = np.log(p12 / (p1 * p2 + eps) + eps)
            npmi = pmi / (-np.log(p12 + eps))
            pair_scores.append(npmi)

        if pair_scores:
            scores.append(float(np.mean(pair_scores)))

    return float(np.mean(scores)) if scores else 0.0


def run_lda(docs: List[str], tokens_list: List[List[str]], cfg: Dict, results_dir: str, random_state: int) -> Dict:
    top_n = cfg["general"]["top_n_words"]
    lda_cfg = cfg["lda"]

    vectorizer = CountVectorizer(
        max_df=lda_cfg["max_df"],
        min_df=lda_cfg["min_df"],
        max_features=lda_cfg["max_features"],
        stop_words="english",
    )
    X = vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=lda_cfg["n_topics"],
        learning_method=lda_cfg["learning_method"],
        learning_decay=lda_cfg["learning_decay"],
        random_state=random_state,
    )

    t0 = time.time()
    lda.fit(X)
    train_seconds = time.time() - t0

    feature_names = vectorizer.get_feature_names_out()
    topics_words = top_words_from_matrix(lda.components_, feature_names, top_n=top_n)

    coherence = coherence_npmi(topics_words, tokens_list)
    diversity = topic_diversity(topics_words)

    # Save topic table
    df = pd.DataFrame(
        {
            "topic_id": list(range(len(topics_words))),
            "top_words": [", ".join(ws) for ws in topics_words],
        }
    )
    out_csv = os.path.join(results_dir, "topics", "lda_topics.csv")
    ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)

    return {
        "coherence_c_v": coherence,        # keep key name for compatibility with README
        "topic_diversity": diversity,
        "n_topics": int(lda_cfg["n_topics"]),
        "train_seconds": train_seconds,
    }


def run_bertopic(docs: List[str], tokens_list: List[List[str]], cfg: Dict, results_dir: str, random_state: int) -> Dict:

    top_n = cfg["general"]["top_n_words"]
    bcfg = cfg["bertopic"]

    # --- Dimensionality reduction and clustering (no UMAP/numba, no HDBSCAN)
    pca_components = int(bcfg.get("pca_n_components", 5))
    k = int(bcfg.get("kmeans_n_clusters", 12))
    dim_model = PCA(n_components=pca_components)
    km = KMeans(n_clusters=k, random_state=random_state)

    # --- Extended stopwords + bigrams to reduce "glue" words and boost diversity
    custom_stops = set(ENGLISH_STOP_WORDS) | {
        "people", "like", "use", "good", "think", "know", "problem", "thanks",
        "edu", "com", "subject", "article", "writes", "lines", "organization",
        "university", "mail", "email", "time", "make", "way", "need"
    }
    vec_model = CountVectorizer(
        stop_words=list(custom_stops),  # CountVectorizer requires a list, not a set
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # --- BERTopic configured with PCA + KMeans + custom vectorizer
    # In v0.17.x, pass alternative clusterers via `hdbscan_model`
    # and alternative reducers via `umap_model` (must expose fit_transform)
    topic_model = BERTopic(
        nr_topics=None,                         # no post-merging
        min_topic_size=bcfg["min_topic_size"],
        embedding_model=bcfg["embedding_model"],
        umap_model=dim_model,                   # PCA instead of UMAP
        hdbscan_model=km,                       # KMeans passed via hdbscan_model
        vectorizer_model=vec_model,             # custom stopwords + bigrams
        calculate_probabilities=False,
        verbose=False,
    )

    # --- Fit the model
    t0 = time.time()
    topics, _ = topic_model.fit_transform(docs)
    train_seconds = time.time() - t0

    # --- Collect top words per topic (guard against -1 even though KMeans shouldn't produce it)
    topic_info = topic_model.get_topics()
    topics_words = []
    valid_topic_ids = [tid for tid in topic_info.keys() if tid != -1]
    valid_topic_ids.sort()
    for tid in valid_topic_ids:
        words = [w for w, _ in topic_info[tid][:top_n]]
        topics_words.append(words)

    # --- Metrics
    coherence = coherence_npmi(topics_words, tokens_list)
    diversity = topic_diversity(topics_words)

    # --- Save topic table
    rows = [
        {
            "topic_id": tid,
            "top_words": ", ".join([w for w, _ in topic_model.get_topics()[tid][:top_n]]),
        }
        for tid in valid_topic_ids
    ]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(results_dir, "topics", "bertopic_topics.csv")
    ensure_dir(out_csv)
    df.to_csv(out_csv, index=False)

    return {
        "coherence_c_v": coherence,            # keep key name for README compatibility
        "topic_diversity": diversity,
        "n_topics": int(len(valid_topic_ids)),
        "train_seconds": train_seconds,
    }



def main():
    parser = argparse.ArgumentParser(description="P2 — Topics: LDA vs BERTopic")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--fast", action="store_true", help="Use small subset for a quick demo")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random_state = int(cfg["general"]["random_state"])

    # Load data
    cats = cfg["general"]["fast"]["categories"] if args.fast else None
    max_docs = int(cfg["general"]["fast"]["max_docs"]) if args.fast else None

    docs, tokens_list = load_20newsgroups_docs(
        fast=args.fast,
        categories=cats,
        max_docs=max_docs,
        random_state=random_state,
    )

    results_dir = "results"
    metrics = {}

    # LDA
    lda_metrics = run_lda(docs, tokens_list, cfg, results_dir, random_state)
    metrics["LDA"] = lda_metrics

    # BERTopic
    bert_metrics = run_bertopic(docs, tokens_list, cfg, results_dir, random_state)
    metrics["BERTopic"] = bert_metrics

    # Save metrics
    save_json(os.path.join(results_dir, "metrics.json"), metrics)

    # Plot
    plot_coherence_diversity(metrics, os.path.join(results_dir, "plots", "coherence_vs_diversity.png"))

    print("Done. See 'results/' for metrics, plots, and topic tables.")
    print(metrics)


if __name__ == "__main__":
    main()
