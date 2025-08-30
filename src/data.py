from __future__ import annotations
import random
import re
from typing import List, Tuple, Optional
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_word_re = re.compile(r"[A-Za-z]+")

def _tokenize(text: str) -> List[str]:
    # Simple, fast tokenizer: keep alphabetic tokens, lowercase, drop very short words
    return [w.lower() for w in _word_re.findall(text) if len(w) > 2]

def _remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]

def load_20newsgroups_docs(
    fast: bool,
    categories: Optional[List[str]] = None,
    max_docs: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[List[str], List[List[str]]]:
    # Returns (docs, tokens_list)
    if fast and categories is None:
        categories = ["comp.graphics", "sci.space", "rec.sport.hockey", "talk.politics.mideast"]

    data = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"))
    docs = data.data

    if max_docs is not None and len(docs) > max_docs:
        random.Random(random_state).shuffle(docs)
        docs = docs[:max_docs]

    tokens_list = [_remove_stopwords(_tokenize(d)) for d in docs]
    return docs, tokens_list
