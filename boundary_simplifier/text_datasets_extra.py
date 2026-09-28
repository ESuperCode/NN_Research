"""
text_datasets_extra.py

Two more real-world-style binary text classification tasks, alongside the
20-Newsgroups topic task already in data_utils.py:
  - scam/spam detection (spam vs. legitimate message)
  - sentiment analysis (positive vs. negative review)

For each, loading tries three things in order:
  1. A user-supplied local CSV (`csv_path=...`) -- use this for your paper.
     Expects two columns identifiable as text and label; column names are
     auto-detected from a short list of common names (see `_read_csv_flex`).
  2. A known public dataset fetched over the network, if reachable.
  3. A bundled synthetic fallback corpus (clearly logged as synthetic),
     purely so the pipeline is runnable offline for development/testing.

IMPORTANT FOR YOUR PAPER: (2) and (3) exist so this code runs everywhere,
but neither is a substitute for a citable, verified real dataset. For
publication-quality numbers, download a real corpus yourself (e.g. the UCI
SMS Spam Collection, or an IMDB/Sentiment140 sentiment dump), and pass it
in via `csv_path`. The network mirrors below are convenience only, not
guaranteed stable, and not something you should cite as your data source.
"""

import io
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

_TEXT_COL_CANDIDATES = ["text", "message", "review", "sms", "content", "body"]
_LABEL_COL_CANDIDATES = ["label", "class", "target", "sentiment", "category", "v1"]


def _read_csv_flex(path_or_buffer):
    df = pd.read_csv(path_or_buffer, encoding="latin-1")
    df.columns = [c.strip().lower() for c in df.columns]
    text_col = next((c for c in _TEXT_COL_CANDIDATES if c in df.columns), None)
    label_col = next((c for c in _LABEL_COL_CANDIDATES if c in df.columns), None)
    if text_col is None or label_col is None:
        # Fall back to the classic 2-column "v1,v2" layout used by several
        # public spam CSV mirrors (label first, text second).
        if df.shape[1] >= 2:
            label_col, text_col = df.columns[0], df.columns[1]
        else:
            raise ValueError(f"Could not identify text/label columns in {df.columns.tolist()}")
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col]
    return texts, labels


def _binarize_labels(labels, positive_values):
    """Maps arbitrary label strings/values to 0/1 given the set of values
    that should map to 1."""
    positive_values = {str(v).strip().lower() for v in positive_values}
    return np.array([1 if str(v).strip().lower() in positive_values else 0 for v in labels])


def _vectorize_and_split(texts, y, max_features, test_size, random_state):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, vectorizer


# ---------------------------------------------------------------------------
# Synthetic fallback corpora (offline only -- see module docstring)
# ---------------------------------------------------------------------------

_SPAM_VOCAB = [
    "winner", "claim", "prize", "urgent", "click", "verify", "suspended",
    "wire", "transfer", "bitcoin", "guarantee", "free", "offer", "limited",
    "congratulations", "lottery", "inheritance", "act", "now", "risk",
    "cash", "bonus", "unsubscribe", "credit", "loan", "apply",
]
_HAM_VOCAB = [
    "meeting", "schedule", "attached", "project", "report", "team", "lunch",
    "reminder", "thanks", "regards", "update", "invoice", "family",
    "weekend", "review", "deadline", "call", "conference", "budget",
    "colleague", "vacation", "appointment", "presentation", "feedback",
]
_POS_VOCAB = [
    "amazing", "brilliant", "loved", "fantastic", "wonderful", "superb",
    "recommend", "excellent", "delightful", "masterpiece", "enjoyed",
    "captivating", "outstanding", "charming", "impressive", "beautiful",
    "touching", "hilarious", "gripping", "flawless",
]
_NEG_VOCAB = [
    "terrible", "awful", "boring", "waste", "disappointing", "poor",
    "worst", "dull", "mediocre", "regret", "unwatchable", "cliche",
    "flat", "tedious", "forgettable", "clumsy", "shallow", "predictable",
    "overrated", "insufferable",
]
_FILLER = [
    "the", "was", "is", "this", "movie", "film", "product", "really",
    "very", "just", "quite", "so", "and", "but", "with", "for", "it",
    "one", "some", "you", "your", "we", "our", "have", "will",
]


def _make_doc(vocab, rng, n_words=25):
    words = []
    for _ in range(n_words):
        words.append(rng.choice(vocab) if rng.random() < 0.55 else rng.choice(_FILLER))
    return " ".join(words)


def _synthetic_corpus(vocab_a, vocab_b, n_per_class, seed):
    rng = random.Random(seed)
    docs = [_make_doc(vocab_a, rng) for _ in range(n_per_class)] + \
           [_make_doc(vocab_b, rng) for _ in range(n_per_class)]
    labels = [0] * n_per_class + [1] * n_per_class
    return docs, np.array(labels)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_scam_detection_data(csv_path=None, max_features=3000, test_size=0.25,
                              random_state=42, n_per_class=500):
    """Binary spam/scam (1) vs. legitimate (0) message classification."""
    if csv_path is not None:
        texts, labels = _read_csv_flex(csv_path)
        y = _binarize_labels(labels, positive_values=["spam", "1", "scam", "true"])
        source = f"user-supplied CSV: {csv_path}"
    else:
        try:
            import urllib.request
            url = ("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
                   "master/data/sms.tsv")
            with urllib.request.urlopen(url, timeout=6) as resp:
                raw = resp.read().decode("latin-1")
            df = pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["label", "text"])
            texts = df["text"].tolist()
            y = _binarize_labels(df["label"], positive_values=["spam"])
            source = f"network fetch: {url}"
        except Exception as e:
            texts, y = _synthetic_corpus(_HAM_VOCAB, _SPAM_VOCAB, n_per_class, random_state)
            source = f"SYNTHETIC offline fallback (network unavailable: {e!r})"

    print(f"[text_datasets_extra] scam detection data source: {source}")
    X_train, X_test, y_train, y_test, vec = _vectorize_and_split(
        texts, y, max_features, test_size, random_state
    )
    return X_train, X_test, y_train, y_test, source


def load_sentiment_data(csv_path=None, max_features=3000, test_size=0.25,
                         random_state=42, n_per_class=500):
    """Binary positive (1) vs. negative (0) sentiment classification."""
    if csv_path is not None:
        texts, labels = _read_csv_flex(csv_path)
        y = _binarize_labels(labels, positive_values=["1", "pos", "positive", "4"])
        source = f"user-supplied CSV: {csv_path}"
    else:
        try:
            import urllib.request
            url = ("https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/"
                   "master/train.csv")
            with urllib.request.urlopen(url, timeout=6) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(raw))
            text_col = next(c for c in df.columns if "tweet" in c.lower() or "text" in c.lower())
            label_col = next(c for c in df.columns if "label" in c.lower())
            texts = df[text_col].astype(str).tolist()
            y = _binarize_labels(df[label_col], positive_values=["0"])  # dataset uses 0=non-hate; kept as illustrative binary split
            source = f"network fetch: {url}"
        except Exception as e:
            texts, y = _synthetic_corpus(_NEG_VOCAB, _POS_VOCAB, n_per_class, random_state)
            source = f"SYNTHETIC offline fallback (network unavailable: {e!r})"

    print(f"[text_datasets_extra] sentiment data source: {source}")
    X_train, X_test, y_train, y_test, vec = _vectorize_and_split(
        texts, y, max_features, test_size, random_state
    )
    return X_train, X_test, y_train, y_test, source


if __name__ == "__main__":
    for name, loader in [("scam", load_scam_detection_data), ("sentiment", load_sentiment_data)]:
        X_train, X_test, y_train, y_test, source = loader()
        print(f"{name}: train={X_train.shape} test={X_test.shape} "
              f"class balance train={np.bincount(y_train)}\n")
