"""Turns ticket text into numbers K-means can work with: TF-IDF on the words,
then squished down to ~50 dimensions with SVD."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import Normalizer

# Redaction placeholders and greeting/boilerplate words carry no topic signal.
REDACTION_TOKENS = ["email_redacted", "phone_redacted", "card_redacted"]
BOILERPLATE = ["hi", "hello", "hey", "team", "support", "thanks", "thank", "please", "help", "regards", "ticket"]
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | set(REDACTION_TOKENS) | set(BOILERPLATE))


@dataclass
class TextVectorizer:
    max_features: int = 5000
    svd_components: int = 50
    random_state: int = 42
    tfidf: Optional[TfidfVectorizer] = field(default=None, repr=False)
    svd: Optional[TruncatedSVD] = field(default=None, repr=False)
    normalizer: Normalizer = field(default_factory=Normalizer, repr=False)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        n_docs = len(texts)
        self.tfidf = TfidfVectorizer(
            stop_words=STOP_WORDS,
            ngram_range=(1, 2),
            min_df=2 if n_docs >= 50 else 1,
            max_df=0.9,
            sublinear_tf=True,
            max_features=self.max_features,
        )
        X = self.tfidf.fit_transform(texts)
        n_comp = min(self.svd_components, X.shape[1] - 1, n_docs - 1)
        if n_comp >= 2:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
            X = self.svd.fit_transform(X)
        else:
            self.svd = None
            X = X.toarray()
        return self.normalizer.fit_transform(X)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.tfidf is None:
            raise RuntimeError("TextVectorizer is not fitted")
        X = self.tfidf.transform(texts)
        X = self.svd.transform(X) if self.svd is not None else X.toarray()
        return self.normalizer.transform(X)

    def top_terms(self, centers: np.ndarray, top_n: int = 8) -> List[List[str]]:
        """Map cluster centres back to TF-IDF space and return the highest-weighted terms."""
        vocab = self.tfidf.get_feature_names_out()
        weights = self.svd.inverse_transform(centers) if self.svd is not None else centers
        return [[str(vocab[i]) for i in np.argsort(row)[::-1][:top_n]] for row in weights]
