from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from .registry import register_dataset
from .base import BaseDataset, BaseDatasetLoader
import numpy as np


@register_dataset("20newsgroups")
class TwentyNewsgroups(BaseDatasetLoader):
    def __init__(self, root: str = None, categories=None, max_features=1000):
        super().__init__("20newsgroups", root)
        self.categories = categories or [
            "alt.atheism",
            "comp.graphics",
            "sci.space",
            "talk.religion.misc",
        ]
        self.max_features = max_features

    def load(self) -> BaseDataset:
        # Load the data
        newsgroups = fetch_20newsgroups(
            subset="train",
            categories=self.categories,
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=42,
        )

        # Convert text to TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, stop_words="english", max_df=0.95, min_df=2
        )
        X = vectorizer.fit_transform(newsgroups.data).toarray()
        y = newsgroups.target

        return BaseDataset(
            name="20newsgroups",
            data=X,
            targets=y,
            feature_names=vectorizer.get_feature_names_out(),
            target_names=newsgroups.target_names,
        )


@register_dataset("20newsgroups_full")
class TwentyNewsgroupsFull(BaseDatasetLoader):
    def __init__(self, root: str = None, max_features=2000):
        super().__init__("20newsgroups_full", root)
        self.max_features = max_features

    def load(self) -> BaseDataset:
        # Load all 20 categories
        newsgroups = fetch_20newsgroups(
            subset="train",
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=42,
        )

        # Convert text to TF-IDF features
        vectorizer = TfidfVectorizer(
            max_features=self.max_features, stop_words="english", max_df=0.95, min_df=2
        )
        X = vectorizer.fit_transform(newsgroups.data).toarray()
        y = newsgroups.target

        return BaseDataset(
            name="20newsgroups_full",
            data=X,
            targets=y,
            feature_names=vectorizer.get_feature_names_out(),
            target_names=newsgroups.target_names,
        )
