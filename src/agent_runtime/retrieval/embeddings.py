# src/agent_runtime/retrieval/embeddings.py
import hashlib
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Convert one text string into a dense feature vector."""
        pass


class FakeEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimensions: int = 4):
        self.dimensions = dimensions

    @staticmethod
    def _contains_any(text: str, keywords: List[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def get_embedding(self, text: str) -> List[float]:
        """
        Deterministic pseudo-embedding provider.

        The implementation maps known test-domain concepts to stable vectors and
        falls back to an MD5-derived vector for repeatable local tests.
        """
        if not text:
            return [0.0] * self.dimensions

        normalized = text.lower()

        time_series_keywords = [
            "arima",
            "time series",
            "forecast",
            "forecasting",
            "historical",
            "sequential",
            "stochastic",
            "series",
            "training",
        ]
        backtest_keywords = [
            "backtest",
            "backtesting",
            "trading",
            "systematic",
            "quantitative",
            "data leakage",
            "leakage",
        ]

        if self._contains_any(normalized, time_series_keywords):
            return [0.9, 0.1, 0.0, 0.0]
        if self._contains_any(normalized, backtest_keywords):
            return [0.0, 0.0, 0.8, 0.2]

        hasher = hashlib.md5(text.encode("utf-8"))
        digest = hasher.digest()

        vector = []
        for i in range(self.dimensions):
            val = digest[i % len(digest)] / 255.0
            vector.append(round(val, 4))
        return vector
