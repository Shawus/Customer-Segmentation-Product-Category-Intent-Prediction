"""
Word2Vec embedding generator for product categories.

Generates fixed-dimension embeddings for product category identifiers
using Skip-gram Word2Vec, enabling the ranking model to learn
product similarity in vector space.
"""
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

from gensim.models import Word2Vec

logging.getLogger("gensim").setLevel(logging.WARNING)


class EmbeddingProcessor:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.data_path = self.current_dir / "data"

        self.vector_dim = 32
        self.window_size = 3
        self.min_count = 1
        self.sg = 1  # Skip-gram
        self.epochs = 10

    def embedding_products(self, df, column):
        """
        Generate Word2Vec embeddings for product categories.

        Args:
            df: DataFrame containing the product category column.
            column: Column name with product identifiers.

        Returns:
            DataFrame with added '{column}_embedding' column.
        """
        items = df[column].unique().tolist()
        tokenized = [[c] for c in items]

        model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_dim,
            window=self.window_size,
            min_count=self.min_count,
            sg=self.sg,
            workers=8,
            epochs=self.epochs,
        )

        embedding = {
            item: model.wv[item] for item in model.wv.key_to_index
        }

        # Save embeddings to JSON for inference use
        with open(self.data_path / f"{column}_embedding.json", "w", encoding="utf-8") as f:
            json.dump({k: v.tolist() for k, v in embedding.items()}, f, ensure_ascii=False)

        df[f"{column}_embedding"] = df[column].map(
            lambda x: embedding.get(x, np.zeros(self.vector_dim))
        )
        df = df.dropna(subset=[f"{column}_embedding"])

        return df

    def expanding_embedding_vector(self, df, column):
        """Expand embedding list column into individual numeric columns."""
        embedding_cols = [f"{column}_embedding_{i}" for i in range(self.vector_dim)]
        df[embedding_cols] = pd.DataFrame(
            df[f"{column}_embedding"].tolist(), index=df.index
        )
        return df
