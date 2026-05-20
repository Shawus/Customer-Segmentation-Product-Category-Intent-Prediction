"""
Data cleaning module for clustering pipeline.

Handles type casting, null handling, and column validation
for raw shipment and CRM data before feature engineering.
"""
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class ColumnConfig:
    """Column type configuration for data cleaning."""
    int_cols: List[str] = field(default_factory=list)
    float_cols: List[str] = field(default_factory=list)
    string_cols: List[str] = field(default_factory=list)
    time_cols: List[str] = field(default_factory=list)


class ClusteringDataCleaning:
    """
    Cleans raw data by enforcing column types and handling nulls.

    Usage:
        config = ColumnConfig(
            int_cols=[""],
            float_cols=["", ""],
            string_cols=["", "", "", "", "", ""]
        )
        cleaner = ClusteringDataCleaning(columns=config)
        cleaned_df = cleaner.clean(raw_df)
    """

    def __init__(self, columns: ColumnConfig):
        self.columns = columns

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply type casting and null removal."""
        df = df.copy()

        for col in self.columns.int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        for col in self.columns.float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.columns.string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({"nan": None, "": None, "None": None})

        for col in self.columns.time_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Drop rows with nulls in critical string columns
        critical_cols = [c for c in self.columns.string_cols if c in df.columns]
        df = df.dropna(subset=critical_cols)

        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
