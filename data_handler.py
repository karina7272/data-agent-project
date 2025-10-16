"""Data handling utilities for the Data Agent project.

The ``DataHandler`` class encapsulates all logic related to loading a
tabular dataset, inferring its schema and basic statistics, cleaning
missing values, performing correlations, clustering and anomaly detection.

This module intentionally keeps all heavy computation in pure Python and
pandas/numpy/scikit‑learn to make it easy to unit test.  It does not
depend on any specific LLM provider; that integration lives in
``agent.py``.  Consumers should construct a ``DataHandler`` with a path
or URL to their dataset and call ``load_dataset`` before executing
queries.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Optional import for anomaly detection.  If ``scikit-learn`` is not
# available at runtime, the anomaly detection and clustering methods
# gracefully degrade.
try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None  # type: ignore


@dataclass
class DataHandler:
    """Encapsulates logic for loading and analysing a dataset.

    Parameters
    ----------
    dataset_path : str
        Local path to a CSV or parquet file.  If the file does not exist
        locally and ``dataset_url`` is provided, the handler will attempt
        to download it.
    dataset_url : Optional[str], optional
        URL from which to download the dataset if it's not already on
        disk.  Supported protocols depend on the runtime environment.
    """

    dataset_path: str
    dataset_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset into a pandas DataFrame.

        If ``dataset_path`` exists, it will be read directly.  Otherwise,
        if ``dataset_url`` is provided, the file is downloaded into a
        temporary directory and then read.  The method infers whether
        the file is CSV or parquet based on its extension.

        Returns
        -------
        pd.DataFrame
            The loaded dataset.
        """
        path = self.dataset_path
        # Attempt to load local file first.
        if not os.path.exists(path):
            if not self.dataset_url:
                raise FileNotFoundError(f"Dataset not found at {path} and no URL provided.")
            path = self._download_dataset(self.dataset_url)
        # Infer file type and load accordingly.
        _, ext = os.path.splitext(path.lower())
        if ext in {".csv", ".txt"}:
            df = pd.read_csv(path)
        elif ext in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported dataset extension: {ext}")
        self.df = df
        return df

    def _download_dataset(self, url: str) -> str:
        """Download a dataset from a remote URL into a temporary file.

        This helper uses ``pandas``' built‑in read_csv/read_parquet to
        stream the dataset directly from the URL.  If this is not
        supported for a particular protocol, consumers should download
        externally and supply a local path.

        Parameters
        ----------
        url : str
            The URL to download from.

        Returns
        -------
        str
            Path to the downloaded file on disk.
        """
        tmp_dir = tempfile.gettempdir()
        local_filename = os.path.join(tmp_dir, os.path.basename(url))
        # simple streaming download via pandas; fallback to requests if needed
        try:
            if url.endswith(".csv") or ".csv" in url:
                df = pd.read_csv(url)
                df.to_csv(local_filename, index=False)
            elif url.endswith(".parquet") or ".parquet" in url:
                df = pd.read_parquet(url)
                df.to_parquet(local_filename, index=False)
            else:
                raise ValueError("Unsupported remote dataset format. Only CSV and Parquet are supported.")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from {url}: {e}")
        return local_filename

    def infer_schema(self) -> Dict[str, str]:
        """Infer a simple schema mapping column names to data types.

        Returns
        -------
        Dict[str, str]
            A mapping from column name to inferred pandas dtype (string).
        """
        if self.df is None:
            raise RuntimeError("Dataset must be loaded before inferring schema.")
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def handle_missing_values(self, strategy: str = "drop") -> None:
        """Handle missing values in the dataset.

        Parameters
        ----------
        strategy : str, optional
            ``"drop"`` removes rows with any missing values.  ``"fill"``
            replaces numeric NaNs with their column mean and categorical
            NaNs with the mode.  Default is ``"drop"``.
        """
        if self.df is None:
            raise RuntimeError("Dataset must be loaded before handling missing values.")
        if strategy == "drop":
            self.df = self.df.dropna().reset_index(drop=True)
        elif strategy == "fill":
            df = self.df.copy()
            for col in df.columns:
                if df[col].dtype.kind in "iufc":  # numeric
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    mode = df[col].mode().iloc[0] if not df[col].mode().empty else None
                    df[col] = df[col].fillna(mode)
            self.df = df
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

    def get_summary(self) -> pd.DataFrame:
        """Compute summary statistics for numeric columns.

        Returns
        -------
        pd.DataFrame
            Summary statistics (count, mean, std, min, max, quartiles).
        """
        if self.df is None:
            raise RuntimeError("Dataset must be loaded before summarizing.")
        numeric_cols = self.df.select_dtypes(include=[np.number])
        return numeric_cols.describe()

    def get_columns(self) -> List[str]:
        """Return the list of column names."""
        if self.df is None:
            raise RuntimeError("Dataset must be loaded before getting columns.")
        return list(self.df.columns)

    def compute_correlation(self) -> pd.DataFrame:
        """Compute the correlation matrix for numeric columns.

        Returns
        -------
        pd.DataFrame
            Correlation matrix of numeric features.
        """
        if self.df is None:
            raise RuntimeError("Dataset must be loaded before computing correlation.")
        numeric_cols = self.df.select_dtypes(include=[np.number])
        if numeric_cols.empty:
            raise ValueError("No numeric columns available for correlation analysis.")
        return numeric_cols.corr()

    def cluster_data(self, n_clusters: int = 3, features: Optional[List[str]] = None) -> Tuple[np.ndarray, KMeans]:
        """Perform k‑means clustering on selected features.

        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters to form.
        features : list of str, optional
            Columns to use for clustering.  If ``None``, all numeric
            columns are used.

        Returns
        -------
        labels : np.ndarray
            Cluster labels for each row.
        model : KMeans
            The fitted k‑means model.
        """
        if self.df is None:
            raise RuntimeError("Dataset must be loaded before clustering.")
        df = self.df
        if features is None:
            features = list(df.select_dtypes(include=[np.number]).columns)
        if not features:
            raise ValueError("No numeric features available for clustering.")
        X = df[features].dropna().to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        return labels, model

    def detect_anomalies(self, feature: str, z_thresh: float = 3.0) -> List[int]:
        """Detect anomalies in a numeric feature using z‑score.

        Parameters
        ----------
        feature : str
            Column name on which to detect anomalies.
        z_thresh : float, optional
            Z‑score threshold beyond which a value is considered an anomaly.
            Default is 3.0.

        Returns
        -------
        List[int]
            Indices of rows in ``self.df`` considered anomalous.
        """
        if self.df is None:
            raise RuntimeError("Dataset must be loaded before detecting anomalies.")
        if feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset.")
        series = self.df[feature].dropna()
        if series.dtype.kind not in "iufc":
            raise ValueError(f"Anomaly detection requires a numeric feature; got {series.dtype}.")
        if stats is None:
            raise RuntimeError("scipy is required for anomaly detection but is not installed.")
        z_scores = np.abs(stats.zscore(series))
        anomalies = np.where(z_scores > z_thresh)[0].tolist()
        return anomalies