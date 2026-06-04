"""Feature engineering for the SSH session dataset.

Reads the session-level CSV produced by ``data_labeling.build_session_dataset``
and prepares feature matrices ready for ML training.

Feature selection rationale (from ssh_session_design.md §3):
- **Drop** ``ip``              – categorical metadata, too many unique values.
- **Drop** ``total_failures``  – leaks labeling rules directly.
- **Drop** ``total_successes`` – leaks Class 4 (Break-in) label.
- **Drop** ``unique_users_count`` – redundant with ``unique_users_ratio``.
- **Drop** ``ts_first``, ``ts_last`` – metadata timestamps.
- **Consider drop** ``total_attempts`` – correlated; default is drop.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# Columns to always drop (metadata / label-leaking)
_DROP_ALWAYS = [
	"ip",
	"total_failures",
	"total_successes",
	"unique_users_count",
	"ts_first",
	"ts_last",
]

# Column to drop by default (correlated with ratios)
_DROP_DEFAULT = [
	"total_attempts",
]

# Final training features (order matters for consistency)
TRAINING_FEATURES = [
	"is_private",
	"session_duration",
	"attempts_per_second",
	"is_single_event",
	"failure_ratio",
	"unique_users_ratio",
	"has_root_attempt",
	"has_valid_user_attempt",
	"max_failure_streak",
	"invalid_user_attempts",
	"has_reverse_mapping_failed",
]

TARGET = "class"

# Class label names
CLASS_NAMES = {
	0: "Normal",
	1: "Single failure",
	2: "Scan",
	3: "Brute-force",
	4: "Break-in",
}


# ---------------------------------------------------------------------------
# Loading & preparation
# ---------------------------------------------------------------------------

def load_session_csv(path: str | Path) -> pd.DataFrame:
	"""Load the session CSV and set correct dtypes."""
	df = pd.read_csv(path)

	# Ensure integer columns
	int_cols = [
		"is_private", "total_attempts", "is_single_event",
		"total_failures", "total_successes", "unique_users_count",
		"has_root_attempt", "has_valid_user_attempt",
		"max_failure_streak", "invalid_user_attempts",
		"has_reverse_mapping_failed", "class",
	]
	for col in int_cols:
		if col in df.columns:
			df[col] = df[col].astype(int)

	return df


def prepare_features(
	df: pd.DataFrame,
	*,
	keep_total_attempts: bool = False,
	log_transform: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
	"""Select and transform features for ML training.

	Parameters
	----------
	df : pd.DataFrame
		Session-level DataFrame (from ``load_session_csv``).
	keep_total_attempts : bool
		If True, retain ``total_attempts`` as a feature.
	log_transform : list of str or None
		Column names to apply log1p transform (good for skewed features).
		Default: ``["session_duration", "attempts_per_second", "max_failure_streak",
		"invalid_user_attempts"]``.

	Returns
	-------
	X : pd.DataFrame
		Feature matrix.
	y : pd.Series
		Target labels (0-4).
	"""

	if log_transform is None:
		log_transform = [
			"session_duration",
			"attempts_per_second",
			"max_failure_streak",
			"invalid_user_attempts",
		]

	# --- Drop columns ---
	drop_cols = list(_DROP_ALWAYS)
	if not keep_total_attempts:
		drop_cols.extend(_DROP_DEFAULT)

	features = [c for c in TRAINING_FEATURES if c not in drop_cols]
	if keep_total_attempts and "total_attempts" not in features:
		features.insert(2, "total_attempts")  # after attempts_per_second

	X = df[features].copy()
	y = df[TARGET].copy()

	# --- Handle edge cases (before transforms) ---
	# When is_single_event=1, attempts_per_second (1/(0+1)=1) and
	# unique_users_ratio (1/1=1.0) are meaningless artefacts.
	# Zero them out so they don't mislead the model.
	if "is_single_event" in X.columns:
		single_mask = X["is_single_event"] == 1
		if "attempts_per_second" in X.columns:
			X.loc[single_mask, "attempts_per_second"] = 0.0
		if "unique_users_ratio" in X.columns:
			X.loc[single_mask, "unique_users_ratio"] = 0.0

	# --- Log transform skewed features ---
	for col in log_transform:
		if col in X.columns:
			X[col] = np.log1p(X[col].astype(float))

	return X, y


def describe_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
	"""Return a summary DataFrame of feature statistics per class."""
	combined = X.copy()
	combined[TARGET] = y

	stats = []
	for cls in sorted(y.unique()):
		subset = combined[combined[TARGET] == cls]
		row = {"class": cls, "label": CLASS_NAMES.get(cls, "?"), "count": len(subset)}
		for col in X.columns:
			row[f"{col}_mean"] = subset[col].mean()
			row[f"{col}_std"] = subset[col].std()
		stats.append(row)

	return pd.DataFrame(stats)
