from __future__ import annotations

from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


class Model_Interpretability:
	"""Model interpretability utilities.
	"""

	@staticmethod
	def _unwrap_estimator(model_or_result: Any) -> Any:
		"""Accept either an estimator or a TrainResult-like object."""
		if hasattr(model_or_result, "best_estimator"):
			return getattr(model_or_result, "best_estimator")
		return model_or_result

	@staticmethod
	def _ensure_dataframe(X: Any, *, feature_names: list[str] | None = None) -> pd.DataFrame:
		if isinstance(X, pd.DataFrame):
			return X
		X_arr = np.asarray(X)
		if X_arr.ndim != 2:
			raise ValueError("X must be 2D")
		if feature_names is None:
			feature_names = [f"f{i}" for i in range(X_arr.shape[1])]
		return pd.DataFrame(X_arr, columns=feature_names)

	@staticmethod
	def _maybe_sample(
		X: pd.DataFrame,
		*,
		max_samples: int | None,
		random_state: int = 42,
	) -> pd.DataFrame:
		if max_samples is None or len(X) <= max_samples:
			return X
		rng = np.random.default_rng(random_state)
		idx = rng.choice(len(X), size=max_samples, replace=False)
		return X.iloc[idx]

	@staticmethod
	def _select_output(shap_values: Any, *, class_index: int | None) -> Any:
		"""Select a single output from multi-output SHAP explanations.

		Works for:
		- shap.Explanation with values shape (n, p) -> returned as-is
		- shap.Explanation with values shape (n, p, k) -> select [:, :, class_index]
		"""
		values = getattr(shap_values, "values", None)
		if values is None:
			return shap_values
		if values.ndim == 2:
			return shap_values
		if values.ndim == 3:
			if class_index is None:
				class_index = 1 if values.shape[-1] > 1 else 0
			return shap_values[:, :, class_index]
		return shap_values

	@staticmethod
	def shap_summary_plot(
		model: Any,
		X: Any,
		*,
		max_samples: int | None = 2000,
		random_state: int = 42,
		class_index: int | None = None,
		plot_type: Literal["dot", "bar"] = "dot",
		title: str | None = None,
		show: bool = False,
	):
		"""Create a SHAP summary plot (global importance).

		Returns a matplotlib Figure.
		"""
		estimator = Model_Interpretability._unwrap_estimator(model)
		X_df = Model_Interpretability._ensure_dataframe(X)
		X_eval = Model_Interpretability._maybe_sample(X_df, max_samples=max_samples, random_state=random_state)

		explainer = shap.Explainer(estimator, X_eval)
		shap_values = explainer(X_eval)
		shap_values = Model_Interpretability._select_output(shap_values, class_index=class_index)

		fig = plt.figure(figsize=(8, 5))
		shap.summary_plot(shap_values, X_eval, plot_type=plot_type, show=show)
		if title:
			plt.title(title)
		return fig

	@staticmethod
	def shap_dependence_plot(
		model: Any,
		X: Any,
		feature: str | int,
		*,
		interaction_feature: str | int | None = "auto",
		max_samples: int | None = 2000,
		random_state: int = 42,
		class_index: int | None = None,
		title: str | None = None,
		show: bool = False,
	):
		"""Create a SHAP dependence plot for a feature.

		Returns a matplotlib Figure.
		"""
		estimator = Model_Interpretability._unwrap_estimator(model)
		X_df = Model_Interpretability._ensure_dataframe(X)
		X_eval = Model_Interpretability._maybe_sample(X_df, max_samples=max_samples, random_state=random_state)

		explainer = shap.Explainer(estimator, X_eval)
		shap_values = explainer(X_eval)
		shap_values = Model_Interpretability._select_output(shap_values, class_index=class_index)

		fig, ax = plt.subplots(figsize=(7, 5))
		shap.dependence_plot(
			feature,
			shap_values.values if hasattr(shap_values, "values") else shap_values,
			X_eval,
			interaction_index=interaction_feature,
			ax=ax,
			show=show,
		)
		if title:
			ax.set_title(title)
		fig.tight_layout()
		return fig

	@staticmethod
	def shap_waterfall_plot(
		model: Any,
		X: Any,
		*,
		row_index: int = 0,
		max_display: int = 15,
		class_index: int | None = None,
		title: str | None = None,
		show: bool = False,
	):
		"""Create a SHAP waterfall plot for a single row (local explanation).

		Returns a matplotlib Figure.
		"""
		estimator = Model_Interpretability._unwrap_estimator(model)
		X_df = Model_Interpretability._ensure_dataframe(X)
		if row_index < 0 or row_index >= len(X_df):
			raise IndexError("row_index out of range")

		# Use a small background for masking speed/stability.
		background = Model_Interpretability._maybe_sample(X_df, max_samples=min(200, len(X_df)), random_state=42)
		explainer = shap.Explainer(estimator, background)

		row = X_df.iloc[[row_index]]
		shap_row = explainer(row)
		shap_row = Model_Interpretability._select_output(shap_row, class_index=class_index)

		fig = plt.figure(figsize=(8, 5))
		shap.plots.waterfall(shap_row[0], max_display=max_display, show=show)
		if title:
			plt.title(title)
		return fig

