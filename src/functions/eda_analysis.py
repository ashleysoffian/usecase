import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import missingno as msno
import rfpimp
import numpy as np
from pandas.api.types import is_numeric_dtype

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Color scheme
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Accent purple
    'success': '#06A77D',      # Green
    'warning': '#F18F01',      # Orange
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'box_fill': '#E8F4F8',     # Light blue for boxes
    'hist_fill': '#2E86AB',    # Blue for histograms
}

class EDA_Analysis:
    """
    EDA visualization toolkit for data analysis.
    
    Provides functions and visualizations for exploratory data analysis
    including missing data analysis, distributions, outliers, and correlations.
    """

    @staticmethod
    def _table_col_widths(headers, rows, *, min_col: float = 0.04, max_col: float = 0.30):
        """Compute relative column widths based on rendered text lengths."""
        if not headers:
            return []

        def _cell_len(v) -> int:
            s = "" if v is None else str(v)
            return max(len(s), 1)

        col_max = [len(str(h)) for h in headers]
        for row in rows:
            for j in range(min(len(headers), len(row))):
                col_max[j] = max(col_max[j], _cell_len(row[j]))

        # Convert lengths to weights; sqrt dampens very long text dominating.
        weights = np.sqrt(np.array(col_max, dtype=float))
        weights = np.maximum(weights, 1.0)
        widths = weights / weights.sum()

        # Enforce min/max per column and re-normalize.
        widths = np.clip(widths, min_col, max_col)
        widths = widths / widths.sum()
        return widths.tolist()

    @staticmethod
    def _table_figsize(n_rows: int, n_cols: int, *, row_height: float = 0.30, col_width: float = 1.05,
                      min_w: float = 6.5, max_w: float = 16.0, min_h: float = 1.6, pad_h: float = 1.0):
        """Compute a tight-ish figure size for table-like plots."""
        # +1 for header row
        h = max(min_h, (n_rows + 1) * row_height + pad_h)
        w = max(min_w, min(max_w, n_cols * col_width))
        return (w, h)
    
    @staticmethod
    def _set_plot_style(ax, title=None, xlabel=None, ylabel=None):
        """
        Apply consistent professional styling to plots.
        
        Args:
            ax: Matplotlib axis object
            title (str, optional): Plot title
            xlabel (str, optional): X-axis label
            ylabel (str, optional): Y-axis label
        """
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11, fontweight='medium')
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11, fontweight='medium')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=10)

    @staticmethod
    def _get_numeric_columns(data, variables=None):
        """
        Helper method to get numeric columns from data.
        
        Args:
            data (DataFrame): Input dataframe
            variables (list, optional): Specific variables to filter
            
        Returns:
            list: List of numeric column names
            
        Raises:
            TypeError: If variables is not a list
        """
        numeric_cols = [col for col in data.columns if is_numeric_dtype(data[col])]
        
        if variables is None:
            return numeric_cols
        
        if not isinstance(variables, list):
            raise TypeError("Expected variables to be a list")
            
        return [col for col in variables if col in numeric_cols]

    @staticmethod
    def data_info(data):
        """
        Generate data info table as matplotlib figure.

        Args:
            data (DataFrame): Input dataframe

        Returns:
            Figure: Matplotlib figure with formatted table
        """
        n_rows = len(data)
        n_cols = len(data.columns)

        def _null_pct(null_count: int) -> str:
            if n_rows == 0:
                return "0.0%"
            return f"{(null_count / n_rows * 100):.1f}%"

        info_df = pd.DataFrame(
            {
                '#': range(n_cols),
                'Column': data.columns,
                'Non-Null Count': data.notna().sum().astype(int).to_numpy(),
                'Null Count': data.isna().sum().astype(int).to_numpy(),
                'Null %': [
                    _null_pct(int(x)) for x in data.isna().sum().astype(int).to_numpy()
                ],
                'Dtype': data.dtypes.astype(str).to_numpy(),
            }
        )

        info_data = info_df.to_numpy().tolist()
        headers = info_df.columns.tolist()
        col_widths = EDA_Analysis._table_col_widths(headers, info_data)

        fig_w, fig_h = EDA_Analysis._table_figsize(len(info_data), len(headers))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis('off')

        table = ax.table(
            cellText=info_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            colWidths=col_widths,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        # Header row
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(COLORS['primary'])
            cell.set_text_props(weight='bold', color='white')

        # Body cells
        for i in range(1, len(info_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                cell.set_facecolor('white')
                cell.set_edgecolor('#CCCCCC')

        title_text = (
            "DataFrame Info\n"
            f"RangeIndex: {n_rows} entries, 0 to {max(n_rows - 1, 0)} | Data columns: {n_cols} total"
        )
        ax.set_title(title_text, fontsize=9, fontweight='bold', pad=20, loc='left')

        fig.tight_layout()
        return fig
    
    @staticmethod
    def descriptive_stats(data, top_value_maxlen: int = 30):
        """
        Generate descriptive statistics table as a matplotlib figure.

        - Numeric columns: count, missing, mean, std, min, 25%, 50%, 75%, max
        - Non-numeric columns: count, missing, unique, top, freq

        Args:
            data (DataFrame): Input dataframe
            top_value_maxlen (int): Max length for 'Top' (categorical) value display

        Returns:
            Figure: Matplotlib figure with formatted table
        """
        n_rows = len(data)
        n_cols = len(data.columns)

        def _fmt(v) -> str:
            """Format values for table cells."""
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return ""
            if isinstance(v, (np.integer, int)):
                return str(int(v))
            if isinstance(v, (np.floating, float)):
                # keep readable, avoid long floats
                return f"{float(v):.1f}"
            s = str(v)
            s = s.replace("\n", " ")
            if len(s) > top_value_maxlen:
                s = s[: top_value_maxlen - 3] + "..."
            return s

        numeric_cols = [c for c in data.columns if is_numeric_dtype(data[c])]
        categorical_cols = [c for c in data.columns if c not in numeric_cols]

        ordered_cols = numeric_cols + categorical_cols

        rows = []
        for i, col in enumerate(ordered_cols):
            s = data[col]
            non_null = int(s.notna().sum())
            missing = int(s.isna().sum())

            row = {
                "#": i,
                "Column": col,
                "Count": non_null,
                "Missing": missing,
                "Mean": "",
                "Std": "",
                "Min": "",
                "25%": "",
                "50%": "",
                "75%": "",
                "Max": "",
                "Unique": "",
                "Top": "",
                "Freq": "",
                "Dtype": str(s.dtype),
            }

            if is_numeric_dtype(s):
                d = s.describe(percentiles=[0.25, 0.5, 0.75])
                row.update(
                    {
                        "Mean": _fmt(d.get("mean")),
                        "Std": _fmt(d.get("std")),
                        "Min": _fmt(d.get("min")),
                        "25%": _fmt(d.get("25%")),
                        "50%": _fmt(d.get("50%")),
                        "75%": _fmt(d.get("75%")),
                        "Max": _fmt(d.get("max")),
                    }
                )
            else:
                # object/category/bool/datetime show top/unique/freq via describe
                d = s.astype("object").describe()
                row.update(
                    {
                        "Unique": _fmt(d.get("unique")),
                        "Top": _fmt(d.get("top")),
                        "Freq": _fmt(d.get("freq")),
                    }
                )

            rows.append(row)

        stats_df = pd.DataFrame(
            rows,
            columns=[
                "#",
                "Column",
                "Count",
                "Missing",
                "Mean",
                "Std",
                "Min",
                "25%",
                "50%",
                "75%",
                "Max",
                "Unique",
                "Top",
                "Freq",
                "Dtype",
            ],
        )

        headers = stats_df.columns.tolist()
        table_data = stats_df.astype(str).to_numpy().tolist()

        col_widths = EDA_Analysis._table_col_widths(headers, table_data, min_col=0.03, max_col=0.22)
        fig_w, fig_h = EDA_Analysis._table_figsize(len(table_data), len(headers), col_width=0.95, min_w=9.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")

        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            colWidths=col_widths,
        )

        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.25)

        # Header styling
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor(COLORS["primary"])
            cell.set_text_props(weight="bold", color="white")

        # Body styling
        for r in range(1, len(table_data) + 1):
            for c in range(len(headers)):
                cell = table[(r, c)]
                cell.set_facecolor("white")
                cell.set_edgecolor("#CCCCCC")

        title_text = (
            "Descriptive Statistics\n"
            f"Rows: {n_rows} | Columns: {n_cols}"
        )
        ax.set_title(title_text, fontsize=9, fontweight="bold", pad=20, loc="left")

        fig.tight_layout()
        return fig

    @staticmethod
    def missing_plot(data):
        """
        Generate missing data visualization.

        Args:
            data (DataFrame): Input dataframe

        Returns:
            Figure: Missing data bar plot
        """
        fig, ax = plt.subplots(figsize=(7, 4))
        msno.bar(data, ax=ax, color=COLORS['primary'], fontsize=9)
        ax.set_title('Missing Data Analysis', fontsize=13, fontweight='bold', pad=12)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def hist_plot(data, variables=None, bins=30, combined=False):
        """
        Generate histogram plots for distribution analysis.

        Args:
            data (DataFrame): Input dataframe
            variables (list, optional): Specific variables to plot. Defaults to all numeric columns.
            bins (int, optional): Number of bins for histogram. Defaults to 30.
            combined (bool, optional): If True, combine all plots into one figure. Defaults to False.

        Returns:
            Figure or list: Single figure if combined=True, list of figures otherwise

        Raises:
            TypeError: If variables is not a list
        """
        cols = [c for c in EDA_Analysis._get_numeric_columns(data, variables) if not data[c].isnull().all()]
        if not cols:
            return [] if not combined else None

        from matplotlib.lines import Line2D

        def _plot_one(ax, series, title, *, show_density_label: bool):
            col_data = series.dropna()
            if col_data.empty:
                ax.set_visible(False)
                return

            mean_val = col_data.mean()
            median_val = col_data.median()

            ax.hist(
                col_data,
                bins=bins,
                color=COLORS['hist_fill'],
                edgecolor='white',
                alpha=0.8,
                linewidth=0.4,
            )

            ax2 = ax.twinx()
            col_data.plot(kind='kde', ax=ax2, color=COLORS['danger'], linewidth=1.2, alpha=0.7)
            ax2.set_ylabel('Density' if show_density_label else '')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_yticks([])

            ax.axvline(mean_val, color=COLORS['danger'], linestyle='--', linewidth=1.0, alpha=0.7,
                       label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color=COLORS['success'], linestyle='--', linewidth=1.0, alpha=0.7,
                       label=f'Median: {median_val:.1f}')

            kde_line = Line2D([0], [0], color=COLORS['danger'], linewidth=1.2, alpha=0.7, label='KDE')
            handles, _ = ax.get_legend_handles_labels()
            handles.append(kde_line)

            ax.text(0.98, 0.98, f"Skew: {col_data.skew():.2f}", transform=ax.transAxes,
                    ha='right', va='top', fontsize=8)
            ax.legend(handles=handles, loc='center right', fontsize=8, framealpha=0.9)

            EDA_Analysis._set_plot_style(ax, title=title, xlabel='', ylabel='Frequency')
            ax.tick_params(labelsize=9)

        if combined:
            n_plots = len(cols)
            grid_cols = min(2, n_plots)
            grid_rows = int(np.ceil(n_plots / grid_cols))
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(10, 3 * grid_rows))
            axes = np.atleast_1d(axes).ravel()

            for idx, col in enumerate(cols):
                try:
                    _plot_one(axes[idx], data[col], col, show_density_label=False)
                except (TypeError, ValueError):
                    axes[idx].set_visible(False)

            for idx in range(len(cols), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle('Distribution Analysis', fontsize=14, fontweight='bold', y=1.00)
            fig.tight_layout()
            return fig

        figs = []
        for col in cols:
            try:
                fig, ax = plt.subplots(figsize=(8, 2.5))
                _plot_one(ax, data[col], f'Distribution: {col}', show_density_label=True)
                fig.tight_layout()
                figs.append(fig)
            except (TypeError, ValueError):
                continue

        return figs


