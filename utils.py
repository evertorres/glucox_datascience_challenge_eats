import pandas as pd
import json
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency

def unpack_json_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Unpacks a DataFrame column containing dictionaries or JSON-formatted strings.
    Expands JSON keys into new DataFrame columns.
    
    Args:
        df (pd.DataFrame): Original DataFrame.
        column_name (str): Name of the column containing JSON/dict data.
        
    Returns:
        pd.DataFrame: DataFrame with new columns added.
                      If a key does not exist in a row, the value will be NaN.
    """
    
    def parse_item(item):
        # If it is already a dictionary, return it as is
        if isinstance(item, dict):
            return item
        # If NaN or None, return empty dictionary
        if pd.isna(item):
            return {}
        # If string, try to parse
        if isinstance(item, str):
            try:
                # Try standard JSON (double quotes)
                return json.loads(item)
            except json.JSONDecodeError:
                try:
                    # Try Python dictionary syntax (single quotes)
                    return ast.literal_eval(item)
                except (ValueError, SyntaxError):
                    return {}
        return {}

    # Extract and normalize data
    # Use tolist() because json_normalize is much faster with lists of dicts than apply(pd.Series)
    normalized_data = pd.json_normalize(df[column_name].apply(parse_item).tolist())
    
    # Ensure index matches original for correct concatenation
    normalized_data.index = df.index
    
    # Concatenate new columns to original DataFrame
    return pd.concat([df, normalized_data], axis=1)

def generate_nudges_measurement_df(df: pd.DataFrame) -> pd.DataFrame:
    df_nudges = df[df['event_type'] == 'nudge_sent'].drop(columns='glucose_value')
    df_nudges = df_nudges.sort_values(by=['patient_id', 'timestamp'])


    df_meas = df[df['event_type'] == 'measurement_logged'].drop(columns='nudge_type')
    df_meas = df_meas.sort_values(by=['patient_id', 'timestamp'])
    
    return df_nudges, df_meas

def add_cumulative_counts(df: pd.DataFrame, group_col: str, type_col: str, prefix: str = 'cumulative') -> pd.DataFrame:
    """
    Generates cumulative count columns for each unique category in type_col.

    Args:
        df (pd.DataFrame): Input DataFrame (must be chronologically ordered).
        group_col (str): Column to group by (e.g., 'patient_id').
        type_col (str): Categorical column to count (e.g., 'nudge_type').
        prefix (str): Prefix for the newly generated columns.

    Returns:
        pd.DataFrame: Original DataFrame with the cumulative columns added.
    """
    # 1. One-hot encode the type column
    dummies = pd.get_dummies(df[type_col], prefix=prefix)
    
    # 2. Calculate cumulative sum grouped by group_col (patient)
    # Temporarily join the group ID to perform the groupby operation
    cumulative = pd.concat([df[[group_col]], dummies], axis=1).groupby(group_col)[dummies.columns].cumsum()
    
    # Add column with the total cumulative sum
    cumulative['cumulative_all'] = cumulative.sum(axis=1)
    
    # 3. Concatenate the calculated columns to the original DataFrame
    return pd.concat([df, cumulative], axis=1)

def mark_responses(df_target: pd.DataFrame, df_source: pd.DataFrame, target_col: str = 'event_id', source_col: str = None, new_column: str = 'has_response') -> pd.DataFrame:
    """
    Marks with 1 in df_target if the target_col value exists in the source_col of df_source.

    Args:
        df_target (pd.DataFrame): Main DataFrame (e.g., df_nudges) where the mark will be created.
        df_source (pd.DataFrame): Reference DataFrame (e.g., pd_asof) containing the successful IDs.
        target_col (str): Column name in df_target used for matching.
        source_col (str, optional): Column name in df_source. If None, target_col is used.
        new_column (str): Name of the new binary column.

    Returns:
        pd.DataFrame: The target DataFrame with the new column added.
    """
    if source_col is None:
        source_col = target_col
        
    # Use isin(), which is vectorized and efficient for membership testing
    df_target[new_column] = df_target[target_col].isin(df_source[source_col]).astype(int)
    return df_target

### Reporting

def mean_responses(df: pd.DataFrame, groupby: str) -> pd.DataFrame:
    df_mean_response = df[df['nudge_type'] == groupby]
    return df_mean_response.groupby(by=['cumulative_'+ groupby, 'age_group', 'risk_segment'])[['has_response']].mean()

def wilson_ci(n_success: float, n_total:float, confidence: float = 0.95) -> tuple:
    
    if n_total == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_success / n_total
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))) / denom
    return center - margin, center + margin

def compute_global_ci(sub: pd.DataFrame, cum_col: str, max_nudge: int=30) -> dict:
    result = {}
    for nudge_n in range(1, max_nudge + 1):
        grp = sub[sub[cum_col] == nudge_n]["has_response"]
        n = len(grp)
        if n == 0:
            continue
        lo, hi = wilson_ci(grp.sum(), n)
        result[nudge_n] = {"n": n, "lower": lo, "upper": hi, "rate": grp.sum() / n}
    return result

def compute_group_rates(sub: pd.DataFrame, cum_col: str, group_col: str, max_nudge: int=30) -> dict:
    results = {}
    for group_val, gdf in sub.groupby(group_col):
        positions, rates = [], []
        for nudge_n in range(1, max_nudge + 1):
            grp = gdf[gdf[cum_col] == nudge_n]["has_response"]
            if len(grp) == 0:
                continue
            positions.append(nudge_n)
            rates.append(grp.sum() / len(grp))
        results[group_val] = {
            "positions": np.array(positions),
            "rates": np.array(rates),
        }
    return results

def plot_nudge_response(
    df: pd.DataFrame,
    group_col: str = None,           # None → global plot without subgroups
    palette: list = None,            # Custom color list (optional)
    fig_title: str = None,           # Figure title (optional)
    filename: str = None,            # Save path (optional, if None it only shows)
    max_nudge: int = 30,
    confidence: float = 0.95,
) -> None:
    """
    Visualizes the response rate (has_response) by cumulative nudge.

    Parameters
    ----------
    df          : DataFrame with columns nudge_type, cumulative__Gentle_Reminder,
                  cumulative__Urgent_Alert, has_response and optionally group_col.
    group_col   : Segmentation column (e.g., 'age_group', 'risk_segment').
                  If None, generates a global plot without subgroups.
    palette     : List of hex colors for subgroups. If None, uses default palette.
    fig_title   : Figure title. If None, it is automatically generated.
    filename    : Output file path (.png). If None, it only displays the figure.
    max_nudge   : Maximum number of cumulative nudges to display (default 30).
    confidence  : Confidence level for the Wilson CI (default 0.95).
    """

    palette = palette or PALETTES["default"]

    # Automatic title if not provided
    if fig_title is None:
        if group_col:
            fig_title = f"Response Rate by {group_col} and Cumulative Nudge"
        else:
            fig_title = "Global Response Rate by Cumulative Nudge"

    fig, axes = plt.subplots(1, 2, figsize=(22, 8), facecolor="#0d0d0d")
    fig.suptitle(fig_title, fontsize=15, color="white", fontweight="bold", y=1.01)

    for cfg, ax in zip(NUDGE_CONFIGS, axes):
        sub = df[df["nudge_type"] == cfg["subset_val"]].copy()
        global_ci = compute_global_ci(sub, cfg["cum_col"], max_nudge)

        ci_positions = np.array(list(global_ci.keys()))
        ci_lower     = np.array([v["lower"] for v in global_ci.values()])
        ci_upper     = np.array([v["upper"] for v in global_ci.values()])
        global_rates = np.array([v["rate"]  for v in global_ci.values()])
        ns           = np.array([v["n"]     for v in global_ci.values()])

        ax.set_facecolor("#1a1a1a")

        # ── Global CI Band (always present) ──
        ax.fill_between(ci_positions, ci_lower, ci_upper,
                        alpha=0.15, color="white", label="Global 95% CI (Wilson)")

        if group_col:
            # ── Subgroup mode: line per category ──
            group_rates = compute_group_rates(sub, cfg["cum_col"], group_col, max_nudge)
            for i, (group_val, data) in enumerate(sorted(group_rates.items())):
                color = palette[i % len(palette)]
                ax.plot(data["positions"], data["rates"],
                        color=color, linewidth=2,
                        marker="o", markersize=4, label=str(group_val))
        else:
            # ── Global mode: total rate line ──
            ax.plot(ci_positions, global_rates,
                    color=cfg["panel_color"], linewidth=2.5,
                    marker="o", markersize=5, label="Global Rate")

        # 50% Reference
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1, label="50%")

        # ── N on secondary axis ──
        ax2 = ax.twinx()
        bar_color = palette[0] if group_col else cfg["panel_color"]
        ax2.bar(ci_positions, ns, color=bar_color, alpha=0.12, width=0.6, label="Total n")
        ax2.set_ylabel("n observations (total)", color="#888888", fontsize=10)
        ax2.tick_params(colors="#888888")
        ax2.spines[:].set_color("#333333")
        ax2.set_ylim(0, ns.max() * 4)

        # ── Aesthetics ──
        ax.set_xlim(0.5, max_nudge + 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(1, max_nudge + 1))
        ax.set_xticklabels(range(1, max_nudge + 1), fontsize=8, color="#cccccc", rotation=45)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_yticklabels([f"{v:.0%}" for v in np.arange(0, 1.1, 0.1)], fontsize=9, color="#cccccc")
        ax.set_xlabel("Cumulative Nudges", color="#aaaaaa", fontsize=11)
        ax.set_ylabel("Response Rate", color="#aaaaaa", fontsize=11)
        ax.set_title(cfg["title"], color=cfg["panel_color"], fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(colors="#cccccc")
        ax.spines[:].set_color("#333333")
        ax.grid(axis="y", color="#2a2a2a", linestyle="--", linewidth=0.7)
        ax.grid(axis="x", color="#2a2a2a", linestyle=":", linewidth=0.5)

        legend_title = group_col if group_col else None
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  facecolor="#222222", edgecolor="#444444",
                  labelcolor="white", fontsize=9,
                  loc="upper right", title=legend_title, title_fontsize=9)

    plt.tight_layout()
    if filename:
        plt.savefig('report/'+ filename, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
        print(f"Saved figure: {'report/'+ filename}")  


# Statistical Test

def chi_test_per_number_of_nudges(df_nudges: pd.DataFrame) -> str:
    
    # 1. Define control parameters
    MIN_OBSERVATIONS = 30  # Standard threshold for rely on Chi-Square
    max_nudge_to_evaluate = 30  # High limit, the code will stop itself based on the threshold
    output = []

    print(f"Searching for the breakpoint vs. Nudge 1 (Threshold: Min {MIN_OBSERVATIONS} patients per nudge)\n")
    print("=" * 65)

    # Iterate from nudge 2 onwards
    for i in range(2, max_nudge_to_evaluate + 1):
        
        # 2. Calculate how many people received this specific Nudge
        n_observations_i = len(df_nudges[df_nudges['cumulative_all'] == i])
        
        # 3. The Stop Rule: If it falls below the threshold, halt the analysis
        if n_observations_i < MIN_OBSERVATIONS:
            print(f"\nANALYSIS STOPPED AT NUDGE {i}")
            output.append(f"\nANALYSIS STOPPED AT NUDGE {i}")
            print(f"   Reason: Only {n_observations_i} observations remaining (Below the threshold of {MIN_OBSERVATIONS}).")
            output.append(f"   Reason: Only {n_observations_i} observations remaining (Below the threshold of {MIN_OBSERVATIONS}).")
            print("   From here on, the risk of false positives/negatives is too high.")
            output.append("   From here on, the risk of false positives/negatives is too high.")
            break # Breaks the loop to avoid unnecessary or statistically weak calculations
            
        # 4. If there's enough data, proceed with the usual filtering
        df_filtered = df_nudges[df_nudges['cumulative_all'].isin([1, i])]
        
        # Create the contingency table
        table = pd.crosstab(df_filtered['cumulative_all'], df_filtered['has_response'])
        
        # Validate that we have a 2x2 matrix
        if table.shape == (2, 2):
            chi2, p_value, dof, expected = chi2_contingency(table)
            
            # Calculate response rates (%) for business context
            rate_nudge_1 = (table.loc[1, 1] / table.loc[1].sum()) * 100
            rate_nudge_i = (table.loc[i, 1] / table.loc[i].sum()) * 100
            
            # Format the conclusion
            if p_value < 0.05:
                result = "YES (Significant difference)"
            else:
                result = "No (No significant difference)"
                
            # Print results adding the N of the evaluated nudge
            print(f"Comparing Nudge 1 vs Nudge {i} (Available sample N={n_observations_i})")
            print(f"  - Response Rate: Nudge 1 ({rate_nudge_1:.1f}%) vs Nudge {i} ({rate_nudge_i:.1f}%)")
            print(f"  - P-value: {p_value:.4f} -> {result}")
            print("-" * 65)
            output.append(f"Comparing Nudge 1 vs Nudge {i} (Available sample N={n_observations_i})")
            output.append(f"  - Response Rate: Nudge 1 ({rate_nudge_1:.1f}%) vs Nudge {i} ({rate_nudge_i:.1f}%)")
            output.append(f"  - P-value: {p_value:.4f} -> {result}")
            output.append("-" * 65)

    return "\n".join(output)

# UI

PALETTES = {
    "default": [
        "#4fc3f7", "#81c784", "#ffb74d", "#f06292",
        "#ba68c8", "#4db6ac", "#ff8a65", "#90a4ae"
    ]
}

NUDGE_CONFIGS = [
    {
        "subset_val": "Gentle_Reminder",
        "cum_col": "cumulative_Gentle_Reminder",
        "title": "Gentle Reminder",
        "panel_color": "#4fc3f7",
    },
    {
        "subset_val": "Urgent_Alert",
        "cum_col": "cumulative_Urgent_Alert",
        "title": "Urgent Alert",
        "panel_color": "#ff7043",
    },
]
