#!/usr/bin/env python3
"""
Multi-species τ-power trend comparison with CI shading.

This script analyzes τ-band power trends across all available fungal species,
computing confidence intervals and creating comparative visualizations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import glob
import datetime as _dt

# Import our custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_tau_data(species_dir: str) -> Optional[pd.DataFrame]:
    """
    Load τ-band timeseries data from a species analysis directory.

    Args:
        species_dir: Path to species analysis directory

    Returns:
        DataFrame with tau power data or None if not found
    """
    csv_path = os.path.join(species_dir, 'tau_band_timeseries.csv')
    if not os.path.exists(csv_path):
        return None

    try:
        # Read the file and skip comment lines (starting with #)
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        # Find the first non-comment line (should be the header)
        header_idx = 0
        for i, line in enumerate(lines):
            if not line.strip().startswith('#'):
                header_idx = i
                break

        # Read CSV starting from the header line
        df = pd.read_csv(csv_path, skiprows=header_idx)
        # Handle different column naming conventions
        time_col = 'time_s' if 'time_s' in df.columns else 'Time'
        if time_col not in df.columns:
            print(f"Warning: No time column found in {csv_path}")
            return None

        # Find tau columns (handle different formats)
        tau_cols = []
        for col in df.columns:
            if col.startswith('tau_') or col.startswith('τ'):
                # Extract numeric value
                if '_' in col:
                    parts = col.split('_')
                    try:
                        tau_val = float(parts[1])
                        tau_cols.append((col, tau_val))
                    except ValueError:
                        continue

        if not tau_cols:
            print(f"Warning: No tau columns found in {csv_path}")
            return None

        # Create standardized dataframe
        result_df = df[[time_col]].copy()
        result_df.rename(columns={time_col: 'time_s'}, inplace=True)

        for col_name, tau_val in tau_cols:
            result_df[f'tau_{tau_val:g}'] = df[col_name]

        return result_df

    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def compute_bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci_level: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals.

    Args:
        data: Array of values
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(data) < 2:
        return (np.nan, np.nan)

    bootstraps = []
    n_samples = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample = data[indices]
        bootstraps.append(np.mean(sample))

    bootstraps = np.array(bootstraps)
    alpha = (1 - ci_level) / 2

    lower_bound = np.percentile(bootstraps, alpha * 100)
    upper_bound = np.percentile(bootstraps, (1 - alpha) * 100)

    return (lower_bound, upper_bound)

def analyze_tau_trends(species_data: Dict[str, pd.DataFrame], tau_values: List[float]) -> Dict:
    """
    Analyze τ-power trends across species with confidence intervals.

    Args:
        species_data: Dictionary mapping species names to their dataframes
        tau_values: List of tau values to analyze

    Returns:
        Analysis results dictionary
    """
    results = {
        'species': {},
        'tau_analysis': {},
        'comparative_stats': {},
        'metadata': {
            'analysis_date': _dt.datetime.now().isoformat(),
            'n_species': len(species_data),
            'tau_values': tau_values,
            'ci_method': 'bootstrap',
            'n_bootstrap': 1000
        }
    }

    # Analyze each tau value across species
    for tau_val in tau_values:
        tau_key = f'tau_{tau_val:g}'
        tau_results = {
            'species_means': {},
            'species_cis': {},
            'comparative_stats': {}
        }

        species_values = []
        species_names = []

        # Collect data for this tau across all species
        for species_name, df in species_data.items():
            if tau_key in df.columns:
                values = df[tau_key].dropna().values
                if len(values) > 0:
                    species_values.append(values)
                    species_names.append(species_name)

                    # Compute mean and CI for this species
                    mean_val = np.mean(values)
                    ci_lower, ci_upper = compute_bootstrap_ci(values)

                    tau_results['species_means'][species_name] = float(mean_val)
                    tau_results['species_cis'][species_name] = {
                        'lower': float(ci_lower),
                        'upper': float(ci_upper),
                        'mean': float(mean_val)
                    }

        # Perform comparative statistics
        if len(species_values) >= 2:
            # One-way ANOVA
            try:
                f_stat, p_val = stats.f_oneway(*species_values)
                tau_results['comparative_stats'] = {
                    'anova_f': float(f_stat),
                    'anova_p': float(p_val),
                    'significant': p_val < 0.05,
                    'n_species_compared': len(species_values)
                }
            except Exception as e:
                print(f"ANOVA failed for tau={tau_val}: {e}")
                tau_results['comparative_stats'] = {'error': str(e)}

        results['tau_analysis'][tau_key] = tau_results

    # Overall species statistics
    for species_name, df in species_data.items():
        species_stats = {
            'n_samples': len(df),
            'time_range': [float(df['time_s'].min()), float(df['time_s'].max())],
            'tau_columns': [col for col in df.columns if col.startswith('tau_')]
        }
        results['species'][species_name] = species_stats

    return results

def create_comparison_plot(results: Dict, output_path: str):
    """
    Create a comparative plot of τ-power trends across species.

    Args:
        results: Analysis results dictionary
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, len(results['tau_analysis']), figsize=(15, 6))
    if len(results['tau_analysis']) == 1:
        axes = [axes]

    tau_values = list(results['tau_analysis'].keys())

    for i, tau_key in enumerate(tau_values):
        ax = axes[i]
        tau_data = results['tau_analysis'][tau_key]

        species_names = list(tau_data['species_means'].keys())
        means = [tau_data['species_means'][name] for name in species_names]
        ci_data = [tau_data['species_cis'][name] for name in species_names]

        # Plot means with error bars
        x_pos = np.arange(len(species_names))
        ax.bar(x_pos, means, alpha=0.7, color='skyblue', label='Mean τ-power')

        # Add confidence intervals
        for j, (name, ci) in enumerate(zip(species_names, ci_data)):
            ax.errorbar(j, ci['mean'],
                       yerr=[[ci['mean'] - ci['lower']], [ci['upper'] - ci['mean']]],
                       fmt='none', color='black', capsize=5, linewidth=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(species_names, rotation=45, ha='right')
        ax.set_title(f'τ = {tau_key.replace("tau_", "")}s')
        ax.set_ylabel('Power')
        ax.grid(True, alpha=0.3)

        # Add significance indicator if available
        if 'comparative_stats' in tau_data and tau_data['comparative_stats'].get('significant', False):
            ax.text(0.02, 0.98, '* p < 0.05',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Comparative plot saved: {output_path}")

def main():
    """Main function to run multi-species τ-trend analysis."""
    print("🧬 Multi-Species τ-Power Trend Analysis")
    print("=" * 50)

    # Configuration
    results_base = '/home/kronos/mushroooom/results/zenodo'
    tau_values = [5.5, 24.5, 104.0]  # Standard tau values
    output_dir = f'/home/kronos/mushroooom/results/zenodo/_composites/{_dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'

    os.makedirs(output_dir, exist_ok=True)

    # Find all species with recent analysis
    print("🔍 Finding species data...")
    species_data = {}

    # Look for species directories
    for item in os.listdir(results_base):
        species_path = os.path.join(results_base, item)
        if os.path.isdir(species_path) and not item.startswith('_'):
            # Find the most recent analysis directory
            subdirs = [d for d in os.listdir(species_path) if os.path.isdir(os.path.join(species_path, d))]
            if subdirs:
                # Sort by timestamp and take the most recent
                subdirs.sort(reverse=True)
                latest_dir = os.path.join(species_path, subdirs[0])

                df = load_tau_data(latest_dir)
                if df is not None:
                    species_data[item] = df
                    print(f"✅ Loaded {item}: {len(df)} samples, {len([c for c in df.columns if c.startswith('tau_')])} tau columns")

    if not species_data:
        print("❌ No species data found!")
        return

    print(f"\n📊 Analyzing {len(species_data)} species...")

    # Perform analysis
    results = analyze_tau_trends(species_data, tau_values)

    # Save results
    json_path = os.path.join(output_dir, 'multispecies_tau_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Analysis results saved: {json_path}")

    # Create comparative plot
    plot_path = os.path.join(output_dir, 'multispecies_tau_comparison.png')
    create_comparison_plot(results, plot_path)

    # Generate summary report
    report_path = os.path.join(output_dir, 'multispecies_analysis_report.md')

    with open(report_path, 'w') as f:
        f.write("# Multi-Species τ-Power Trend Analysis Report\n\n")
        f.write(f"**Analysis Date:** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Species Analyzed:** {len(species_data)}\n\n")

        f.write("## Species Summary\n\n")
        for species, stats in results['species'].items():
            f.write(f"- **{species}:** {stats['n_samples']} samples, {len(stats['tau_columns'])} τ values\n")

        f.write("\n## Comparative Statistics\n\n")
        for tau_key, tau_data in results['tau_analysis'].items():
            tau_val = tau_key.replace('tau_', '')
            f.write(f"### τ = {tau_val}s\n\n")

            if 'comparative_stats' in tau_data:
                comp_stats = tau_data['comparative_stats']
                if 'anova_f' in comp_stats:
                    f.write(f"- **ANOVA F-statistic:** {comp_stats['anova_f']:.3f}\n")
                    f.write(f"- **p-value:** {comp_stats['anova_p']:.6f}\n")
                    f.write(f"- **Significant difference:** {'Yes' if comp_stats.get('significant', False) else 'No'}\n")

            f.write("\n**Species comparison:**\n")
            for species, ci_data in tau_data['species_cis'].items():
                f.write(f"- **{species}:** {ci_data['mean']:.4f} [{ci_data['lower']:.4f}, {ci_data['upper']:.4f}]\n")
            f.write("\n")

        f.write("\n## Files Generated\n\n")
        f.write(f"- `multispecies_tau_analysis.json` - Complete analysis results\n")
        f.write(f"- `multispecies_tau_comparison.png` - Comparative visualization\n")
        f.write(f"- `multispecies_analysis_report.md` - This summary report\n")

    print(f"✅ Summary report saved: {report_path}")

    # Print key findings
    print("\n🎯 Key Findings:")
    print(f"   • Analyzed {len(species_data)} fungal species")
    print(f"   • Compared τ-power across {len(tau_values)} time scales")
    print("   • Generated comparative statistics and visualizations")

    # Highlight significant differences
    significant_findings = []
    for tau_key, tau_data in results['tau_analysis'].items():
        if 'comparative_stats' in tau_data:
            comp_stats = tau_data['comparative_stats']
            if comp_stats.get('significant', False):
                tau_val = tau_key.replace('tau_', '')
                significant_findings.append(f"τ={tau_val}s (p={comp_stats['anova_p']:.3f})")

    if significant_findings:
        print(f"   • Significant inter-species differences found for: {', '.join(significant_findings)}")
    else:
        print("   • No significant inter-species differences detected")

    print(f"\n📁 Results saved to: {output_dir}")

if __name__ == '__main__':
    main()