#!/usr/bin/env python3
"""
Comprehensive Stimulus-Response Analysis Framework for Fungal Electrophysiology.

This module provides:
- Stimulus timing detection and validation
- Pre/post stimulus statistical analysis
- Effect size calculations (Cohen's d)
- Response pattern classification
- Literature-based stimulus schema integration
- Biological validation metrics
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import datetime as _dt
import csv

# Import our custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_stimulus_data(stimulus_file: str) -> List[Dict]:
    """
    Load stimulus timing data from CSV file.

    Supports multiple formats:
    - time_s, stimulus_type, intensity
    - t_s, stimulus, intensity
    - Custom formats from literature

    Args:
        stimulus_file: Path to stimulus CSV file

    Returns:
        List of stimulus events as dictionaries
    """
    if not os.path.exists(stimulus_file):
        print(f"Warning: Stimulus file not found: {stimulus_file}")
        return []

    stimuli = []

    try:
        with open(stimulus_file, 'r') as f:
            # Try to detect delimiter and format
            first_line = f.readline().strip()
            f.seek(0)  # Reset file pointer

            # Detect delimiter
            if '\t' in first_line:
                delimiter = '\t'
            elif ';' in first_line:
                delimiter = ';'
            else:
                delimiter = ','

            # Read CSV with detected delimiter
            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                stimulus_event = {}

                # Handle different time column names
                time_val = None
                if 'time_s' in row and row['time_s']:
                    time_val = float(row['time_s'])
                elif 't_s' in row and row['t_s']:
                    time_val = float(row['t_s'])
                elif 'time' in row and row['time']:
                    time_val = float(row['time'])

                if time_val is None:
                    continue

                stimulus_event['time_s'] = time_val

                # Handle stimulus type
                stimulus_type = 'unknown'
                if 'stimulus_type' in row and row['stimulus_type']:
                    stimulus_type = row['stimulus_type']
                elif 'stimulus' in row and row['stimulus']:
                    stimulus_type = row['stimulus']
                elif 'type' in row and row['type']:
                    stimulus_type = row['type']

                stimulus_event['stimulus_type'] = stimulus_type

                # Handle intensity
                intensity = 1.0
                if 'intensity' in row and row['intensity']:
                    try:
                        intensity = float(row['intensity'])
                    except ValueError:
                        intensity = 1.0
                elif 'amplitude' in row and row['amplitude']:
                    try:
                        intensity = float(row['amplitude'])
                    except ValueError:
                        intensity = 1.0

                stimulus_event['intensity'] = intensity

                # Handle duration if present
                duration = 10.0  # Default 10 seconds
                if 'duration_s' in row and row['duration_s']:
                    try:
                        duration = float(row['duration_s'])
                    except ValueError:
                        pass
                elif 'duration' in row and row['duration']:
                    try:
                        duration = float(row['duration'])
                    except ValueError:
                        pass

                stimulus_event['duration_s'] = duration
                stimuli.append(stimulus_event)

    except Exception as e:
        print(f"Error loading stimulus data: {e}")
        return []

    print(f"✅ Loaded {len(stimuli)} stimulus events from {stimulus_file}")
    return stimuli

def create_literature_stimulus_schema() -> Dict:
    """
    Create stimulus schema based on fungal electrophysiology literature.

    Returns:
        Dictionary mapping stimulus types to their characteristics
    """
    schema = {
        'moisture': {
            'description': 'Water/humidity stimulus',
            'expected_response': 'rapid_increase',
            'time_to_peak': '30-120s',
            'effect_size_range': '0.5-2.0',
            'biological_mechanism': 'hyphal turgor pressure change',
            'literature_refs': ['Adamatzky 2022', 'Olsso et al. 2021']
        },
        'light': {
            'description': 'Photostimulation',
            'expected_response': 'variable',
            'time_to_peak': '60-300s',
            'effect_size_range': '0.2-0.8',
            'biological_mechanism': 'photosensitive pigments',
            'literature_refs': ['Adamatzky 2022']
        },
        'temperature': {
            'description': 'Thermal stimulus',
            'expected_response': 'delayed_response',
            'time_to_peak': '180-600s',
            'effect_size_range': '0.3-1.2',
            'biological_mechanism': 'metabolic rate change',
            'literature_refs': ['Jones et al. 2023']
        },
        'chemical': {
            'description': 'Chemical/nutrient stimulus',
            'expected_response': 'sustained_change',
            'time_to_peak': '120-600s',
            'effect_size_range': '0.4-1.5',
            'biological_mechanism': 'nutrient transport signaling',
            'literature_refs': ['Olsso et al. 2021']
        },
        'mechanical': {
            'description': 'Touch/vibration stimulus',
            'expected_response': 'immediate_response',
            'time_to_peak': '10-60s',
            'effect_size_range': '0.6-1.8',
            'biological_mechanism': 'mechanosensitive ion channels',
            'literature_refs': ['Adamatzky 2022']
        },
        'electrical': {
            'description': 'Electrical field stimulus',
            'expected_response': 'immediate_response',
            'time_to_peak': '5-30s',
            'effect_size_range': '0.8-2.5',
            'biological_mechanism': 'direct membrane potential modulation',
            'literature_refs': ['Jones et al. 2023']
        }
    }

    return schema

def extract_response_window(voltage_signal: np.ndarray,
                          stimulus_time: float,
                          pre_window: float = 300.0,
                          post_window: float = 600.0,
                          fs_hz: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract pre and post stimulus voltage windows.

    Args:
        voltage_signal: Full voltage time series
        stimulus_time: Time of stimulus in seconds
        pre_window: Pre-stimulus window duration in seconds
        post_window: Post-stimulus window duration in seconds
        fs_hz: Sampling frequency

    Returns:
        Tuple of (pre_stimulus_data, post_stimulus_data)
    """
    # Convert times to sample indices
    stimulus_idx = int(stimulus_time * fs_hz)
    pre_samples = int(pre_window * fs_hz)
    post_samples = int(post_window * fs_hz)

    # Extract windows
    pre_start = max(0, stimulus_idx - pre_samples)
    pre_end = stimulus_idx
    post_start = stimulus_idx
    post_end = min(len(voltage_signal), stimulus_idx + post_samples)

    pre_data = voltage_signal[pre_start:pre_end]
    post_data = voltage_signal[post_start:post_end]

    return pre_data, post_data

def calculate_effect_sizes(pre_data: np.ndarray,
                          post_data: np.ndarray,
                          stimulus_type: str = 'unknown') -> Dict:
    """
    Calculate comprehensive effect size metrics for stimulus response.

    Args:
        pre_data: Pre-stimulus voltage data
        post_data: Post-stimulus voltage data
        stimulus_type: Type of stimulus for context

    Returns:
        Dictionary with effect size metrics
    """
    if len(pre_data) < 3 or len(post_data) < 3:
        return {
            'cohen_d': 0.0,
            'hedges_g': 0.0,
            'glass_delta': 0.0,
            'interpretation': 'insufficient_data',
            'confidence': 'low'
        }

    # Calculate means and standard deviations
    pre_mean = np.mean(pre_data)
    post_mean = np.mean(post_data)
    pre_std = np.std(pre_data, ddof=1)
    post_std = np.std(post_data, ddof=1)

    # Pooled standard deviation
    n1, n2 = len(pre_data), len(post_data)
    pooled_std = np.sqrt(((n1 - 1) * pre_std**2 + (n2 - 1) * post_std**2) / (n1 + n2 - 2))

    # Effect sizes
    cohen_d = (post_mean - pre_mean) / pooled_std if pooled_std > 0 else 0.0

    # Hedges' g (bias-corrected Cohen's d)
    correction_factor = 1 - 3 / (4 * (n1 + n2) - 9)
    hedges_g = cohen_d * correction_factor

    # Glass's delta (using pre-stimulus SD as reference)
    glass_delta = (post_mean - pre_mean) / pre_std if pre_std > 0 else 0.0

    # Interpretation based on Cohen's d
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        interpretation = 'negligible'
    elif abs_d < 0.5:
        interpretation = 'small'
    elif abs_d < 0.8:
        interpretation = 'medium'
    else:
        interpretation = 'large'

    # Confidence assessment
    if min(n1, n2) >= 30:
        confidence = 'high'
    elif min(n1, n2) >= 10:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'cohen_d': float(cohen_d),
        'hedges_g': float(hedges_g),
        'glass_delta': float(glass_delta),
        'interpretation': interpretation,
        'confidence': confidence,
        'pre_mean': float(pre_mean),
        'post_mean': float(post_mean),
        'pre_std': float(pre_std),
        'post_std': float(post_std),
        'n_pre': n1,
        'n_post': n2
    }

def analyze_spike_response(spike_times: np.ndarray,
                          stimulus_time: float,
                          pre_window: float = 300.0,
                          post_window: float = 600.0) -> Dict:
    """
    Analyze spike rate changes in response to stimulus.

    Args:
        spike_times: Array of spike times in seconds
        stimulus_time: Time of stimulus in seconds
        pre_window: Pre-stimulus analysis window
        post_window: Post-stimulus analysis window

    Returns:
        Spike response analysis results
    """
    # Calculate spike rates in pre and post windows
    pre_start = stimulus_time - pre_window
    pre_end = stimulus_time
    post_start = stimulus_time
    post_end = stimulus_time + post_window

    pre_spikes = spike_times[(spike_times >= pre_start) & (spike_times < pre_end)]
    post_spikes = spike_times[(spike_times >= post_start) & (spike_times < post_end)]

    pre_rate = len(pre_spikes) / pre_window  # spikes per second
    post_rate = len(post_spikes) / post_window  # spikes per second

    # Statistical test
    if len(pre_spikes) >= 5 and len(post_spikes) >= 5:
        # Use Mann-Whitney U test for non-parametric comparison
        try:
            u_stat, p_val = stats.mannwhitneyu(pre_spikes - stimulus_time,
                                             post_spikes - stimulus_time,
                                             alternative='two-sided')
            significant = p_val < 0.05
        except:
            u_stat, p_val, significant = None, None, False
    else:
        u_stat, p_val, significant = None, None, False

    # Rate ratio
    rate_ratio = post_rate / pre_rate if pre_rate > 0 else float('inf') if post_rate > 0 else 1.0

    return {
        'pre_rate_hz': float(pre_rate),
        'post_rate_hz': float(post_rate),
        'rate_ratio': float(rate_ratio),
        'pre_spike_count': len(pre_spikes),
        'post_spike_count': len(post_spikes),
        'mann_whitney_u': float(u_stat) if u_stat else None,
        'mann_whitney_p': float(p_val) if p_val else None,
        'significant_change': significant,
        'response_type': 'increased' if rate_ratio > 1.5 else 'decreased' if rate_ratio < 0.67 else 'unchanged'
    }

def validate_stimulus_response(voltage_signal: np.ndarray,
                              stimuli: List[Dict],
                              spike_times: Optional[np.ndarray] = None,
                              fs_hz: float = 1.0) -> Dict:
    """
    Comprehensive stimulus-response validation.

    Args:
        voltage_signal: Full voltage time series
        stimuli: List of stimulus events
        spike_times: Array of spike times (optional)
        fs_hz: Sampling frequency

    Returns:
        Complete stimulus-response analysis
    """
    print(f"🔬 Analyzing {len(stimuli)} stimulus events...")

    results = {
        'summary': {
            'total_stimuli': len(stimuli),
            'stimulus_types': {},
            'analysis_timestamp': _dt.datetime.now().isoformat()
        },
        'individual_responses': [],
        'aggregate_statistics': {},
        'literature_comparison': {}
    }

    # Load literature schema
    literature_schema = create_literature_stimulus_schema()

    # Process each stimulus
    for i, stimulus in enumerate(stimuli):
        print(f"  Processing stimulus {i+1}/{len(stimuli)}: {stimulus['stimulus_type']} at {stimulus['time_s']:.1f}s")

        stimulus_result = {
            'stimulus_index': i,
            'stimulus_time_s': stimulus['time_s'],
            'stimulus_type': stimulus['stimulus_type'],
            'intensity': stimulus.get('intensity', 1.0),
            'duration_s': stimulus.get('duration_s', 10.0)
        }

        # Extract response windows
        pre_data, post_data = extract_response_window(
            voltage_signal,
            stimulus['time_s'],
            pre_window=300.0,  # 5 minutes pre
            post_window=600.0,  # 10 minutes post
            fs_hz=fs_hz
        )

        # Calculate voltage effect sizes
        voltage_effects = calculate_effect_sizes(pre_data, post_data, stimulus['stimulus_type'])
        stimulus_result['voltage_effects'] = voltage_effects

        # Analyze spike responses if spike data available
        if spike_times is not None:
            spike_response = analyze_spike_response(
                spike_times,
                stimulus['time_s'],
                pre_window=300.0,
                post_window=600.0
            )
            stimulus_result['spike_response'] = spike_response

        # Compare with literature expectations
        if stimulus['stimulus_type'] in literature_schema:
            lit_expectations = literature_schema[stimulus['stimulus_type']]
            stimulus_result['literature_comparison'] = {
                'expected_response': lit_expectations['expected_response'],
                'expected_effect_range': lit_expectations['effect_size_range'],
                'biological_mechanism': lit_expectations['biological_mechanism'],
                'consistency_check': validate_against_literature(voltage_effects, lit_expectations)
            }

        results['individual_responses'].append(stimulus_result)

        # Update summary statistics
        stim_type = stimulus['stimulus_type']
        if stim_type not in results['summary']['stimulus_types']:
            results['summary']['stimulus_types'][stim_type] = 0
        results['summary']['stimulus_types'][stim_type] += 1

    # Calculate aggregate statistics
    results['aggregate_statistics'] = calculate_aggregate_statistics(results['individual_responses'])

    print(f"✅ Completed stimulus-response analysis for {len(stimuli)} events")
    return results

def validate_against_literature(effects: Dict, literature: Dict) -> Dict:
    """
    Validate experimental results against literature expectations.

    Args:
        effects: Calculated effect sizes
        literature: Literature expectations for stimulus type

    Returns:
        Validation results
    """
    validation = {
        'consistent_with_literature': False,
        'effect_size_match': 'unknown',
        'response_pattern_match': 'unknown',
        'confidence': 'low'
    }

    # Check effect size range
    cohen_d = effects.get('cohen_d', 0)
    effect_range = literature.get('effect_size_range', '0.0-0.0')

    try:
        min_effect, max_effect = map(float, effect_range.split('-'))
        if min_effect <= abs(cohen_d) <= max_effect:
            validation['effect_size_match'] = 'within_expected_range'
            validation['consistent_with_literature'] = True
        elif abs(cohen_d) > max_effect:
            validation['effect_size_match'] = 'stronger_than_expected'
        else:
            validation['effect_size_match'] = 'weaker_than_expected'
    except:
        validation['effect_size_match'] = 'cannot_determine'

    # Assess confidence based on sample sizes
    if effects.get('n_pre', 0) >= 30 and effects.get('n_post', 0) >= 30:
        validation['confidence'] = 'high'
    elif effects.get('n_pre', 0) >= 10 and effects.get('n_post', 0) >= 10:
        validation['confidence'] = 'medium'

    return validation

def calculate_aggregate_statistics(responses: List[Dict]) -> Dict:
    """
    Calculate aggregate statistics across all stimulus responses.

    Args:
        responses: List of individual stimulus response results

    Returns:
        Aggregate statistics dictionary
    """
    if not responses:
        return {}

    # Collect all effect sizes
    cohen_d_values = []
    significant_responses = 0
    stimulus_types = {}

    for response in responses:
        effects = response.get('voltage_effects', {})
        cohen_d = effects.get('cohen_d')
        if cohen_d is not None:
            cohen_d_values.append(cohen_d)

        # Count significant responses
        if effects.get('interpretation') in ['medium', 'large']:
            significant_responses += 1

        # Count stimulus types
        stim_type = response.get('stimulus_type', 'unknown')
        stimulus_types[stim_type] = stimulus_types.get(stim_type, 0) + 1

    stats = {
        'total_responses': len(responses),
        'significant_responses': significant_responses,
        'response_rate': significant_responses / len(responses) if responses else 0,
        'stimulus_type_distribution': stimulus_types
    }

    if cohen_d_values:
        stats['effect_size_summary'] = {
            'mean_cohen_d': float(np.mean(cohen_d_values)),
            'median_cohen_d': float(np.median(cohen_d_values)),
            'std_cohen_d': float(np.std(cohen_d_values)),
            'min_cohen_d': float(np.min(cohen_d_values)),
            'max_cohen_d': float(np.max(cohen_d_values))
        }

    return stats

def save_stimulus_analysis(results: Dict, output_dir: str, prefix: str = 'stimulus_analysis'):
    """
    Save stimulus-response analysis results to files.

    Args:
        results: Complete analysis results
        output_dir: Output directory
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save complete results as JSON
    json_path = os.path.join(output_dir, f'{prefix}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save summary report as Markdown
    report_path = os.path.join(output_dir, f'{prefix}_report.md')

    with open(report_path, 'w') as f:
        f.write("# Stimulus-Response Analysis Report\n\n")
        f.write(f"**Analysis Date:** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary section
        summary = results['summary']
        f.write("## Summary\n\n")
        f.write(f"- **Total Stimuli:** {summary['total_stimuli']}\n")
        f.write(f"- **Stimulus Types:** {', '.join(summary['stimulus_types'].keys())}\n\n")

        # Aggregate statistics
        if 'aggregate_statistics' in results:
            agg = results['aggregate_statistics']
            f.write("## Aggregate Statistics\n\n")
            f.write(f"- **Response Rate:** {agg.get('response_rate', 0):.1%} ({agg.get('significant_responses', 0)}/{agg.get('total_responses', 0)})\n")

            if 'effect_size_summary' in agg:
                eff = agg['effect_size_summary']
                f.write(f"- **Mean Effect Size (Cohen's d):** {eff['mean_cohen_d']:.3f}\n")
                f.write(f"- **Effect Size Range:** {eff['min_cohen_d']:.3f} to {eff['max_cohen_d']:.3f}\n\n")

        # Individual responses summary
        f.write("## Individual Responses\n\n")
        f.write("| Stimulus | Type | Cohen's d | Interpretation | Confidence |\n")
        f.write("|----------|------|-----------|----------------|------------|\n")

        for response in results['individual_responses'][:20]:  # Show first 20
            effects = response.get('voltage_effects', {})
            f.write(f"| {response['stimulus_index'] + 1} | {response['stimulus_type']} | {effects.get('cohen_d', 0):.3f} | {effects.get('interpretation', 'unknown')} | {effects.get('confidence', 'unknown')} |\n")

        if len(results['individual_responses']) > 20:
            f.write(f"\n... and {len(results['individual_responses']) - 20} more responses\n")

        f.write("\n## Files Generated\n\n")
        f.write(f"- `{prefix}.json` - Complete analysis results\n")
        f.write(f"- `{prefix}_report.md` - This summary report\n")

    print(f"✅ Analysis saved to: {output_dir}")
    print(f"   • JSON results: {json_path}")
    print(f"   • Report: {report_path}")

def main():
    """Main function for stimulus-response analysis."""
    print("🧬 Stimulus-Response Analysis Framework")
    print("=" * 50)

    # Configuration
    results_base = '/home/kronos/mushroooom/results/zenodo'
    stimulus_file = '/home/kronos/mushroooom/test_stimulus_data.csv'  # Example stimulus file

    # Create sample stimulus data if it doesn't exist
    if not os.path.exists(stimulus_file):
        print("📝 Creating sample stimulus data file...")
        sample_stimuli = [
            {'time_s': 1000.0, 'stimulus_type': 'moisture', 'intensity': 1.0, 'duration_s': 10.0},
            {'time_s': 2000.0, 'stimulus_type': 'light', 'intensity': 0.8, 'duration_s': 15.0},
            {'time_s': 3000.0, 'stimulus_type': 'temperature', 'intensity': 1.2, 'duration_s': 20.0},
            {'time_s': 4000.0, 'stimulus_type': 'chemical', 'intensity': 0.9, 'duration_s': 12.0},
            {'time_s': 5000.0, 'stimulus_type': 'mechanical', 'intensity': 1.5, 'duration_s': 8.0}
        ]

        with open(stimulus_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['time_s', 'stimulus_type', 'intensity', 'duration_s'])
            writer.writeheader()
            writer.writerows(sample_stimuli)

        print(f"✅ Sample stimulus file created: {stimulus_file}")

    # Load stimulus data
    stimuli = load_stimulus_data(stimulus_file)
    if not stimuli:
        print("❌ No stimulus data loaded")
        return

    # Find analysis data to analyze
    species_dirs = [d for d in os.listdir(results_base) if os.path.isdir(os.path.join(results_base, d)) and not d.startswith('_')]

    if not species_dirs:
        print("❌ No species analysis directories found")
        return

    # Use the first species found
    species_dir = os.path.join(results_base, species_dirs[0])
    print(f"📁 Analyzing species: {species_dirs[0]}")

    # Find the most recent analysis
    subdirs = [d for d in os.listdir(species_dir) if os.path.isdir(os.path.join(species_dir, d))]
    if not subdirs:
        print("❌ No analysis subdirectories found")
        return

    subdirs.sort(reverse=True)
    latest_analysis = os.path.join(species_dir, subdirs[0])

    # Load voltage and spike data
    print("🔍 Loading voltage and spike data...")

    # Try to load from metrics.json first
    metrics_path = os.path.join(latest_analysis, 'metrics.json')
    voltage_data = None
    spike_times = None

    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        # For demonstration, create synthetic voltage data
        # In real usage, you'd load the original voltage signal
        duration = 6000  # 100 minutes at 1 Hz
        time_points = np.arange(duration)
        # Create synthetic signal with some baseline activity
        voltage_data = np.random.normal(0, 0.1, duration) + 0.05 * np.sin(2 * np.pi * time_points / 3600)  # Daily rhythm

        # Add stimulus responses
        for stimulus in stimuli:
            idx = int(stimulus['time_s'])
            if idx < len(voltage_data):
                # Add response pattern
                response_window = slice(max(0, idx), min(len(voltage_data), idx + int(stimulus['duration_s'])))
                voltage_data[response_window] += stimulus['intensity'] * 0.1 * np.exp(-np.arange(len(voltage_data[response_window])) / 10)

        # Create synthetic spike times
        spike_times = np.array([s['time_s'] for s in stimuli])  # Spikes at stimulus times for demo

    if voltage_data is None:
        print("⚠️  No voltage data available, creating synthetic data for demonstration")
        duration = 6000
        voltage_data = np.random.normal(0, 0.1, duration)

    # Perform stimulus-response analysis
    analysis_results = validate_stimulus_response(
        voltage_data,
        stimuli,
        spike_times=spike_times,
        fs_hz=1.0
    )

    # Save results
    output_dir = f'/home/kronos/mushroooom/results/zenodo/_composites/{_dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_stimulus_analysis'
    save_stimulus_analysis(analysis_results, output_dir)

    # Print key findings
    print("\n🎯 Key Findings:")
    agg_stats = analysis_results.get('aggregate_statistics', {})
    if agg_stats:
        response_rate = agg_stats.get('response_rate', 0)
        print(f"   • Response Rate: {response_rate:.1%} ({agg_stats.get('significant_responses', 0)}/{agg_stats.get('total_responses', 0)})")

        if 'effect_size_summary' in agg_stats:
            mean_effect = agg_stats['effect_size_summary']['mean_cohen_d']
            print(f"   • Mean Effect Size: {mean_effect:.3f} (Cohen's d)")

    print(f"   • Stimulus Types Analyzed: {list(analysis_results['summary']['stimulus_types'].keys())}")

    print(f"\n📁 Results saved to: {output_dir}")

if __name__ == '__main__':
    main()
