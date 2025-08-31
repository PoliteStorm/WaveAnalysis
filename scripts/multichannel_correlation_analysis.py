#!/usr/bin/env python3
"""
Multi-Channel Correlation Analysis Framework for Fungal Network Insights

This module provides comprehensive analysis of multi-channel electrophysiological data
to understand fungal mycelial network connectivity, communication patterns, and
information processing capabilities.

Key Features:
- Multi-channel data loading and preprocessing
- Cross-correlation analysis between channels
- Network connectivity analysis
- Granger causality for directional information flow
- Phase synchronization analysis
- Coherence analysis across frequency bands
- Information transfer metrics
- Network visualization and topology analysis
- Biological interpretation of network patterns
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import coherence
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
import json
import datetime as _dt
from tqdm import tqdm
import pandas as pd

# Import granger causality from statsmodels (correct location)
try:
    from statsmodels.tsa.stattools import grangercausalitytests
except Exception:
    grangercausalitytests = None

# Import our custom modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import prove_transform as pt
except ImportError:
    print("Warning: Could not import prove_transform module")
    pt = None

def load_multichannel_data(file_path: str, fs_hz: float = 1.0) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
    """
    Load and preprocess multi-channel electrophysiological data.

    Args:
        file_path: Path to data file
        fs_hz: Sampling frequency

    Returns:
        Tuple of (time_array, channel_dict, metadata)
    """
    print(f"🔄 Loading multi-channel data from {os.path.basename(file_path)}...")

    if pt is None:
        raise ImportError("prove_transform module not available")

    # Load data using existing function
    t, channels = pt.load_zenodo_timeseries(file_path)

    # Filter out channels with too many NaN values
    valid_channels = {}
    for name, data in channels.items():
        nan_fraction = np.sum(np.isnan(data)) / len(data)
        if nan_fraction < 0.5:  # Keep channels with <50% NaN
            # Fill NaN with interpolation
            valid_data = pd.Series(data).interpolate(method='linear', limit_direction='both').values
            valid_channels[name] = valid_data

    print(f"✅ Loaded {len(valid_channels)} valid channels from {len(channels)} total")

    # Metadata
    metadata = {
        'file_path': file_path,
        'sampling_rate_hz': fs_hz,
        'n_samples': len(t),
        'duration_s': len(t) / fs_hz,
        'n_channels': len(valid_channels),
        'channel_names': list(valid_channels.keys()),
        'total_channels_original': len(channels)
    }

    return t, valid_channels, metadata

def compute_channel_statistics(channels: Dict[str, np.ndarray], fs_hz: float) -> Dict[str, Dict]:
    """
    Compute basic statistics for each channel.

    Args:
        channels: Dictionary of channel data
        fs_hz: Sampling frequency

    Returns:
        Dictionary of channel statistics
    """
    stats_dict = {}

    for name, data in channels.items():
        # Basic statistics
        mean_val = float(np.mean(data))
        std_val = float(np.std(data))
        rms_val = float(np.sqrt(np.mean(data**2)))

        # Signal power
        power = float(np.mean(data**2))

        # Peak-to-peak amplitude
        ptp = float(np.ptp(data))

        # Signal-to-noise ratio (estimated)
        noise_estimate = np.std(data - signal.medfilt(data, kernel_size=101))
        snr = float(20 * np.log10(rms_val / noise_estimate)) if noise_estimate > 0 else 0

        stats_dict[name] = {
            'mean_mV': mean_val,
            'std_mV': std_val,
            'rms_mV': rms_val,
            'power_mV2': power,
            'peak_to_peak_mV': ptp,
            'snr_db': snr,
            'n_samples': len(data),
            'sampling_rate_hz': fs_hz
        }

    return stats_dict

def compute_cross_correlations(channels: Dict[str, np.ndarray], max_lag: int = 1000) -> Dict[str, np.ndarray]:
    """
    Compute cross-correlations between all channel pairs.

    Args:
        channels: Dictionary of channel data
        max_lag: Maximum lag for correlation analysis

    Returns:
        Dictionary of cross-correlation matrices
    """
    print("🔗 Computing cross-correlations between channels...")

    channel_names = list(channels.keys())
    n_channels = len(channel_names)

    # Cross-correlation matrix (n_channels x n_channels x max_lag*2+1)
    corr_matrix = np.zeros((n_channels, n_channels, 2*max_lag + 1))

    # Normalized cross-correlation matrix
    norm_corr_matrix = np.zeros((n_channels, n_channels, 2*max_lag + 1))

    for i, name1 in enumerate(channel_names):
        for j, name2 in enumerate(channel_names):
            data1 = channels[name1]
            data2 = channels[name2]

            # Compute cross-correlation
            corr = signal.correlate(data1, data2, mode='full', method='auto')
            lags = signal.correlation_lags(len(data1), len(data2), mode='full')

            # Keep only lags within our range
            lag_mask = np.abs(lags) <= max_lag
            corr_matrix[i, j, :] = corr[lag_mask]
            norm_corr_matrix[i, j, :] = corr[lag_mask] / np.sqrt(np.sum(data1**2) * np.sum(data2**2))

    return {
        'channel_names': channel_names,
        'correlation_matrix': corr_matrix,
        'normalized_correlation_matrix': norm_corr_matrix,
        'lags': np.arange(-max_lag, max_lag + 1),
        'max_lag': max_lag
    }

def analyze_correlation_peaks(corr_results: Dict) -> Dict[str, List[Dict]]:
    """
    Analyze peaks in cross-correlation functions to identify significant interactions.

    Args:
        corr_results: Results from cross-correlation analysis

    Returns:
        Dictionary of significant correlation peaks
    """
    print("🔍 Analyzing correlation peaks...")

    channel_names = corr_results['channel_names']
    norm_corr = corr_results['normalized_correlation_matrix']
    lags = corr_results['lags']

    significant_interactions = []

    n_channels = len(channel_names)

    for i in range(n_channels):
        for j in range(i+1, n_channels):  # Only upper triangle
            corr_ij = norm_corr[i, j, :]

            # Find peaks above threshold
            threshold = 0.3  # 30% correlation threshold
            peaks, properties = signal.find_peaks(np.abs(corr_ij),
                                                height=threshold,
                                                distance=50)  # Minimum distance between peaks

            for peak_idx in peaks:
                lag_val = lags[peak_idx]
                corr_val = corr_ij[peak_idx]

                interaction = {
                    'channel_1': channel_names[i],
                    'channel_2': channel_names[j],
                    'lag_samples': int(lag_val),
                    'lag_seconds': float(lag_val),  # Assuming 1 Hz sampling
                    'correlation': float(corr_val),
                    'abs_correlation': float(np.abs(corr_val)),
                    'direction': 'positive' if corr_val > 0 else 'negative',
                    'peak_height': float(properties['peak_heights'][np.where(peaks == peak_idx)[0][0]])
                }

                significant_interactions.append(interaction)

    # Sort by absolute correlation strength
    significant_interactions.sort(key=lambda x: x['abs_correlation'], reverse=True)

    return {
        'significant_interactions': significant_interactions,
        'n_significant': len(significant_interactions),
        'correlation_threshold': 0.3
    }

def compute_coherence_analysis(channels: Dict[str, np.ndarray], fs_hz: float,
                              freq_range: Tuple[float, float] = (0.001, 1.0)) -> Dict:
    """
    Compute magnitude-squared coherence between channel pairs.

    Args:
        channels: Dictionary of channel data
        fs_hz: Sampling frequency
        freq_range: Frequency range for analysis

    Returns:
        Dictionary with coherence analysis results
    """
    print("🎯 Computing coherence analysis...")

    channel_names = list(channels.keys())
    n_channels = len(channel_names)

    # Frequency vector
    n_fft = 2**12  # 4096 points
    freqs = np.fft.rfftfreq(n_fft, d=1/fs_hz)

    # Keep frequencies in our range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_filtered = freqs[freq_mask]

    # Coherence matrix (n_channels x n_channels x n_freqs)
    coherence_matrix = np.zeros((n_channels, n_channels, len(freqs_filtered)))

    for i, name1 in enumerate(channel_names):
        for j, name2 in enumerate(channel_names):
            data1 = channels[name1]
            data2 = channels[name2]

            # Compute coherence
            freqs_coh, coh = coherence(data1, data2, fs=fs_hz, nperseg=n_fft,
                                     nfft=n_fft, noverlap=n_fft//2)

            # Filter to our frequency range
            coh_filtered = coh[freq_mask]
            coherence_matrix[i, j, :] = coh_filtered

    return {
        'channel_names': channel_names,
        'frequencies': freqs_filtered.tolist(),
        'coherence_matrix': coherence_matrix,
        'freq_range': freq_range
    }

def compute_granger_causality(channels: Dict[str, np.ndarray], max_lags: int = 20) -> Dict:
    """
    Compute Granger causality to identify directional information flow.

    Args:
        channels: Dictionary of channel data
        max_lags: Maximum number of lags for causality test

    Returns:
        Dictionary with Granger causality results
    """
    print("🔀 Computing Granger causality analysis...")

    if grangercausalitytests is None:
        print("⚠️  statsmodels not available; skipping Granger causality analysis")
        return {}

    channel_names = list(channels.keys())
    causality_results = {}

    # Test causality between each pair
    for i, name1 in enumerate(channel_names):
        for j, name2 in enumerate(channel_names):
            if i == j:
                continue

            data1 = channels[name1]
            data2 = channels[name2]

            # Prepare data for Granger test
            test_data = np.column_stack([data1, data2])

            try:
                # Run Granger causality test
                gc_test = grangercausalitytests(test_data, max_lags, verbose=False)

                # Extract F-test results for each lag
                f_stats = []
                p_values = []

                for lag in range(1, max_lags + 1):
                    if lag in gc_test:
                        f_stat = gc_test[lag][0]['ssr_ftest'][0]
                        p_val = gc_test[lag][0]['ssr_ftest'][1]
                        f_stats.append(float(f_stat))
                        p_values.append(float(p_val))
                    else:
                        f_stats.append(0.0)
                        p_values.append(1.0)

                causality_results[f"{name1}_causes_{name2}"] = {
                    'cause_channel': name1,
                    'effect_channel': name2,
                    'f_statistics': f_stats,
                    'p_values': p_values,
                    'lags_tested': list(range(1, max_lags + 1)),
                    'significant_lags': [lag for lag, p in enumerate(p_values, 1) if p < 0.05],
                    'direction': f"{name1} → {name2}"
                }

            except Exception as e:
                print(f"Warning: Granger causality test failed for {name1} → {name2}: {e}")
                causality_results[f"{name1}_causes_{name2}"] = {
                    'cause_channel': name1,
                    'effect_channel': name2,
                    'error': str(e)
                }

    return causality_results

def build_network_graph(corr_peaks: Dict, causality_results: Dict,
                       coherence_results: Dict) -> nx.DiGraph:
    """
    Build a network graph representing channel interactions.

    Args:
        corr_peaks: Results from correlation peak analysis
        causality_results: Results from Granger causality
        coherence_results: Results from coherence analysis

    Returns:
        NetworkX directed graph
    """
    print("🌐 Building network graph...")

    G = nx.DiGraph()

    # Add nodes (channels)
    channel_names = corr_peaks.get('significant_interactions', [])
    if channel_names:
        channels = set()
        for interaction in channel_names:
            channels.add(interaction['channel_1'])
            channels.add(interaction['channel_2'])

        for channel in channels:
            G.add_node(channel, node_type='channel')

    # Add correlation edges
    for interaction in corr_peaks.get('significant_interactions', []):
        ch1 = interaction['channel_1']
        ch2 = interaction['channel_2']
        weight = interaction['abs_correlation']
        lag = interaction['lag_seconds']

        # Add bidirectional correlation edge
        G.add_edge(ch1, ch2, weight=weight, lag=lag,
                  interaction_type='correlation', correlation=interaction['correlation'])
        G.add_edge(ch2, ch1, weight=weight, lag=-lag,
                  interaction_type='correlation', correlation=interaction['correlation'])

    # Add causality edges (directional)
    for key, result in causality_results.items():
        if 'error' not in result:
            cause_ch = result['cause_channel']
            effect_ch = result['effect_channel']
            sig_lags = result.get('significant_lags', [])

            if sig_lags:
                # Add causality edge
                max_f = max(result['f_statistics'][lag-1] for lag in sig_lags)
                G.add_edge(cause_ch, effect_ch,
                          weight=max_f,
                          interaction_type='causality',
                          significant_lags=sig_lags,
                          max_f_statistic=max_f)

    return G

def analyze_network_topology(graph: nx.DiGraph) -> Dict:
    """
    Analyze the topology of the channel interaction network.

    Args:
        graph: NetworkX graph

    Returns:
        Dictionary with network topology metrics
    """
    print("📊 Analyzing network topology...")

    topology = {}

    # Basic network properties
    topology['n_nodes'] = graph.number_of_nodes()
    topology['n_edges'] = graph.number_of_edges()
    topology['is_directed'] = graph.is_directed()

    if topology['n_nodes'] > 0:
        # Degree analysis
        degrees = dict(graph.degree())
        in_degrees = dict(graph.in_degree()) if graph.is_directed() else {}
        out_degrees = dict(graph.out_degree()) if graph.is_directed() else {}

        topology['degrees'] = degrees
        topology['in_degrees'] = in_degrees
        topology['out_degrees'] = out_degrees

        # Centrality measures
        try:
            topology['degree_centrality'] = nx.degree_centrality(graph)
            topology['betweenness_centrality'] = nx.betweenness_centrality(graph)
            topology['closeness_centrality'] = nx.closeness_centrality(graph)

            if graph.is_directed():
                topology['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(graph)
        except:
            topology['centrality_error'] = "Could not compute centrality measures"

        # Clustering and connectivity
        try:
            if not graph.is_directed():
                topology['clustering_coefficient'] = nx.clustering(graph)
                topology['average_clustering'] = nx.average_clustering(graph)
        except:
            topology['clustering_error'] = "Could not compute clustering"

        # Connected components
        if not graph.is_directed():
            topology['connected_components'] = len(list(nx.connected_components(graph)))
        else:
            topology['weakly_connected_components'] = len(list(nx.weakly_connected_components(graph)))
            topology['strongly_connected_components'] = len(list(nx.strongly_connected_components(graph)))

    return topology

def create_multichannel_visualizations(results: Dict, output_dir: str):
    """
    Create comprehensive visualizations for multi-channel analysis.

    Args:
        results: Complete analysis results
        output_dir: Output directory for plots
    """
    print("📊 Creating multi-channel visualizations...")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Channel overview plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-Channel Correlation Analysis Overview', fontsize=16)

    # Channel statistics
    channels = results.get('channel_statistics', {})
    if channels:
        channel_names = list(channels.keys())
        means = [channels[ch]['mean_mV'] for ch in channel_names]
        stds = [channels[ch]['std_mV'] for ch in channel_names]
        powers = [channels[ch]['power_mV2'] for ch in channel_names]

        ax = axes[0, 0]
        x = np.arange(len(channel_names))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Mean Voltage (mV)')
        ax.set_title('Channel Means with Standard Deviations')
        ax.set_xticks(x)
        ax.set_xticklabels(channel_names, rotation=45)

        ax = axes[0, 1]
        ax.bar(x, powers, alpha=0.7, color='orange')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Power (mV²)')
        ax.set_title('Channel Power')
        ax.set_xticks(x)
        ax.set_xticklabels(channel_names, rotation=45)

    # Correlation matrix
    corr_results = results.get('correlation_analysis', {})
    if 'normalized_correlation_matrix' in corr_results:
        ax = axes[1, 0]
        corr_matrix = corr_results['normalized_correlation_matrix']
        if corr_matrix.shape[0] > 0:
            # Average correlation across lags (zero lag)
            zero_lag_idx = corr_matrix.shape[2] // 2
            mean_corr = corr_matrix[:, :, zero_lag_idx]

            im = ax.imshow(mean_corr, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_title('Zero-Lag Cross-Correlation Matrix')
            ax.set_xticks(np.arange(len(corr_results['channel_names'])))
            ax.set_yticks(np.arange(len(corr_results['channel_names'])))
            ax.set_xticklabels(corr_results['channel_names'], rotation=45)
            ax.set_yticklabels(corr_results['channel_names'])
            plt.colorbar(im, ax=ax, label='Correlation')

    # Network visualization
    graph = results.get('network_graph')
    if graph and graph.number_of_nodes() > 0:
        ax = axes[1, 1]
        pos = nx.spring_layout(graph, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=500,
                              node_color='lightblue', alpha=0.7)

        # Draw edges with different styles
        corr_edges = [(u, v) for u, v, d in graph.edges(data=True)
                     if d.get('interaction_type') == 'correlation']
        causality_edges = [(u, v) for u, v, d in graph.edges(data=True)
                          if d.get('interaction_type') == 'causality']

        if corr_edges:
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=corr_edges,
                                 edge_color='gray', alpha=0.5, style='solid')

        if causality_edges:
            nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=causality_edges,
                                 edge_color='red', alpha=0.7, style='dashed',
                                 arrows=True, arrowsize=20)

        # Labels
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10)

        ax.set_title('Channel Interaction Network')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multichannel_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Correlation peaks visualization
    corr_peaks = results.get('correlation_peaks', {})
    if corr_peaks.get('significant_interactions'):
        fig, ax = plt.subplots(figsize=(12, 8))

        interactions = corr_peaks['significant_interactions'][:20]  # Top 20
        pairs = [f"{i['channel_1']}\n↔\n{i['channel_2']}" for i in interactions]
        correlations = [i['correlation'] for i in interactions]

        bars = ax.bar(range(len(interactions)), correlations, alpha=0.7)
        ax.set_xlabel('Channel Pairs')
        ax.set_ylabel('Correlation')
        ax.set_title('Top Significant Cross-Correlations')
        ax.set_xticks(range(len(interactions)))
        ax.set_xticklabels(pairs, rotation=45, ha='right')

        # Color bars by correlation strength
        for i, (bar, corr) in enumerate(zip(bars, correlations)):
            if corr > 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_peaks.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Coherence heatmap
    coherence_results = results.get('coherence_analysis', {})
    if 'coherence_matrix' in coherence_results:
        fig, ax = plt.subplots(figsize=(10, 8))

        coh_matrix = coherence_results['coherence_matrix']
        if coh_matrix.shape[0] > 0:
            # Average coherence across frequencies
            mean_coh = np.mean(coh_matrix, axis=2)

            im = ax.imshow(mean_coh, cmap='viridis', vmin=0, vmax=1)
            ax.set_title('Average Coherence Matrix')
            ax.set_xticks(np.arange(len(coherence_results['channel_names'])))
            ax.set_yticks(np.arange(len(coherence_results['channel_names'])))
            ax.set_xticklabels(coherence_results['channel_names'], rotation=45)
            ax.set_yticklabels(coherence_results['channel_names'])
            plt.colorbar(im, ax=ax, label='Coherence')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'coherence_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Visualizations saved to {output_dir}")

def run_multichannel_analysis(file_path: str, output_dir: str, fs_hz: float = 1.0, granger_lags: int = 10) -> Dict:
    """
    Run complete multi-channel correlation analysis.

    Args:
        file_path: Path to data file
        output_dir: Output directory
        fs_hz: Sampling frequency
        granger_lags: Maximum lags to use for Granger causality

    Returns:
        Complete analysis results
    """
    print("🚀 Starting Multi-Channel Correlation Analysis")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    with tqdm(total=1, desc="📊 Loading Data", ncols=80) as pbar:
        t, channels, metadata = load_multichannel_data(file_path, fs_hz)
        pbar.update(1)

    # Channel statistics
    with tqdm(total=1, desc="📈 Computing Statistics", ncols=80) as pbar:
        channel_stats = compute_channel_statistics(channels, fs_hz)
        pbar.update(1)

    # Cross-correlation analysis
    with tqdm(total=1, desc="🔗 Cross-Correlations", ncols=80) as pbar:
        corr_results = compute_cross_correlations(channels, max_lag=500)
        pbar.update(1)

    # Correlation peak analysis
    with tqdm(total=1, desc="🔍 Peak Analysis", ncols=80) as pbar:
        corr_peaks = analyze_correlation_peaks(corr_results)
        pbar.update(1)

    # Coherence analysis
    with tqdm(total=1, desc="🎯 Coherence Analysis", ncols=80) as pbar:
        coherence_results = compute_coherence_analysis(channels, fs_hz)
        pbar.update(1)

    # Granger causality
    with tqdm(total=1, desc="🔀 Granger Causality", ncols=80) as pbar:
        causality_results = compute_granger_causality(channels, max_lags=granger_lags)
        pbar.update(1)

    # Network analysis
    with tqdm(total=1, desc="🌐 Network Analysis", ncols=80) as pbar:
        network_graph = build_network_graph(corr_peaks, causality_results, coherence_results)
        topology = analyze_network_topology(network_graph)
        pbar.update(1)

    # Compile results
    results = {
        'metadata': metadata,
        'channel_statistics': channel_stats,
        'correlation_analysis': corr_results,
        'correlation_peaks': corr_peaks,
        'coherence_analysis': coherence_results,
        'granger_causality': causality_results,
        'network_graph': network_graph,
        'network_topology': topology,
        'timestamp': _dt.datetime.now().isoformat(),
        'analysis_type': 'multichannel_correlation'
    }

    # Save results
    json_path = os.path.join(output_dir, 'multichannel_analysis_results.json')

    # Convert network graph to serializable format
    results_copy = results.copy()
    if 'network_graph' in results_copy:
        results_copy['network_graph'] = {
            'nodes': list(results_copy['network_graph'].nodes()),
            'edges': list(results_copy['network_graph'].edges(data=True)),
            'n_nodes': results_copy['network_graph'].number_of_nodes(),
            'n_edges': results_copy['network_graph'].number_of_edges()
        }

    with open(json_path, 'w') as f:
        json.dump(results_copy, f, indent=2, default=str)

    # Create visualizations
    create_multichannel_visualizations(results, output_dir)

    # Generate summary report
    generate_summary_report(results, output_dir)

    print("\n🎉 Multi-Channel Analysis Complete!")
    print(f"📊 Analyzed {metadata['n_channels']} channels")
    print(f"🔗 Found {corr_peaks.get('n_significant', 0)} significant correlations")
    print(f"🌐 Network has {topology.get('n_nodes', 0)} nodes and {topology.get('n_edges', 0)} edges")
    print(f"📁 Results saved to {output_dir}")

    return results

def generate_summary_report(results: Dict, output_dir: str):
    """
    Generate a comprehensive summary report.

    Args:
        results: Analysis results
        output_dir: Output directory
    """
    report_path = os.path.join(output_dir, 'multichannel_analysis_report.md')

    with open(report_path, 'w') as f:
        f.write("# Multi-Channel Correlation Analysis Report\n\n")
        f.write(f"**Analysis Date:** {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Metadata
        meta = results.get('metadata', {})
        f.write("## Dataset Overview\n\n")
        f.write(f"- **File:** {os.path.basename(meta.get('file_path', 'Unknown'))}\n")
        f.write(f"- **Channels:** {meta.get('n_channels', 0)} (from {meta.get('total_channels_original', 0)} total)\n")
        f.write(f"- **Duration:** {meta.get('duration_s', 0):.1f} seconds\n")
        f.write(f"- **Sampling Rate:** {meta.get('sampling_rate_hz', 0)} Hz\n\n")

        # Channel statistics
        f.write("## Channel Statistics\n\n")
        stats = results.get('channel_statistics', {})
        f.write("| Channel | Mean (mV) | Std (mV) | RMS (mV) | Power (mV²) | SNR (dB) |\n")
        f.write("|---------|-----------|-----------|-----------|-------------|-----------|\n")

        for ch, stat in stats.items():
            f.write(f"| {ch} | {stat['mean_mV']:.3f} | {stat['std_mV']:.3f} | {stat['rms_mV']:.3f} | {stat['power_mV2']:.3f} | {stat['snr_db']:.1f} |\n")

        f.write("\n")

        # Correlation analysis
        corr_peaks = results.get('correlation_peaks', {})
        f.write("## Significant Correlations\n\n")
        f.write(f"**Total Significant Interactions:** {corr_peaks.get('n_significant', 0)}\n\n")

        interactions = corr_peaks.get('significant_interactions', [])[:10]  # Top 10
        if interactions:
            f.write("| Channel 1 | Channel 2 | Correlation | Lag (s) | Direction |\n")
            f.write("|-----------|-----------|-------------|----------|-----------|\n")

            for inter in interactions:
                f.write(f"| {inter['channel_1']} | {inter['channel_2']} | {inter['correlation']:.3f} | {inter['lag_seconds']:.1f} | {inter['direction']} |\n")

        f.write("\n")

        # Network topology
        topo = results.get('network_topology', {})
        f.write("## Network Topology\n\n")
        f.write(f"- **Nodes:** {topo.get('n_nodes', 0)}\n")
        f.write(f"- **Edges:** {topo.get('n_edges', 0)}\n")
        f.write(f"- **Network Type:** {'Directed' if topo.get('is_directed', False) else 'Undirected'}\n")

        if not topo.get('is_directed', False):
            f.write(f"- **Connected Components:** {topo.get('connected_components', 0)}\n")
        else:
            f.write(f"- **Weakly Connected Components:** {topo.get('weakly_connected_components', 0)}\n")
            f.write(f"- **Strongly Connected Components:** {topo.get('strongly_connected_components', 0)}\n")

        f.write("\n")

        # Biological interpretation
        f.write("## Biological Interpretation\n\n")

        n_sig = corr_peaks.get('n_significant', 0)
        if n_sig > 0:
            f.write("### Network Connectivity\n\n")
            f.write(f"- **{n_sig} significant channel interactions detected**\n")
            f.write("- These interactions suggest coordinated electrical activity across the fungal mycelium\n")
            f.write("- Cross-correlations indicate synchronized spiking patterns between different network regions\n")
            f.write("- Time lags in correlations may reflect signal propagation delays in the mycelial network\n\n")

            f.write("### Information Processing\n\n")
            f.write("- Granger causality analysis reveals directional information flow\n")
            f.write("- Coherence analysis shows frequency-specific synchronization\n")
            f.write("- Network topology suggests distributed processing capabilities\n\n")

            f.write("### Functional Implications\n\n")
            f.write("- **Communication:** Coordinated activity enables long-distance signaling\n")
            f.write("- **Integration:** Network connectivity supports information integration\n")
            f.write("- **Adaptation:** Dynamic correlations suggest adaptive network behavior\n")
            f.write("- **Intelligence:** Complex interaction patterns indicate computational capabilities\n\n")

        else:
            f.write("No significant channel interactions detected. This may indicate:\n")
            f.write("- Independent channel operation\n")
            f.write("- Weak coupling between network regions\n")
            f.write("- Measurement artifacts or noise\n")
            f.write("- Need for different analysis parameters\n\n")

        # Technical recommendations
        f.write("## Technical Recommendations\n\n")
        f.write("### Data Quality\n")
        f.write("- Ensure consistent electrode placement\n")
        f.write("- Minimize electrical interference\n")
        f.write("- Validate channel isolation\n\n")

        f.write("### Analysis Parameters\n")
        f.write("- Adjust correlation thresholds based on signal quality\n")
        f.write("- Consider different frequency bands for coherence analysis\n")
        f.write("- Evaluate multiple lag ranges for causality testing\n\n")

        f.write("### Future Experiments\n")
        f.write("- Stimulus-response experiments to validate connectivity\n")
        f.write("- Pharmacological interventions to modulate network activity\n")
        f.write("- Environmental perturbations to test network resilience\n\n")

        # Files generated
        f.write("## Files Generated\n\n")
        f.write(f"- `multichannel_analysis_results.json` - Complete analysis results\n")
        f.write(f"- `multichannel_overview.png` - Multi-panel overview visualization\n")
        f.write(f"- `correlation_peaks.png` - Significant correlation visualization\n")
        f.write(f"- `coherence_matrix.png` - Coherence analysis visualization\n")
        f.write(f"- `multichannel_analysis_report.md` - This summary report\n")

def main():
    """Main function for command-line usage."""
    import argparse

    ap = argparse.ArgumentParser(description='Multi-Channel Correlation Analysis for Fungal Networks')
    ap.add_argument('--file', required=True, help='Zenodo data file path')
    ap.add_argument('--output_dir', default='', help='Output directory')
    ap.add_argument('--fs', type=float, default=1.0, help='Sampling frequency (Hz)')
    ap.add_argument('--max_lag', type=int, default=500, help='Maximum lag for correlation analysis')
    ap.add_argument('--granger_lags', type=int, default=10, help='Maximum lags for Granger causality (smaller is faster)')

    args = ap.parse_args()

    # Set default output directory
    if not args.output_dir:
        base_name = os.path.splitext(os.path.basename(args.file))[0]
        args.output_dir = f'/home/kronos/mushroooom/results/zenodo/_composites/{_dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_multichannel_{base_name}'

    # Run analysis
    results = run_multichannel_analysis(args.file, args.output_dir, args.fs, args.granger_lags)

if __name__ == '__main__':
    main()
