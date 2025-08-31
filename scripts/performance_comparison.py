#!/usr/bin/env python3
"""
Performance Comparison of Analysis Methods

Compares execution times and results quality across different analysis implementations.
"""

import os
import time
import json
import datetime as _dt
from typing import Dict, List
import subprocess
import matplotlib.pyplot as plt

def time_analysis(script_path: str, args: List[str], description: str) -> Dict:
    """
    Time the execution of an analysis script.

    Args:
        script_path: Path to the analysis script
        args: Command line arguments
        description: Description of the analysis

    Returns:
        Timing and result information
    """
    print(f"⏱️  Running {description}...")

    start_time = time.time()

    try:
        cmd = ['python3', script_path] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout

        end_time = time.time()
        execution_time = end_time - start_time

        success = result.returncode == 0

        return {
            'description': description,
            'execution_time': execution_time,
            'success': success,
            'return_code': result.returncode,
            'stdout_length': len(result.stdout),
            'stderr_length': len(result.stderr),
            'script': os.path.basename(script_path)
        }

    except subprocess.TimeoutExpired:
        end_time = time.time()
        return {
            'description': description,
            'execution_time': end_time - start_time,
            'success': False,
            'timeout': True,
            'script': os.path.basename(script_path)
        }
    except Exception as e:
        end_time = time.time()
        return {
            'description': description,
            'execution_time': end_time - start_time,
            'success': False,
            'error': str(e),
            'script': os.path.basename(script_path)
        }

def run_performance_comparison(data_file: str, output_dir: str) -> Dict:
    """
    Run performance comparison of different analysis methods.

    Args:
        data_file: Path to data file
        output_dir: Output directory

    Returns:
        Performance comparison results
    """
    print("🚀 Performance Comparison of Analysis Methods")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Define analysis methods to compare
    methods = [
        {
            'script': '/home/kronos/mushroooom/scripts/ultra_fast_analysis.py',
            'args': ['--file', data_file],
            'description': 'Ultra-Fast Analysis'
        },
        {
            'script': '/home/kronos/mushroooom/scripts/fast_multichannel_analysis.py',
            'args': ['--file', data_file, '--quiet'],
            'description': 'Fast Multi-Channel Analysis'
        },
        {
            'script': '/home/kronos/mushroooom/analyze_metrics.py',
            'args': ['--file', data_file, '--scan_channels'],
            'description': 'Original Multi-Channel Analysis'
        }
    ]

    results = []

    for method in methods:
        if os.path.exists(method['script']):
            result = time_analysis(method['script'], method['args'], method['description'])
            results.append(result)

            status = "✅" if result['success'] else "❌"
            time_str = ".2f"
            print(f"{status} {result['description']}: {time_str}")

            if not result['success']:
                if 'timeout' in result:
                    print("   ⏰ Timed out")
                elif 'error' in result:
                    print(f"   💥 Error: {result['error']}")
                else:
                    print(f"   ⚠️  Failed with code {result['return_code']}")
        else:
            print(f"⚠️  Script not found: {method['script']}")

    # Create performance report
    report = {
        'metadata': {
            'data_file': data_file,
            'timestamp': _dt.datetime.now().isoformat(),
            'comparison_type': 'analysis_performance'
        },
        'results': results,
        'summary': {
            'total_methods': len(results),
            'successful_methods': sum(1 for r in results if r['success']),
            'fastest_method': min((r for r in results if r['success']), key=lambda x: x['execution_time'])['description'] if any(r['success'] for r in results) else None,
            'slowest_method': max((r for r in results if r['success']), key=lambda x: x['execution_time'])['description'] if any(r['success'] for r in results) else None,
        }
    }

    if any(r['success'] for r in results):
        successful_times = [r['execution_time'] for r in results if r['success']]
        report['summary']['speedup_ratio'] = max(successful_times) / min(successful_times)

    # Save report
    report_path = os.path.join(output_dir, 'performance_comparison.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Create visualization
    create_performance_plot(results, output_dir)

    print("\n📊 Performance Comparison Summary:")
    print(f"   • Methods tested: {len(results)}")
    print(f"   • Successful: {sum(1 for r in results if r['success'])}")
    if report['summary']['fastest_method']:
        print(f"   • Fastest: {report['summary']['fastest_method']}")
        if 'speedup_ratio' in report['summary']:
            print(f"   • Speedup ratio: {report['summary']['speedup_ratio']:.1f}x")
    print(f"📁 Report saved to: {report_path}")

    return report

def create_performance_plot(results: List[Dict], output_dir: str):
    """
    Create performance comparison visualization.

    Args:
        results: Performance results
        output_dir: Output directory
    """
    successful_results = [r for r in results if r['success']]

    if not successful_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Execution time comparison
    methods = [r['description'] for r in successful_results]
    times = [r['execution_time'] for r in successful_results]

    bars = ax1.bar(methods, times, alpha=0.7, color=['red', 'orange', 'green'])
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Analysis Speed Comparison')
    ax1.tick_params(axis='x', rotation=45)

    # Add time labels
    for bar, time_val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                '.1f', ha='center', va='bottom')

    # Success rate
    all_methods = [r['description'] for r in results]
    success_rates = [1 if r['success'] else 0 for r in results]

    ax2.bar(all_methods, success_rates, alpha=0.7, color=['green' if s else 'red' for s in success_rates])
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Method Reliability')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=45)

    # Add success labels
    for i, (method, success) in enumerate(zip(all_methods, success_rates)):
        ax2.text(i, success + 0.05, 'Success' if success else 'Failed',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function."""
    import argparse

    ap = argparse.ArgumentParser(description='Performance Comparison of Analysis Methods')
    ap.add_argument('--file', required=True, help='Data file to analyze')
    ap.add_argument('--output_dir', help='Output directory for results')

    args = ap.parse_args()

    if not args.output_dir:
        timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        args.output_dir = f'/home/kronos/mushroooom/results/zenodo/_composites/{timestamp}_performance_comparison'

    report = run_performance_comparison(args.file, args.output_dir)

if __name__ == '__main__':
    main()
