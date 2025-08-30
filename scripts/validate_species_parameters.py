#!/usr/bin/env python3
"""
Validation script for species-specific sampling rates and parameters.
Tests the research-optimized configurations to ensure they work correctly.
"""
import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any


def load_config(species_name: str) -> Dict:
    """Load species configuration."""
    config_path = f"configs/{species_name}.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


def validate_nyquist_criteria(config: Dict, species_name: str) -> Dict:
    """Validate Nyquist sampling criteria based on research spiking rates."""
    fs = config.get('fs_hz', 1.0)
    min_isi = config.get('min_isi_s', 120.0)

    # Estimate maximum spike frequency from minimum ISI
    max_spike_freq = 1.0 / min_isi if min_isi > 0 else 0

    # Nyquist criterion: fs > 2 * max_frequency
    nyquist_required = 2 * max_spike_freq
    is_adequate = fs >= nyquist_required

    return {
        'species': species_name,
        'fs_hz': fs,
        'min_isi_s': min_isi,
        'estimated_max_spike_freq': max_spike_freq,
        'nyquist_required': nyquist_required,
        'sampling_adequate': is_adequate,
        'margin': fs / nyquist_required if nyquist_required > 0 else float('inf')
    }


def validate_parameter_consistency() -> Dict:
    """Validate that all parameters are consistent and research-based."""
    results = {}
    species_list = [
        'Schizophyllum_commune',
        'Enoki_fungi_Flammulina_velutipes',
        'Ghost_Fungi_Omphalotus_nidiformis',
        'Cordyceps_militari',
        'Pleurotus_djamor'
    ]

    print("🔬 Validating species-specific parameters...")

    for species in species_list:
        config = load_config(species)
        if not config:
            print(f"⚠️  No config found for {species}")
            continue

        # Validate Nyquist criteria
        nyquist_validation = validate_nyquist_criteria(config, species)

        # Check research basis
        research_basis = config.get('research_basis', '')
        parameter_version = config.get('parameter_version', '')

        # Validate parameter ranges
        fs = config.get('fs_hz', 1.0)
        min_amp = config.get('min_amp_mV', 0.1)
        min_isi = config.get('min_isi_s', 120.0)
        baseline_win = config.get('baseline_win_s', 600.0)

        parameter_validation = {
            'fs_range': 0.5 <= fs <= 10.0,
            'amp_range': 0.01 <= min_amp <= 0.5,
            'isi_range': 10 <= min_isi <= 3600,
            'baseline_range': 60 <= baseline_win <= 3600
        }

        results[species] = {
            'config': config,
            'nyquist_validation': nyquist_validation,
            'research_documented': bool(research_basis),
            'parameter_validation': parameter_validation,
            'all_valid': (
                nyquist_validation['sampling_adequate'] and
                all(parameter_validation.values()) and
                bool(research_basis)
            )
        }

        # Print validation results
        status = "✅" if results[species]['all_valid'] else "⚠️"
        print(f"{status} {species}:")
        print(f"   fs={fs}Hz, ISI_min={min_isi}s")
        print(".1f")
        print(f"   Research: {'✅' if research_basis else '❌'}")
        print()

    return results


def generate_validation_report(results: Dict) -> str:
    """Generate a comprehensive validation report."""
    timestamp = datetime.now().isoformat()

    report = f"""# Species-Specific Parameter Validation Report

**Generated:** {timestamp}
**By:** joe knowles
**For:** peer_review

## Summary

Validated {len(results)} species configurations against research-based criteria.

## Detailed Results

"""

    all_valid = True
    for species, data in results.items():
        nyquist = data['nyquist_validation']
        params = data['parameter_validation']

        report += f"""### {species.replace('_', ' ')}

**Sampling Rate:** {nyquist['fs_hz']} Hz
**Min ISI:** {nyquist['min_isi_s']} s
**Nyquist Required:** {nyquist['nyquist_required']:.3f} Hz
**Sampling Adequate:** {'✅' if nyquist['sampling_adequate'] else '❌'}
**Safety Margin:** {nyquist['margin']:.1f}x

**Parameter Validation:**
- FS Range: {'✅' if params['fs_range'] else '❌'} (0.5-10 Hz)
- Amp Range: {'✅' if params['amp_range'] else '❌'} (0.01-0.5 mV)
- ISI Range: {'✅' if params['isi_range'] else '❌'} (10-3600 s)
- Baseline Range: {'✅' if params['baseline_range'] else '❌'} (60-3600 s)

**Research Basis:** {'✅' if data['research_documented'] else '❌'}

**Overall:** {'✅ VALID' if data['all_valid'] else '⚠️ NEEDS REVIEW'}

"""

        if not data['all_valid']:
            all_valid = False

    report += f"""

## Conclusion

**Overall Status:** {'✅ ALL SPECIES VALID' if all_valid else '⚠️ SOME ISSUES FOUND'}

**Recommendations:**
- {'All parameters are research-validated and ready for production use.' if all_valid else 'Review and fix flagged parameters before deployment.'}
- Monitor computational performance with higher sampling rates.
- Validate biological accuracy through controlled experiments.

---
**Report Version:** 1.0
**Validation Date:** {timestamp}
"""

    return report


def main():
    """Main validation function."""
    results = validate_parameter_consistency()

    # Generate and save report
    report = generate_validation_report(results)
    with open('species_parameter_validation_report.md', 'w') as f:
        f.write(report)

    print("📄 Validation report saved to: species_parameter_validation_report.md")

    # Summary
    valid_count = sum(1 for r in results.values() if r['all_valid'])
    total_count = len(results)

    print("\n📊 VALIDATION SUMMARY:")
    print(f"  Species validated: {total_count}")
    print(f"  Fully valid: {valid_count}")
    print(f"  Need review: {total_count - valid_count}")

    if valid_count == total_count:
        print("\n🎉 All species parameters are research-validated!")
    else:
        print("\n⚠️  Some parameters need review - check the detailed report.")


if __name__ == '__main__':
    main()
