#!/usr/bin/env python3
"""
Audit script to ensure all results are properly timestamped, authored by Joe Knowles,
and managed professionally in consistent directory structures.
"""
import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def validate_timestamp_format(timestamp: str) -> bool:
    """Check if timestamp follows ISO format with T separator."""
    try:
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


def validate_author(metadata: Dict) -> bool:
    """Check if author is correctly set to joe knowles."""
    return metadata.get('created_by', '').lower() == 'joe knowles'


def validate_intended_for(metadata: Dict) -> bool:
    """Check if intended_for is set appropriately."""
    intended = metadata.get('intended_for', '').lower()
    return 'peer' in intended or 'review' in intended


def audit_json_file(filepath: str) -> Dict:
    """Audit a single JSON file for metadata consistency."""
    issues = []

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Check timestamp
        timestamp = data.get('timestamp', '')
        if not timestamp:
            issues.append('Missing timestamp')
        elif not validate_timestamp_format(timestamp):
            issues.append(f'Invalid timestamp format: {timestamp}')

        # Check author
        if not validate_author(data):
            issues.append(f'Invalid author: {data.get("created_by", "missing")}')

        # Check intended_for
        if not validate_intended_for(data):
            issues.append(f'Invalid intended_for: {data.get("intended_for", "missing")}')

        return {
            'filepath': filepath,
            'valid': len(issues) == 0,
            'issues': issues,
            'metadata': {
                'timestamp': timestamp,
                'created_by': data.get('created_by', ''),
                'intended_for': data.get('intended_for', '')
            }
        }

    except Exception as e:
        return {
            'filepath': filepath,
            'valid': False,
            'issues': [f'Error reading file: {str(e)}'],
            'metadata': {}
        }


def audit_directory_structure(base_path: str) -> Dict:
    """Audit directory structure for consistency."""
    issues = []

    # Check for consistent timestamp format in directory names
    dirs = glob.glob(os.path.join(base_path, '**'), recursive=True)
    timestamp_dirs = [d for d in dirs if os.path.isdir(d) and any(c.isdigit() for c in os.path.basename(d))]

    for dir_path in timestamp_dirs:
        dirname = os.path.basename(dir_path)
        # Should contain timestamp-like patterns
        if not ('T' in dirname or '-' in dirname):
            continue

        # Check for consistent timestamp format
        parts = dirname.split('T')
        if len(parts) == 2:
            date_part, time_part = parts
            if not (len(date_part) == 10 and len(time_part) >= 8):  # YYYY-MM-DDTHH:MM:SS
                issues.append(f'Inconsistent timestamp format in directory: {dirname}')

    return {
        'base_path': base_path,
        'total_directories': len([d for d in dirs if os.path.isdir(d)]),
        'timestamp_directories': len(timestamp_dirs),
        'issues': issues
    }


def find_all_result_files(base_path: str = 'results') -> List[str]:
    """Find all JSON result files in the results directory."""
    json_files = []

    # Common result file patterns
    patterns = [
        '**/metrics.json',
        '**/results.json',
        '**/index.json',
        '**/*_ci.json',
        '**/*_ablation.json',
        '**/multi_species_*.json',
        'ablation_study_*.json'
    ]

    for pattern in patterns:
        json_files.extend(glob.glob(os.path.join(base_path, pattern), recursive=True))

    # Also find JSON files in root results directories
    json_files.extend(glob.glob(os.path.join(base_path, '*.json')))

    return list(set(json_files))  # Remove duplicates


def generate_consistency_report() -> Dict:
    """Generate a comprehensive consistency report."""
    print("🔍 Auditing results consistency...")

    all_json_files = find_all_result_files()
    print(f"Found {len(all_json_files)} JSON result files to audit")

    audit_results = []
    valid_count = 0
    invalid_count = 0

    for filepath in all_json_files:
        result = audit_json_file(filepath)
        audit_results.append(result)

        if result['valid']:
            valid_count += 1
        else:
            invalid_count += 1

    # Audit directory structure
    dir_audit = audit_directory_structure('results')

    # Generate report
    report = {
        'timestamp': datetime.now().isoformat(),
        'created_by': 'joe knowles',
        'intended_for': 'peer_review',
        'audit_summary': {
            'total_files_audited': len(all_json_files),
            'valid_files': valid_count,
            'invalid_files': invalid_count,
            'validity_rate': f"{(valid_count / len(all_json_files) * 100):.1f}%" if all_json_files else "0%"
        },
        'directory_structure': dir_audit,
        'file_audits': audit_results,
        'recommendations': []
    }

    # Generate recommendations
    if invalid_count > 0:
        report['recommendations'].append("Fix invalid metadata in flagged files")
    if dir_audit['issues']:
        report['recommendations'].append("Standardize directory naming conventions")
    if not report['recommendations']:
        report['recommendations'].append("All files appear consistent! ✅")

    return report


def save_audit_report(report: Dict):
    """Save the audit report to file."""
    timestamp = datetime.now().isoformat(timespec='seconds')
    filename = f"results_audit_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"📄 Audit report saved to: {filename}")

    # Also generate a summary markdown
    md_filename = f"results_audit_{timestamp}.md"
    with open(md_filename, 'w') as f:
        f.write("# Results Consistency Audit Report\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**By:** joe knowles\n")
        f.write(f"**For:** peer_review\n\n")

        f.write("## Summary\n\n")
        summary = report['audit_summary']
        f.write(f"- **Total files audited:** {summary['total_files_audited']}\n")
        f.write(f"- **Valid files:** {summary['valid_files']}\n")
        f.write(f"- **Invalid files:** {summary['invalid_files']}\n")
        f.write(f"- **Validity rate:** {summary['validity_rate']}\n\n")

        if report['recommendations']:
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
            f.write("\n")

        if summary['invalid_files'] > 0:
            f.write("## Invalid Files\n\n")
            for audit in report['file_audits']:
                if not audit['valid']:
                    f.write(f"### {audit['filepath']}\n")
                    for issue in audit['issues']:
                        f.write(f"- ❌ {issue}\n")
                    f.write("\n")

    print(f"📄 Summary report saved to: {md_filename}")


def main():
    """Main audit function."""
    report = generate_consistency_report()
    save_audit_report(report)

    # Print summary
    summary = report['audit_summary']
    print("\n📊 AUDIT SUMMARY:")
    print(f"  Total files: {summary['total_files_audited']}")
    print(f"  Valid: {summary['valid_files']}")
    print(f"  Invalid: {summary['invalid_files']}")
    print(f"  Validity rate: {summary['validity_rate']}")

    if summary['invalid_files'] > 0:
        print("\n⚠️  Issues found! Check the detailed report for fixes needed.")
    else:
        print("\n✅ All results are properly timestamped and authored!")


if __name__ == '__main__':
    main()
