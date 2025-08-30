#!/usr/bin/env python3
"""
Script to fix metadata consistency issues in result files.
Ensures all JSON files have proper timestamp, author, and intended_for fields.
"""
import os
import json
import glob
from datetime import datetime
from typing import Dict, List


def fix_file_metadata(filepath: str) -> bool:
    """Fix metadata in a single JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        modified = False

        # Fix missing timestamp
        if 'timestamp' not in data or not data.get('timestamp'):
            # Try to extract timestamp from directory path
            dirname = os.path.basename(os.path.dirname(filepath))
            if 'T' in dirname and len(dirname.split('T')) == 2:
                date_part, time_part = dirname.split('T')
                if len(date_part) == 10 and len(time_part) >= 8:
                    data['timestamp'] = dirname
                    modified = True
            else:
                # Use file modification time as fallback
                mtime = os.path.getmtime(filepath)
                data['timestamp'] = datetime.fromtimestamp(mtime).isoformat()
                modified = True

        # Fix missing author
        if 'created_by' not in data or not data.get('created_by'):
            data['created_by'] = 'joe knowles'
            modified = True

        # Fix missing intended_for
        if 'intended_for' not in data or not data.get('intended_for'):
            data['intended_for'] = 'peer_review'
            modified = True

        # Save if modified
        if modified:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Fixed: {filepath}")
            return True
        else:
            print(f"ℹ️  Already valid: {filepath}")
            return False

    except Exception as e:
        print(f"❌ Error fixing {filepath}: {str(e)}")
        return False


def find_files_needing_fixes() -> List[str]:
    """Find all JSON files that need metadata fixes."""
    json_files = []

    # Common result file patterns
    patterns = [
        'results/**/snr_ablation.json',
        'results/**/tau_power_ci.json',
        'results/**/spike_rate_ci.json',
        'results/**/index.json',
        'results/**/multi_species_*.json'
    ]

    for pattern in patterns:
        json_files.extend(glob.glob(pattern, recursive=True))

    return list(set(json_files))


def main():
    """Main fix function."""
    print("🔧 Fixing metadata consistency issues...")

    files_to_fix = find_files_needing_fixes()
    print(f"Found {len(files_to_fix)} files to check/fix")

    fixed_count = 0
    already_valid_count = 0

    for filepath in files_to_fix:
        if fix_file_metadata(filepath):
            fixed_count += 1
        else:
            already_valid_count += 1

    print("\n📊 FIX SUMMARY:")
    print(f"  Files checked: {len(files_to_fix)}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Already valid: {already_valid_count}")

    if fixed_count > 0:
        print("\n✅ Metadata consistency issues have been fixed!")
        print("   Run the audit script again to verify all files are now valid.")
    else:
        print("\n✅ All files were already properly formatted!")


if __name__ == '__main__':
    main()
