#!/usr/bin/env python3
"""
Generate controlled stimulus protocol and CSV template for validation experiments.
"""

import os
import datetime as _dt


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Generate stimulus protocol and CSV template')
    ap.add_argument('--out_dir', default='deliverables/stimulus_protocol')
    ap.add_argument('--species', default='Schizophyllum_commune')
    args = ap.parse_args()

    ts = _dt.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    out_dir = os.path.join(args.out_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    # Protocol markdown
    md_path = os.path.join(out_dir, 'protocol.md')
    with open(md_path, 'w') as f:
        f.write('# Controlled Stimulus Validation Protocol\n\n')
        f.write(f'- Created: {ts}\n')
        f.write(f'- Species: {args.species}\n')
        f.write('- Author: joe knowles\n\n')
        f.write('## Stimuli\n')
        f.write('- Moisture: 0.5 mL sterile water at t=+0 s (baseline 10 min, post 20 min)\n')
        f.write('- Light: 1000 lux LED for 60 s (baseline 10 min, post 20 min)\n')
        f.write('- Temperature: +2°C air pulse for 60 s (baseline 10 min, post 20 min)\n\n')
        f.write('## Recording\n')
        f.write('- Sampling rate: 1 Hz (match configs)\n')
        f.write('- Channels: differential pairs as available\n')
        f.write('- Environment: stable humidity/temperature\n\n')
        f.write('## CSV Template\n')
        f.write('- Columns: time_s,stimulus_type,intensity,notes\n')
        f.write('- time_s: seconds from recording start\n')
        f.write('- stimulus_type: moisture | light | temperature\n')
        f.write('- intensity: free text (e.g., 0.5ml, 1000lux, +2C)\n')

    # CSV template
    csv_path = os.path.join(out_dir, 'stimulus_events_template.csv')
    with open(csv_path, 'w') as f:
        f.write('time_s,stimulus_type,intensity,notes\n')
        f.write('600,moisture,0.5ml,after 10 min baseline\n')
        f.write('2400,light,1000lux,after 30 min\n')
        f.write('3600,temperature,+2C,after 1 h\n')

    print(out_dir)


if __name__ == '__main__':
    main()


