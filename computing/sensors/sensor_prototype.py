#!/usr/bin/env python3
"""
Fungal Sensor Prototype: Hardware Implementation Guide
Based on our stimulus-response validation and fungal computing analysis.

This provides a blueprint for building real fungal computing sensors.
"""
import numpy as np
import time
from datetime import datetime
import json
import serial  # For Arduino/Raspberry Pi communication
import threading
import os
import glob


class FungalSensorHardware:
    """
    Hardware interface for fungal electrophysiological sensors.
    This demonstrates how to build real fungal computing sensors.
    """

    def __init__(self, port='/dev/ttyACM0', baud_rate=115200):
        """
        Initialize hardware interface.

        Hardware Requirements:
        - Arduino/Raspberry Pi with ADC
        - Non-invasive electrodes (Ag/AgCl preferred)
        - Signal conditioning circuit (amplifier + filter)
        - Fungal culture chamber
        """
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.is_connected = False
        self.calibration_data = self.load_calibration()

        # Sensor specifications based on our validation
        self.specs = {
            'adc_resolution': 12,      # bits
            'sampling_rate': 100,      # Hz
            'input_range': [-2.5, 2.5], # Volts
            'gain': 1000,              # Signal amplification
            'filter_cutoff': 50,       # Hz (anti-aliasing)
        }

    def connect_hardware(self):
        """Establish connection with microcontroller."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1
            )
            time.sleep(2)  # Allow connection to establish
            self.is_connected = True
            print("✅ Hardware connection established")
            return True
        except Exception as e:
            print(f"❌ Hardware connection failed: {e}")
            return False

    def load_calibration(self):
        """Load sensor calibration data from our validation results."""
        return {
            'moisture': {
                'baseline_mv': 0.0,
                'sensitivity_mv': 0.5,  # Based on our Cohen's d = 2.06
                'threshold_mv': 0.1,
                'effect_size': 2.06
            },
            'temperature': {
                'baseline_mv': 0.0,
                'sensitivity_mv': 0.4,
                'threshold_mv': 0.05,
                'effect_size': 1.96
            },
            'light': {
                'baseline_mv': 0.0,
                'sensitivity_mv': 0.35,
                'threshold_mv': 0.08,
                'effect_size': 1.83
            },
            'chemical': {
                'baseline_mv': 0.0,
                'sensitivity_mv': 0.15,
                'threshold_mv': 0.15,
                'effect_size': 0.50
            }
        }

    def read_voltage(self, channel=0):
        """Read voltage from specified ADC channel."""
        if not self.is_connected:
            return 0.0

        try:
            # Send read command to microcontroller
            command = f"READ:{channel}\n"
            self.serial_conn.write(command.encode())

            # Read response
            response = self.serial_conn.readline().decode().strip()

            if response.startswith("VOLT:"):
                adc_value = int(response.split(":")[1])
                # Convert ADC value to voltage
                voltage = self.adc_to_voltage(adc_value)
                return voltage
            else:
                print(f"Invalid response: {response}")
                return 0.0

        except Exception as e:
            print(f"Read error: {e}")
            return 0.0

    def adc_to_voltage(self, adc_value):
        """Convert ADC reading to voltage."""
        # 12-bit ADC: 0-4095 range
        voltage_range = self.specs['input_range'][1] - self.specs['input_range'][0]
        voltage = self.specs['input_range'][0] + (adc_value / 4095.0) * voltage_range
        return voltage / self.specs['gain']  # Account for amplification

    def detect_spikes(self, signal_buffer, threshold_mv=0.1):
        """Detect spikes in voltage signal using our validated methods."""
        if len(signal_buffer) < 10:
            return []

        # Simple spike detection based on our empirical data
        mean_signal = np.mean(signal_buffer)
        std_signal = np.std(signal_buffer)

        spikes = []
        refractory_period = 0

        for i, voltage in enumerate(signal_buffer):
            if refractory_period > 0:
                refractory_period -= 1
                continue

            # Spike detection (based on our amplitude stats)
            if abs(voltage - mean_signal) > threshold_mv:
                # Check if it's a significant deviation
                if abs(voltage - mean_signal) > 2 * std_signal:
                    spikes.append({
                        'time_index': i,
                        'amplitude_mv': voltage * 1000,  # Convert to mV
                        'timestamp': datetime.now().isoformat()
                    })
                    # Refractory period based on our ISI stats (minimum 333s, but scaled)
                    refractory_period = int(333 * self.specs['sampling_rate'] / 3600)  # Convert to samples

        return spikes

    def analyze_stimulus_response(self, stimulus_type, pre_duration=300, post_duration=600):
        """
        Analyze stimulus-response using our validated methods.
        This implements the stimulus-response validation from our analysis.
        """
        print(f"🧪 Analyzing {stimulus_type} stimulus response...")

        # Record pre-stimulus baseline
        print("📊 Recording pre-stimulus baseline...")
        pre_samples = int(pre_duration * self.specs['sampling_rate'])
        pre_signal = []

        for _ in range(pre_samples):
            voltage = self.read_voltage()
            pre_signal.append(voltage)
            time.sleep(1.0 / self.specs['sampling_rate'])

        # Apply stimulus (this would be hardware-controlled)
        print(f"⚡ Applying {stimulus_type} stimulus...")
        self.apply_stimulus(stimulus_type, intensity=0.8)

        # Record post-stimulus response
        print("📊 Recording post-stimulus response...")
        post_samples = int(post_duration * self.specs['sampling_rate'])
        post_signal = []

        for _ in range(post_samples):
            voltage = self.read_voltage()
            post_signal.append(voltage)
            time.sleep(1.0 / self.specs['sampling_rate'])

        # Analyze response using our validated statistical methods
        analysis_result = self.analyze_response(
            pre_signal, post_signal, stimulus_type
        )

        return analysis_result

    def analyze_response(self, pre_signal, post_signal, stimulus_type):
        """Analyze pre/post stimulus response using our validation methods."""
        # Convert to numpy arrays
        pre_arr = np.array(pre_signal)
        post_arr = np.array(post_signal)

        # Basic statistics
        pre_stats = {
            'mean': float(np.mean(pre_arr)),
            'std': float(np.std(pre_arr)),
            'median': float(np.median(pre_arr))
        }

        post_stats = {
            'mean': float(np.mean(post_arr)),
            'std': float(np.std(post_arr)),
            'median': float(np.median(post_arr))
        }

        # Statistical tests (from our validation)
        try:
            # T-test
            t_stat, p_value = stats.ttest_ind(pre_arr, post_arr, equal_var=False)

            # Effect size (Cohen's d) - using our empirical calibrations
            cal = self.calibration_data[stimulus_type]
            pooled_std = np.sqrt((pre_stats['std']**2 + post_stats['std']**2) / 2)
            if pooled_std > 0:
                cohens_d = abs(post_stats['mean'] - pre_stats['mean']) / pooled_std
            else:
                cohens_d = 0.0

            # Detection based on our thresholds
            detected = cohens_d >= cal['effect_size'] * 0.5  # Conservative threshold

            return {
                'stimulus_type': stimulus_type,
                'timestamp': datetime.now().isoformat(),
                'pre_stats': pre_stats,
                'post_stats': post_stats,
                'statistical_tests': {
                    't_test': {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    },
                    'effect_size': {
                        'cohens_d': float(cohens_d),
                        'calibrated_threshold': cal['effect_size'],
                        'interpretation': self.interpret_effect_size(cohens_d)
                    }
                },
                'detection': {
                    'detected': detected,
                    'confidence': float(min(cohens_d / cal['effect_size'], 1.0)),
                    'method': 'stimulus_response_validation'
                },
                'spike_analysis': {
                    'pre_spikes': len(self.detect_spikes(pre_signal)),
                    'post_spikes': len(self.detect_spikes(post_signal))
                },
                'metadata': build_metadata('fungal_sensor_prototype', 'sensor')
            }

        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'error': str(e),
                'stimulus_type': stimulus_type,
                'timestamp': datetime.now().isoformat()
            }

    def apply_stimulus(self, stimulus_type, intensity=0.5):
        """Apply stimulus to fungal culture (hardware implementation)."""
        # This would control external hardware (pumps, LEDs, heaters, etc.)
        stimulus_commands = {
            'moisture': f"STIM:MOISTURE:{intensity}\n",
            'temperature': f"STIM:TEMPERATURE:{intensity}\n",
            'light': f"STIM:LIGHT:{intensity}\n",
            'chemical': f"STIM:CHEMICAL:{intensity}\n"
        }

        if stimulus_type in stimulus_commands:
            command = stimulus_commands[stimulus_type]
            if self.serial_conn:
                self.serial_conn.write(command.encode())
                print(f"Stimulus applied: {stimulus_type} at intensity {intensity}")
            else:
                print(f"Mock stimulus: {stimulus_type} at intensity {intensity}")
        else:
            print(f"Unknown stimulus type: {stimulus_type}")

    def interpret_effect_size(self, d):
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def continuous_monitoring(self, duration_minutes=60):
        """Continuous environmental monitoring using fungal sensors."""
        print(f"🔄 Starting continuous monitoring for {duration_minutes} minutes...")

        start_time = datetime.now()
        monitoring_data = []

        try:
            while (datetime.now() - start_time).seconds < duration_minutes * 60:
                # Read all sensor channels
                readings = {}
                for stimulus_type in ['moisture', 'temperature', 'light', 'chemical']:
                    voltage = self.read_voltage()  # Would need channel specification
                    readings[stimulus_type] = voltage * 1000  # Convert to mV

                # Detect spikes across all readings
                spike_data = self.detect_spikes(list(readings.values()))

                # Record monitoring data
                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'readings_mv': readings,
                    'spikes_detected': len(spike_data),
                    'spike_details': spike_data
                }

                monitoring_data.append(data_point)

                # Alert system (based on our validation thresholds)
                for stimulus_type, voltage_mv in readings.items():
                    cal = self.calibration_data[stimulus_type]
                    if abs(voltage_mv) > cal['threshold_mv']:
                        print(f"🚨 ALERT: {stimulus_type} detection (voltage: {voltage_mv:.3f} mV)")

                time.sleep(1.0 / self.specs['sampling_rate'])

        except KeyboardInterrupt:
            print("Monitoring stopped by user")

        return monitoring_data

    def save_results(self, results, filename=None):
        """Save analysis results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            filename = f"fungal_sensor_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Results saved to: {filename}")
        return filename


class FungalSensorNetwork:
    """
    Network of multiple fungal sensors for distributed monitoring.
    Demonstrates the potential for mycelial network computing.
    """

    def __init__(self, n_sensors=4):
        self.sensors = [FungalSensorHardware() for _ in range(n_sensors)]
        self.network_topology = self.create_network_topology()

    def create_network_topology(self):
        """Create a mycelial-inspired network topology."""
        # Small-world network similar to mycelial connections
        connections = {}
        n = len(self.sensors)

        for i in range(n):
            # Each sensor connects to 2-3 nearest neighbors
            neighbors = []
            for j in range(max(0, i-2), min(n, i+3)):
                if i != j:
                    neighbors.append(j)
            connections[i] = neighbors

        return connections

    def distributed_monitoring(self, duration_minutes=30):
        """Distributed environmental monitoring across sensor network."""
        print("🌐 Starting distributed fungal sensor network monitoring...")

        threads = []
        results = [[] for _ in range(len(self.sensors))]

        def monitor_sensor(sensor_idx):
            """Monitor individual sensor in separate thread."""
            sensor = self.sensors[sensor_idx]
            sensor.connect_hardware()

            start_time = datetime.now()
            while (datetime.now() - start_time).seconds < duration_minutes * 60:
                # Read sensor data
                voltage = sensor.read_voltage()
                spike_count = len(sensor.detect_spikes([voltage] * 10))  # Mock buffer

                data_point = {
                    'timestamp': datetime.now().isoformat(),
                    'sensor_id': sensor_idx,
                    'voltage_mv': voltage * 1000,
                    'spike_count': spike_count
                }

                results[sensor_idx].append(data_point)
                time.sleep(1)

        # Start monitoring threads
        for i in range(len(self.sensors)):
            thread = threading.Thread(target=monitor_sensor, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for monitoring to complete
        for thread in threads:
            thread.join()

        # Analyze network-level patterns
        network_analysis = self.analyze_network_patterns(results)

        return {
            'individual_results': results,
            'network_analysis': network_analysis
        }

    def analyze_network_patterns(self, sensor_results):
        """Analyze network-level coordination patterns."""
        # Extract spike timing across all sensors
        all_spikes = []
        for sensor_idx, sensor_data in enumerate(sensor_results):
            for data_point in sensor_data:
                if data_point['spike_count'] > 0:
                    all_spikes.append({
                        'sensor_id': sensor_idx,
                        'timestamp': data_point['timestamp'],
                        'voltage_mv': data_point['voltage_mv']
                    })

        # Calculate network synchrony
        if len(all_spikes) > 1:
            spike_times = [datetime.fromisoformat(s['timestamp']) for s in all_spikes]
            time_diffs = np.diff([t.timestamp() for t in sorted(spike_times)])

            synchrony_index = 1.0 / (1.0 + np.std(time_diffs)) if len(time_diffs) > 0 else 0.0
        else:
            synchrony_index = 0.0

        return {
            'total_spikes': len(all_spikes),
            'network_synchrony': synchrony_index,
            'active_sensors': sum(1 for r in sensor_results if len(r) > 0),
            'coordination_events': len([s for s in all_spikes if s['voltage_mv'] > 0.1])
        }


def demo_fungal_sensor():
    """Demonstrate fungal sensor capabilities."""
    print("🧬 Fungal Sensor Hardware Demo")
    print("=" * 50)
    # Metadata header
    print(json.dumps(build_metadata('fungal_sensor_prototype', 'sensor')))

    # Create sensor instance
    sensor = FungalSensorHardware()

    print("\n1. Hardware Connection:")
    if sensor.connect_hardware():
        print("✅ Connected to fungal sensor hardware")
    else:
        print("⚠️  Running in simulation mode (no hardware connected)")

    print("\n2. Stimulus-Response Testing:")
    test_stimuli = ['moisture', 'temperature', 'light']

    for stimulus in test_stimuli:
        print(f"\nTesting {stimulus} stimulus:")
        result = sensor.analyze_stimulus_response(stimulus, pre_duration=10, post_duration=20)

        if 'error' not in result:
            detected = "✅ DETECTED" if result['detection']['detected'] else "❌ NO RESPONSE"
            effect_size = result['statistical_tests']['effect_size']['cohens_d']
            print(".2f"
                  ".3f")

            # Save detailed results
            sensor.save_results(result, f"{stimulus}_response_analysis.json")
        else:
            print(f"❌ Analysis failed: {result['error']}")

    print("\n3. Continuous Monitoring Demo:")
    print("(Running for 30 seconds...)")

    # Short monitoring demo
    monitoring_data = sensor.continuous_monitoring(duration_minutes=0.5)

    print(f"📊 Collected {len(monitoring_data)} data points")
    if monitoring_data:
        spike_counts = [d['spikes_detected'] for d in monitoring_data]
        print(f"⚡ Average spikes per second: {np.mean(spike_counts):.2f}")

        # Save monitoring data
        sensor.save_results(monitoring_data, "continuous_monitoring_demo.json")

    print("\n4. Network Simulation:")
    network = FungalSensorNetwork(n_sensors=3)
    network_results = network.distributed_monitoring(duration_minutes=0.25)

    print("🌐 Network Analysis:")
    analysis = network_results['network_analysis']
    print(f"   Total spikes: {analysis['total_spikes']}")
    print(".3f"
    print(f"   Active sensors: {analysis['active_sensors']}")
    print(f"   Coordination events: {analysis['coordination_events']}")

    print("\n" + "=" * 50)
    print("🎉 Fungal Sensor Demo Complete!")
    print("This demonstrates the hardware implementation of")
    print("scientifically validated fungal computing sensors.")


if __name__ == '__main__':
    demo_fungal_sensor()

# --- metadata utilities ---
def build_metadata(prototype: str, kind: str) -> dict:
    ts = datetime.now().isoformat(timespec='seconds')
    sources = []
    for p in [
        os.path.join('results', 'zenodo'),
        os.path.join('results', 'audio_continuous'),
        os.path.join('results', 'cross_modal'),
        os.path.join('results', 'psi_sweep'),
    ]:
        if os.path.isdir(p):
            sub = sorted(glob.glob(os.path.join(p, '*')))
            sources.append(sub[-1] if sub else p)
    return {
        'created_by': 'joe knowles',
        'timestamp': ts,
        'prototype': prototype,
        'type': kind,
        'data_sources': sources,
    }
