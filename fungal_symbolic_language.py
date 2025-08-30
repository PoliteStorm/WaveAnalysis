#!/usr/bin/env python3
"""
Fungal Symbolic Language (FSL)
A formal symbolic system for fungal computing algorithms

This creates a mathematical notation for expressing fungal computational processes,
similar to how lambda calculus or Boolean algebra formalize traditional computing.
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class FungalSymbolicLanguage:
    """
    Formal symbolic language for fungal computing.
    Defines operators and expressions for fungal computational processes.
    """

    def __init__(self):
        # Core fungal operators
        self.operators = {
            # Stimulus operators
            '⊕': 'stimulus_addition',      # ⊕(moisture, temperature)
            '⊗': 'stimulus_modulation',   # ⊗(light, intensity)
            '△': 'stimulus_threshold',     # △(chemical, threshold)

            # Rhythm operators
            '∿': 'rhythm_sine',           # ∿(frequency, phase, amplitude)
            '∿∿': 'rhythm_combination',    # ∿∿(fast∿, medium∿, slow∿)
            '⟲': 'rhythm_phase_reset',    # ⟲(spike_time)

            # Membrane operators
            '∇': 'membrane_gradient',     # ∇(potential, dt)
            '∫': 'membrane_integration',  # ∫(stimuli, rhythms, dt)
            'τ': 'membrane_decay',        # τ(potential, rate)

            # Spike operators
            '⚡': 'spike_generation',      # ⚡(probability, threshold)
            '⟨⟩': 'spike_amplitude',       # ⟨spike⟩(mean, std)
            '⌈⌉': 'spike_refractory',      # ⌈duration⌉(min, max, distribution)

            # Network operators
            '⟷': 'connection_bidirectional', # ⟷(neuron_i, neuron_j, strength)
            '⟶': 'connection_unidirectional', # ⟶(source, target, strength)
            '⊙': 'network_synchrony',     # ⊙(spikes, window)
            '∑': 'network_summation',     # ∑(neuron_outputs)

            # Logic operators (fungal style)
            '∧': 'fungal_and',            # ∧(input_a, input_b) - synchrony based
            '∨': 'fungal_or',             # ∨(input_a, input_b) - any spike
            '¬': 'fungal_not',            # ¬(input) - inverse synchrony
            '⊻': 'fungal_xor',            # ⊻(input_a, input_b) - alternating

            # Information operators
            'ℋ': 'entropy_calculation',   # ℋ(spike_train) - Shannon entropy
            '𝒟': 'distance_metric',       # 𝒟(train_a, train_b) - spike distance
            'ℙ': 'probability_distribution', # ℙ(events) - probability density

            # Time operators
            '∂': 'temporal_derivative',   # ∂(signal, dt)
            '∫': 'temporal_integral',     # ∫(signal, window)
            '⟨ ⟩': 'temporal_average',     # ⟨signal⟩(window)

            # Validation operators
            '✓': 'validation_check',      # ✓(hypothesis, data)
            'δ': 'effect_size',           # δ(pre_stats, post_stats)
            'ρ': 'correlation_coefficient', # ρ(signal_a, signal_b)
        }

    def create_fungal_expression(self, operation: str, *operands) -> str:
        """Create a symbolic expression for fungal computation."""
        if operation not in self.operators:
            raise ValueError(f"Unknown operator: {operation}")

        op_symbol = operation
        operands_str = ', '.join(str(op) for op in operands)
        return f"{op_symbol}({operands_str})"

    def compile_algorithm(self, algorithm_name: str) -> Dict:
        """Compile named fungal algorithms into symbolic expressions."""

        algorithms = {
            'fungal_neuron_update': self._neuron_update_algorithm(),
            'stimulus_response_validation': self._stimulus_response_algorithm(),
            'network_synchrony_logic': self._network_logic_algorithm(),
            'multi_scale_rhythm_processing': self._rhythm_processing_algorithm(),
            'information_entropy_computation': self._entropy_algorithm(),
        }

        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        return algorithms[algorithm_name]

    def _neuron_update_algorithm(self) -> Dict:
        """Symbolic representation of fungal neuron update algorithm."""
        return {
            'name': 'Fungal Neuron Update',
            'symbolic_form': """
            neuron_update(𝒩ᵢ, dt, 𝒮) ≔
                rhythm_signal ← ∿∿(∿(f_fast, φ_fast, A_fast),
                                 ∿(f_medium, φ_medium, A_medium),
                                 ∿(f_slow, φ_slow, A_slow))

                stimulus_input ← ∑_{s∈𝒮} s ⊗ intensity_s × δ_s
                membrane_potential ← ∫(∇(𝒩ᵢ.potential) + rhythm_signal + stimulus_input, dt)

                if refractory_period > 0:
                    membrane_potential ← τ(membrane_potential, 0.1)
                    refractory_period ← refractory_period - dt

                spike_probability ← P_spike × dt × (1 + |membrane_potential|)

                if random() < spike_probability ∧ refractory_period ≤ 0:
                    amplitude ← ⟨spike⟩(μ_amplitude, σ_amplitude)
                    refractory_duration ← ⌈duration⌉(ISI_min, ISI_max, lognormal)
                    ⟲(spike_time)
                    return amplitude
                else:
                    return ∅
            """,
            'description': 'Complete fungal neuron state update with multi-scale rhythms and stimulus integration'
        }

    def _stimulus_response_algorithm(self) -> Dict:
        """Symbolic representation of stimulus-response validation."""
        return {
            'name': 'Stimulus-Response Validation',
            'symbolic_form': """
            validate_stimulus_response(𝒱_signal, 𝒮_times, W_pre, W_post, f_s) ≔
                stimulus_indices ← round(𝒮_times × f_s)

                ∀ t_stim ∈ stimulus_indices:
                    pre_segment ← 𝒱_signal[t_stim - W_pre : t_stim]
                    post_segment ← 𝒱_signal[t_stim : t_stim + W_post]

                    pre_stats ← {μ_pre, σ_pre, median_pre, min_pre, max_pre}
                    post_stats ← {μ_post, σ_post, median_post, min_post, max_post}

                    t_stat, p_value ← ttest_ind(pre_segment, post_segment)
                    effect_size ← δ(pre_stats, post_stats)  # Cohen's d

                    significance ← p_value < α
                    detection ← effect_size ≥ threshold

                    response ← {
                        stimulus_time: t_stim / f_s,
                        pre_statistics: pre_stats,
                        post_statistics: post_stats,
                        statistical_tests: {t_stat, p_value, significance},
                        effect_size: {δ_value, interpretation},
                        signal_changes: {Δμ, Δmedian, Δσ, Δpercent},
                        detection_result: detection
                    }

                aggregate_results ← {
                    total_stimuli: |𝒮_times|,
                    detection_rate: |responses_detected| / |𝒮_times| × 100%,
                    mean_effect_size: ⟨δ_values⟩,
                    median_p_value: median(p_values),
                    validation_confidence: ✓(hypothesis, empirical_data)
                }

                return {individual_responses, aggregate_results}
            """,
            'description': 'Formal validation of fungal stimulus-response capabilities'
        }

    def _network_logic_algorithm(self) -> Dict:
        """Symbolic representation of fungal logic operations."""
        return {
            'name': 'Network Synchrony Logic',
            'symbolic_form': """
            fungal_AND(a, b) ≔
                stimuli ← {input_a: a, input_b: b}
                network_response ← simulate_network(𝒩, duration, stimuli)

                synchrony_index ← ⊙(spike_times, analysis_window)
                logic_output ← synchrony_index ≥ synchrony_threshold

                return logic_output

            fungal_OR(a, b) ≔
                stimuli ← {input_a: a, input_b: b}
                network_response ← simulate_network(𝒩, duration, stimuli)

                any_spike ← ∃ spike ∈ spike_times : spike.amplitude > threshold
                logic_output ← any_spike

                return logic_output

            fungal_XOR(a, b) ≔
                stimuli ← {input_a: a, input_b: b}
                network_response ← simulate_network(𝒩, duration, stimuli)

                spike_alternation ← |spike_count_a - spike_count_b| / (spike_count_a + spike_count_b)
                logic_output ← spike_alternation > alternation_threshold

                return logic_output

            fungal_NOT(input) ≔
                stimulus ← {input_signal: input}
                network_response ← simulate_network(𝒩, duration, stimulus)

                inverse_synchrony ← 1 - ⊙(spike_times, analysis_window)
                logic_output ← inverse_synchrony ≥ inverse_threshold

                return logic_output
            """,
            'description': 'Fungal logic operations based on network synchrony patterns'
        }

    def _rhythm_processing_algorithm(self) -> Dict:
        """Symbolic representation of multi-scale rhythm processing."""
        return {
            'name': 'Multi-Scale Rhythm Processing',
            'symbolic_form': """
            multi_scale_rhythm_processing(𝒱_signal, τ_scales, window_type) ≔
                rhythm_bands ← ∅

                ∀ τ ∈ τ_scales:
                    # Apply √t transform to each scale
                    u_domain ← √t_transform(𝒱_signal, τ)
                    spectral_power ← ℱ{u_domain}  # Fourier transform

                    # Extract band-specific features
                    band_power ← ∫ spectral_power[f_min : f_max] df
                    peak_frequency ← argmax(spectral_power)
                    bandwidth ← FWHM(spectral_power, peak_frequency)

                    rhythm_bands[τ] ← {
                        power: band_power,
                        centroid: peak_frequency,
                        bandwidth: bandwidth,
                        phase_coherence: ⟨|ℱ{u_domain}|⟩
                    }

                # Rhythm band interactions
                power_ratio_fast_medium ← rhythm_bands[τ_fast].power / rhythm_bands[τ_medium].power
                power_ratio_medium_slow ← rhythm_bands[τ_medium].power / rhythm_bands[τ_slow].power

                # Cross-band synchrony
                synchrony_matrix ← ∅
                ∀ i,j ∈ rhythm_bands:
                    phase_diff ← ∠(rhythm_bands[i].phase) - ∠(rhythm_bands[j].phase)
                    synchrony_matrix[i,j] ← ⟨cos(phase_diff)⟩

                # Rhythm-based features for classification
                rhythm_features ← {
                    band_powers: [rhythm_bands[τ].power ∀ τ],
                    power_ratios: [power_ratio_fast_medium, power_ratio_medium_slow],
                    synchrony_patterns: synchrony_matrix,
                    dominant_scale: argmax(rhythm_bands[τ].power ∀ τ)
                }

                return rhythm_features
            """,
            'description': 'Multi-scale temporal rhythm processing and analysis'
        }

    def _entropy_algorithm(self) -> Dict:
        """Symbolic representation of information entropy computation."""
        return {
            'name': 'Information Entropy Computation',
            'symbolic_form': """
            fungal_entropy_computation(spike_train, time_window, bin_size) ≔
                # Temporal binning
                time_bins ← partition([0, time_window], bin_size)
                spike_counts ← [count_spikes_in_bin(spike_train, bin) ∀ bin ∈ time_bins]

                # Probability distribution
                total_spikes ← ∑ spike_counts
                probabilities ← spike_counts / total_spikes
                probabilities ← probabilities[probabilities > 0]  # Remove zeros

                # Shannon entropy calculation
                ℋ ← -∑ probabilities × log₂(probabilities)

                # Normalized entropy (0-1 scale)
                ℋ_normalized ← ℋ / log₂(|probabilities|)
                ℋ_normalized ← ℋ_normalized if not isnan(ℋ_normalized) else 0

                # Entropy rate (bits per second)
                ℋ_rate ← ℋ / time_window

                # Conditional entropy for sequential patterns
                ℋ_conditional ← ∅
                for lag ∈ [1, 2, 3, ..., max_lag]:
                    joint_probabilities ← compute_joint_probabilities(spike_counts, lag)
                    ℋ_conditional[lag] ← ℋ(joint_probabilities) - ℋ(marginal_probabilities)

                # Mutual information
                ℋ_mutual ← ℋ(spike_counts) + ℋ(reference_signal) - ℋ(joint_spike_reference)

                entropy_metrics ← {
                    shannon_entropy: ℋ,
                    normalized_entropy: ℋ_normalized,
                    entropy_rate: ℋ_rate,
                    conditional_entropy: ℋ_conditional,
                    mutual_information: ℋ_mutual,
                    predictability: 1 - ℋ_normalized
                }

                return entropy_metrics
            """,
            'description': 'Information-theoretic analysis of fungal spiking patterns'
        }

    def translate_to_mathematical_notation(self, algorithm_name: str) -> str:
        """Translate fungal algorithm to standard mathematical notation."""
        algorithm = self.compile_algorithm(algorithm_name)

        # Translation mappings
        translations = {
            '⊕': '+',
            '⊗': '×',
            '△': 'θ',
            '∿': 'sin',
            '∿∿': '∑',
            '⟲': 'reset',
            '∇': 'd/dt',
            '∫': '∫',
            'τ': 'exp(-t/τ)',
            '⚡': 'spike',
            '⟨⟩': 'N',
            '⌈⌉': 'lognormal',
            '⟷': '↔',
            '⟶': '→',
            '⊙': 'sync',
            '∑': '∑',
            '∧': '∧',
            '∨': '∨',
            '¬': '¬',
            '⊻': '⊕',
            'ℋ': 'H',
            '𝒟': 'd',
            'ℙ': 'P',
            '∂': '∂',
            '⟨ ⟩': '〈 〉',
            '✓': 'valid',
            'δ': 'd',
            'ρ': 'ρ'
        }

        mathematical_form = algorithm['symbolic_form']
        for fungal_symbol, math_symbol in translations.items():
            mathematical_form = mathematical_form.replace(fungal_symbol, math_symbol)

        return mathematical_form

    def create_fungal_programming_example(self) -> str:
        """Create an example fungal computing program in symbolic language."""
        return """
        # Fungal Computing Program: Environmental Monitor
        # ===============================================

        program EnvironmentalMonitor ≔
            # Initialize fungal network
            network ← create_network(n_neurons=20, connectivity=0.3)

            # Define sensor thresholds (based on empirical data)
            thresholds ≔ {
                moisture: △(moisture, 0.1),
                temperature: △(temperature, 0.05),
                chemical: △(chemical, 0.15)
            }

            # Main monitoring loop
            while monitoring_active:
                # Read environmental stimuli
                stimuli ← read_sensors()

                # Process through fungal network
                ∀ stimulus ∈ stimuli:
                    if stimulus.intensity > thresholds[stimulus.type]:
                        response ← network.simulate(duration=300, stimuli={stimulus})
                        entropy ← ℋ(response.spike_train)
                        synchrony ← ⊙(response.spikes, window=100)

                        # Alert system
                        if entropy > 2.0 ∧ synchrony > 0.7:
                            alert ← generate_alert(stimulus, entropy, synchrony)
                            output ← alert

                # Adaptive learning
                network ← update_connections(network, stimuli, responses)

        # Example execution
        monitor ← EnvironmentalMonitor()
        result ← monitor.run(duration=3600)  # 1 hour monitoring
        """


def demonstrate_fungal_symbolic_language():
    """Demonstrate the fungal symbolic language capabilities."""
    print("🧬 Fungal Symbolic Language (FSL) Demonstration")
    print("=" * 60)

    fsl = FungalSymbolicLanguage()

    # Show core operators
    print("\n1. Core FSL Operators:")
    print("   Stimulus: ⊕ ⊗ △")
    print("   Rhythm: ∿ ∿∿ ⟲")
    print("   Membrane: ∇ ∫ τ")
    print("   Spike: ⚡ ⟨⟩ ⌈⌉")
    print("   Network: ⟷ ⟶ ⊙ ∑")
    print("   Logic: ∧ ∨ ¬ ⊻")
    print("   Information: ℋ 𝒟 ℙ")
    print("   Time: ∂ ∫ ⟨ ⟩")
    print("   Validation: ✓ δ ρ")

    # Example expressions
    print("\n2. Example FSL Expressions:")
    examples = [
        ("stimulus_combination", fsl.create_fungal_expression('⊕', 'moisture', 'temperature')),
        ("rhythm_generation", fsl.create_fungal_expression('∿∿', 'fast∿', 'medium∿', 'slow∿')),
        ("spike_probability", fsl.create_fungal_expression('⚡', 'P_spike', 'threshold')),
        ("network_connection", fsl.create_fungal_expression('⟷', 'neuron_i', 'neuron_j', 'strength')),
        ("logic_and", fsl.create_fungal_expression('∧', 'input_a', 'input_b')),
        ("entropy_calculation", fsl.create_fungal_expression('ℋ', 'spike_train')),
        ("effect_size", fsl.create_fungal_expression('δ', 'pre_stats', 'post_stats')),
    ]

    for name, expression in examples:
        print(f"   {name}: {expression}")

    # Show key algorithms
    print("\n3. Key Fungal Algorithms:")

    algorithms_to_show = [
        'fungal_neuron_update',
        'stimulus_response_validation',
        'network_synchrony_logic'
    ]

    for alg_name in algorithms_to_show:
        print(f"\n   📋 {alg_name.upper().replace('_', ' ')}:")
        alg = fsl.compile_algorithm(alg_name)
        print(f"   Description: {alg['description']}")

        # Show first few lines of symbolic form
        symbolic_lines = alg['symbolic_form'].strip().split('\n')[:5]
        for line in symbolic_lines:
            if line.strip():
                print(f"   {line.strip()}")

    # Show mathematical translation
    print("\n4. Mathematical Translation:")
    math_notation = fsl.translate_to_mathematical_notation('fungal_neuron_update')
    math_lines = math_notation.strip().split('\n')[:6]
    print("   neuron_update(N_i, dt, S) ≔")
    for line in math_lines[1:6]:
        if line.strip():
            print(f"   {line.strip()}")

    # Show programming example
    print("\n5. Example Fungal Program:")
    program_lines = fsl.create_fungal_programming_example().strip().split('\n')[:15]
    for line in program_lines:
        if line.strip():
            print(f"   {line.strip()}")

    print("\n" + "=" * 60)
    print("🎉 FSL Demonstration Complete!")
    print("FSL provides a formal symbolic system for fungal computing,")
    print("bridging biological processes with mathematical formalism.")


if __name__ == '__main__':
    demonstrate_fungal_symbolic_language()
