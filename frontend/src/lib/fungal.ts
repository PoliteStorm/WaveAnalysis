/*
Fungal computing simulation models in TypeScript, adapted from the Python
implementations in `fungal_computing_simulator.py`.
*/

export type StimulusMap = Record<string, number>;

class SeededRandom {
  private state: number;

  constructor(seed: number = 123456789) {
    // Xorshift32
    this.state = seed >>> 0;
    if (this.state === 0) {
      this.state = 0xdeadbeef;
    }
  }

  next(): number {
    // Returns float in [0,1)
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return (this.state & 0xffffffff) / 0x100000000;
  }

  normal(mean = 0, std = 1): number {
    // Box-Muller transform
    const u1 = Math.max(this.next(), 1e-12);
    const u2 = Math.max(this.next(), 1e-12);
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * std;
  }

  logNormal(mu: number, sigma: number): number {
    return Math.exp(this.normal(mu, sigma));
  }
}

function hashString(input: string): number {
  // Simple FNV-1a hash
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

export class FungalNeuronTS {
  private species: string;
  private rng: SeededRandom;
  private speciesFactor: number = 1.0;

  private spikeStats = {
    meanIsiS: 3540.0,
    stdIsiS: 3658.7,
    minIsiS: 333.0,
    maxIsiS: 11429.0,
    meanAmplitude: 0.19,
    stdAmplitude: 0.45,
    spikeProbabilityPerS: 0.001,
  };

  private rhythmBands = {
    fast: { freqHz: 1 / 5.5, power: 0.22 },
    medium: { freqHz: 1 / 24.5, power: 0.20 },
    slow: { freqHz: 1 / 104.0, power: 0.58 },
  };

  private membranePotential = 0.0;
  private timeSinceLastSpikeS = 0.0;
  private refractoryPeriodS = 0.0;

  constructor(species: string = "schizophyllum_commune", seed: number = 42) {
    this.species = species;
    const h = hashString(this.species);
    this.speciesFactor = 1.0 + ((h % 1000) / 10000);
    this.rng = new SeededRandom((seed ^ h) >>> 0);
  }

  update(dtS: number, stimuli?: StimulusMap): number | null {
    const stim = stimuli ?? {};

    // Multi-scale rhythm
    let rhythmSignal = 0.0;
    this.timeSinceLastSpikeS += dtS;
    for (const band of Object.values(this.rhythmBands)) {
      const phase = 2 * Math.PI * band.freqHz * this.timeSinceLastSpikeS;
      rhythmSignal += band.power * Math.sin(phase);
    }

    // Stimulus integration
    let stimulusInput = 0.0;
    for (const [stimulusType, intensity] of Object.entries(stim)) {
      if (stimulusType === "moisture") stimulusInput += intensity * 2.06;
      else if (stimulusType === "temperature") stimulusInput += intensity * 1.96;
      else if (stimulusType === "light") stimulusInput += intensity * 1.83;
      else if (stimulusType === "chemical") stimulusInput += intensity * 0.50;
      else if (stimulusType === "mechanical") stimulusInput += intensity * 0.65;
      else if (stimulusType === "network") stimulusInput += intensity;
    }

    this.membranePotential += rhythmSignal * dtS;
    this.membranePotential += stimulusInput * dtS;

    if (this.refractoryPeriodS > 0) {
      this.refractoryPeriodS -= dtS;
      this.membranePotential *= 0.1;
    }

    let spikeProbability = this.spikeStats.spikeProbabilityPerS * dtS * this.speciesFactor;
    spikeProbability *= 1 + Math.abs(this.membranePotential);

    if (this.rng.next() < spikeProbability && this.refractoryPeriodS <= 0) {
      const amplitude = this.rng.normal(
        this.spikeStats.meanAmplitude,
        this.spikeStats.stdAmplitude,
      );

      let refractoryDuration = this.rng.logNormal(
        Math.log(this.spikeStats.meanIsiS),
        Math.log(1 + this.spikeStats.stdIsiS / this.spikeStats.meanIsiS),
      );
      refractoryDuration = Math.min(
        Math.max(refractoryDuration, this.spikeStats.minIsiS),
        this.spikeStats.maxIsiS,
      );

      this.refractoryPeriodS = refractoryDuration;
      this.timeSinceLastSpikeS = 0.0;
      return amplitude;
    }

    return null;
  }
}

export interface NetworkSimResult {
  spikeTimes: Array<{ t: number; neuronId: number; amplitude: number }>;
  networkActivity: number[];
  synchrony: number[];
  spikeRatePerS: number;
  networkEntropyBits: number;
  timePoints: number[];
  connections: number;
}

export class FungalNetworkTS {
  private neurons: FungalNeuronTS[] = [];
  private connectivity: number;
  // Key format: `${i}->${j}`
  private connections: Map<string, number> = new Map();

  constructor(nNeurons: number = 10, connectivity: number = 0.3) {
    // Store connectivity for reference
    this.connectivity = connectivity;
    for (let i = 0; i < nNeurons; i++) {
      this.neurons.push(new FungalNeuronTS(`neuron_${i}`, i));
    }
    // Use stored connectivity to determine link probability and weight scaling
    for (let i = 0; i < nNeurons; i++) {
      for (let j = i + 1; j < nNeurons; j++) {
        if (Math.random() < this.connectivity) {
          const strength = -Math.log(Math.max(1 - Math.random(), 1e-9)) * 0.5;
          this.connections.set(`${i}->${j}`, strength);
          this.connections.set(`${j}->${i}`, strength);
        }
      }
    }
  }

  simulate(durationS: number, dtS: number = 1.0, stimuli?: Record<number, StimulusMap>): NetworkSimResult {
    const stimByTime = stimuli ?? {};
    const timePoints: number[] = [];
    const spikeTimes: Array<{ t: number; neuronId: number; amplitude: number }> = [];
    const networkActivity: number[] = [];
    const synchrony: number[] = [];

    let t = 0.0;
    while (t < durationS - 1e-9) {
      timePoints.push(t);
      const currentStim = stimByTime[t] ?? {};
      const spikesThisStep: Array<{ neuronId: number; amplitude: number }> = [];

      for (let i = 0; i < this.neurons.length; i++) {
        let networkInput = 0.0;
        for (const [key, strength] of this.connections) {
          const [, dst] = key.split("->").map((x) => parseInt(x, 10));
          if (dst === i) {
            // Very simple exponentially decaying influence if any recent spike happened
            // We approximate by using network activity at previous step later; here we just sum strength
            networkInput += 0.01 * strength;
          }
        }
        const combinedStim: StimulusMap = { ...currentStim };
        if (networkInput !== 0) combinedStim.network = (combinedStim.network ?? 0) + networkInput;
        const spike = this.neurons[i].update(dtS, combinedStim);
        if (spike !== null) {
          spikesThisStep.push({ neuronId: i, amplitude: spike });
        }
      }

      networkActivity.push(spikesThisStep.length / Math.max(1, this.neurons.length));
      for (const s of spikesThisStep) {
        spikeTimes.push({ t, neuronId: s.neuronId, amplitude: s.amplitude });
      }
      if (spikesThisStep.length > 1) {
        synchrony.push(spikesThisStep.length / Math.max(1, this.neurons.length));
      } else {
        synchrony.push(0);
      }
      t += dtS;
    }

    const spikeRatePerS = spikeTimes.length / Math.max(1e-9, durationS);
    const networkEntropyBits = this.computeEntropy(spikeTimes, durationS);

    return {
      spikeTimes,
      networkActivity,
      synchrony,
      spikeRatePerS,
      networkEntropyBits,
      timePoints,
      connections: this.connections.size,
    };
  }

  private computeEntropy(spikeTimes: Array<{ t: number }>, durationS: number): number {
    if (spikeTimes.length === 0) return 0.0;
    const nBins = Math.max(10, Math.floor(durationS / 10));
    const counts = new Array<number>(nBins).fill(0);
    for (const s of spikeTimes) {
      let idx = Math.floor((s.t / durationS) * nBins);
      if (idx >= nBins) idx = nBins - 1;
      counts[idx] += 1;
    }
    const sum = counts.reduce((a, b) => a + b, 0);
    if (sum <= 0) return 0.0;
    let H = 0.0;
    for (const c of counts) {
      if (c <= 0) continue;
      const p = c / sum;
      H += -p * Math.log2(p);
    }
    return H;
  }
}

export interface SensorResponse {
  stimulusType: string;
  stimulusIntensity: number;
  detected: boolean;
  responseStrength: number;
  spikeRate: number;
  networkSynchrony: number;
  informationEntropyBits: number;
}

export class FungalSensorTS {
  private targetStimulus: string;
  private network: FungalNetworkTS;
  private calibration = {
    moisture: { sensitivity: 2.06, threshold: 0.1 },
    temperature: { sensitivity: 1.96, threshold: 0.05 },
    light: { sensitivity: 1.83, threshold: 0.08 },
    chemical: { sensitivity: 0.5, threshold: 0.15 },
    mechanical: { sensitivity: 0.65, threshold: 0.12 },
  } as const;

  constructor(targetStimulus: string = "moisture") {
    this.targetStimulus = targetStimulus;
    this.network = new FungalNetworkTS(5, 0.4);
  }

  sense(intensity: number, noiseLevel: number = 0.1): SensorResponse {
    const noisy = intensity + (Math.random() * 2 - 1) * noiseLevel;
    const stimuli: Record<number, StimulusMap> = {
      1000: { [this.targetStimulus]: noisy },
      2000: { [this.targetStimulus]: noisy },
      3000: { [this.targetStimulus]: noisy },
    };
    const result = this.network.simulate(5000, 1, stimuli);
    const responseStrength = result.networkEntropyBits * intensity;
    const cal = (this.calibration as any)[this.targetStimulus] ?? { threshold: 0.1 };
    const detected = responseStrength > cal.threshold;
    return {
      stimulusType: this.targetStimulus,
      stimulusIntensity: intensity,
      detected,
      responseStrength,
      spikeRate: result.spikeRatePerS,
      networkSynchrony: result.synchrony.reduce((a, b) => a + b, 0) / Math.max(1, result.synchrony.length),
      informationEntropyBits: result.networkEntropyBits,
    };
  }
}

export class FungalComputerTS {
  private processingNetwork: FungalNetworkTS;

  constructor() {
    this.processingNetwork = new FungalNetworkTS(5, 0.8);
  }

  logicalAnd(inputA: boolean, inputB: boolean): boolean {
    const stimuli: Record<number, StimulusMap> = {};
    if (inputA) stimuli[1000] = { ...(stimuli[1000] ?? {}), input_a: 1.0 };
    if (inputB) stimuli[1000] = { ...(stimuli[1000] ?? {}), input_b: 1.0 };
    const result = this.processingNetwork.simulate(3000, 1, stimuli);
    const avgSync = result.synchrony.reduce((a, b) => a + b, 0) / Math.max(1, result.synchrony.length);
    return avgSync > 0.3;
  }

  patternRecognition(inputPattern: number[]): "simple_pattern" | "moderate_pattern" | "complex_pattern" {
    const stimuli: Record<number, StimulusMap> = {};
    for (let i = 0; i < inputPattern.length; i++) {
      stimuli[i] = { pattern: inputPattern[i] };
    }
    const result = this.processingNetwork.simulate(inputPattern.length * 100, 1, stimuli);
    const H = result.networkEntropyBits;
    if (H > 2.0) return "complex_pattern";
    if (H > 1.0) return "moderate_pattern";
    return "simple_pattern";
  }
}

