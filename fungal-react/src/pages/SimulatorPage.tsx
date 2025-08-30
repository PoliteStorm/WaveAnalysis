import { useMemo, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

type Stimulus = 'moisture' | 'temperature' | 'light' | 'chemical' | 'mechanical'

type Neuron = {
  membranePotential: number
  lastSpikeTime: number
  refractory: number
}

const rhythmBands = [
  { freq: 1 / 5.5, power: 0.22 },
  { freq: 1 / 24.5, power: 0.20 },
  { freq: 1 / 104.0, power: 0.58 },
]

const stimulusGains: Record<Stimulus, number> = {
  moisture: 2.06,
  temperature: 1.96,
  light: 1.83,
  chemical: 0.5,
  mechanical: 0.65,
}

function simulateNetwork(
  duration: number,
  dt: number,
  nNeurons: number,
  stimuliTimes: number[],
  stimulus: Stimulus,
  intensity: number,
) {
  const neurons: Neuron[] = Array.from({ length: nNeurons }, () => ({
    membranePotential: 0,
    lastSpikeTime: 0,
    refractory: 0,
  }))

  const timePoints: number[] = []
  const networkActivity: number[] = []
  const spikes: { t: number; neuron: number }[] = []

  for (let t = 0; t < duration; t += dt) {
    timePoints.push(t)
    const hasStimulus = stimuliTimes.includes(t)
    const stimValue = hasStimulus ? intensity * stimulusGains[stimulus] : 0

    let active = 0
    for (let i = 0; i < nNeurons; i++) {
      const n = neurons[i]
      // rhythm
      let rhythm = 0
      for (const band of rhythmBands) {
        const phase = 2 * Math.PI * band.freq * n.lastSpikeTime
        rhythm += band.power * Math.sin(phase)
      }
      n.membranePotential += (rhythm + stimValue) * dt
      if (n.refractory > 0) {
        n.refractory -= dt
        n.membranePotential *= 0.1
      }
      const p0 = 0.001 * dt
      const spikeProb = p0 * (1 + Math.abs(n.membranePotential))
      const r = Math.random()
      if (r < spikeProb && n.refractory <= 0) {
        spikes.push({ t, neuron: i })
        n.lastSpikeTime = 0
        // approximate refractory from lognormal-ish spread
        n.refractory = Math.max(333, Math.min(11429, 3540 * Math.exp((Math.random() - 0.5))))
        active++
      } else {
        n.lastSpikeTime += dt
      }
    }
    networkActivity.push(active / nNeurons)
  }

  const chartData = timePoints.map((t, idx) => ({ t, activity: networkActivity[idx] }))
  return { chartData, spikes }
}

export default function SimulatorPage() {
  const [nNeurons, setNNeurons] = useState(8)
  const [intensity, setIntensity] = useState(0.8)
  const [stimulus, setStimulus] = useState<Stimulus>('moisture')
  const [duration, setDuration] = useState(3000)
  const [dt, setDt] = useState(10)

  const { chartData } = useMemo(() => {
    return simulateNetwork(duration, dt, nNeurons, [1000, 2000], stimulus, intensity)
  }, [duration, dt, nNeurons, stimulus, intensity])

  return (
    <div style={{ padding: 16 }}>
      <h1>Fungal Network Simulator</h1>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        <label>
          Neurons: <input type="number" value={nNeurons} min={2} max={64} onChange={(e) => setNNeurons(parseInt(e.target.value || '0'))} />
        </label>
        <label>
          Duration (s): <input type="number" value={duration} step={100} onChange={(e) => setDuration(parseInt(e.target.value || '0'))} />
        </label>
        <label>
          dt (s): <input type="number" value={dt} step={1} onChange={(e) => setDt(parseInt(e.target.value || '0'))} />
        </label>
        <label>
          Stimulus:
          <select value={stimulus} onChange={(e) => setStimulus(e.target.value as Stimulus)}>
            <option value="moisture">moisture</option>
            <option value="temperature">temperature</option>
            <option value="light">light</option>
            <option value="chemical">chemical</option>
            <option value="mechanical">mechanical</option>
          </select>
        </label>
        <label>
          Intensity: <input type="number" value={intensity} step={0.1} onChange={(e) => setIntensity(parseFloat(e.target.value || '0'))} />
        </label>
      </div>

      <div style={{ height: 360, marginTop: 16 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" label={{ value: 't (s)', position: 'insideBottomRight', offset: -5 }} />
            <YAxis label={{ value: 'activity', angle: -90, position: 'insideLeft' }} domain={[0, 1]} />
            <Tooltip />
            <Line type="monotone" dataKey="activity" stroke="#1976d2" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

