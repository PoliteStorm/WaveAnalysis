import { useMemo, useState } from 'react'
import { FungalSensorTS } from '../lib/fungal'

export function SensorDemo() {
  const [stimulus, setStimulus] = useState<'moisture' | 'temperature' | 'light' | 'chemical' | 'mechanical'>('moisture')
  const [intensity, setIntensity] = useState(0.5)
  const sensor = useMemo(() => new FungalSensorTS(stimulus), [stimulus])
  const response = sensor.sense(intensity)
  return (
    <div style={{ border: '1px solid #ddd', padding: 12, borderRadius: 8 }}>
      <h3>Fungal Sensor</h3>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', flexWrap: 'wrap' }}>
        <label>
          Stimulus
          <select value={stimulus} onChange={(e) => setStimulus(e.target.value as any)} style={{ marginLeft: 8 }}>
            <option value="moisture">moisture</option>
            <option value="temperature">temperature</option>
            <option value="light">light</option>
            <option value="chemical">chemical</option>
            <option value="mechanical">mechanical</option>
          </select>
        </label>
        <label>
          Intensity: {intensity.toFixed(2)}
          <input type="range" min={0} max={1} step={0.01} value={intensity} onChange={(e) => setIntensity(parseFloat(e.target.value))} style={{ marginLeft: 8 }} />
        </label>
        <div>
          Detected: <strong>{response.detected ? 'Yes' : 'No'}</strong>
        </div>
        <div>Entropy: {response.informationEntropyBits.toFixed(3)} bits</div>
        <div>Spike rate: {response.spikeRate.toFixed(3)} s^-1</div>
      </div>
    </div>
  )
}

