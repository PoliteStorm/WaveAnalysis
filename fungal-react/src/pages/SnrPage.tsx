import { useEffect, useMemo, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer, CartesianGrid } from 'recharts'

type AblationItem = {
  window: string
  detrend_u: boolean
  snr: number
  concentration: number
}

type SnrAblation = {
  u0: number
  tau: number
  sqrt_ablation: AblationItem[]
  stft: { snr: number; concentration: number }
}

export default function SnrPage() {
  const [data, setData] = useState<SnrAblation | null>(null)

  useEffect(() => {
    fetch('/data/Schizophyllum_commune/2025-08-22T00:13:04/snr_ablation.json')
      .then((r) => r.json())
      .then(setData)
      .catch(() => setData(null))
  }, [])

  const chartData = useMemo(() => {
    if (!data) return []
    const sqrt = data.sqrt_ablation.map((d) => ({
      name: `sqrt_${d.window}${d.detrend_u ? '_detrend' : ''}`,
      snr: d.snr,
      concentration: d.concentration,
    }))
    const stft = [{ name: 'stft', snr: data.stft.snr, concentration: data.stft.concentration }]
    return [...sqrt, ...stft]
  }, [data])

  return (
    <div style={{ padding: 16 }}>
      <h1>SNR Ablation</h1>
      {!data && <p>Loading…</p>}
      {data && (
        <>
          <p>
            tau: <b>{data.tau}</b> | u0: <b>{data.u0.toFixed(2)}</b>
          </p>
          <div style={{ height: 360 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-20} textAnchor="end" height={60} interval={0} />
                <YAxis yAxisId="left" label={{ value: 'SNR', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: 'Concentration', angle: 90, position: 'insideRight' }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="snr" stroke="#8884d8" name="SNR" />
                <Line yAxisId="right" type="monotone" dataKey="concentration" stroke="#82ca9d" name="Concentration" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  )
}

