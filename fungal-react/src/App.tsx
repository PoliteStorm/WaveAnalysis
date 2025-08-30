import { useEffect, useState } from 'react'
import './App.css'

type ManifestEntry = {
  species: string
  date: string
  files: string[]
}

type Manifest = {
  species: string[]
  entries: ManifestEntry[]
}

function App() {
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [selected, setSelected] = useState<ManifestEntry | null>(null)
  const [metrics, setMetrics] = useState<any | null>(null)

  useEffect(() => {
    fetch('/data/manifest.json')
      .then((r) => r.json())
      .then((m: Manifest) => {
        setManifest(m)
        if (m.entries.length > 0) {
          setSelected(m.entries[0])
        }
      })
      .catch((e) => console.error('Failed to load manifest', e))
  }, [])

  useEffect(() => {
    if (!selected) return
    const base = `/data/${selected.species}/${selected.date}`
    fetch(`${base}/metrics.json`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => setMetrics(null))
  }, [selected])

  return (
    <div style={{ padding: 16 }}>
      <h1>Fungal Computing Explorer</h1>
      {!manifest && <p>Loading manifest…</p>}
      {manifest && (
        <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
          <div>
            <h3>Datasets</h3>
            <ul>
              {manifest.entries.map((e) => (
                <li key={`${e.species}-${e.date}`}>
                  <button
                    onClick={() => setSelected(e)}
                    style={{
                      cursor: 'pointer',
                      background: selected?.date === e.date && selected?.species === e.species ? '#eef' : 'white',
                    }}
                  >
                    {e.species} / {e.date}
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <div style={{ flex: 1 }}>
            <h3>Metrics</h3>
            {!metrics && <p>Select a dataset.</p>}
            {metrics && (
              <div>
                <p>
                  <b>File</b>: {metrics.file}
                </p>
                <p>
                  <b>Channel</b>: {metrics.channel} | <b>fs_hz</b>: {metrics.fs_hz}
                </p>
                <p>
                  <b>Spike count</b>: {metrics.spike_count}
                </p>
                <div style={{ display: 'flex', gap: 24 }}>
                  <div>
                    <h4>Amplitude stats</h4>
                    <pre style={{ background: '#f7f7f7', padding: 12 }}>
{JSON.stringify(metrics.amplitude_stats, null, 2)}
                    </pre>
                  </div>
                  <div>
                    <h4>ISI stats</h4>
                    <pre style={{ background: '#f7f7f7', padding: 12 }}>
{JSON.stringify(metrics.isi_stats, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default App
