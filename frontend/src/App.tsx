import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import { SpikeRaster } from './components/SpikeRaster'
import { TauBandsChart } from './components/TauBandsChart'
import { LogicGateDemo } from './components/LogicGateDemo'
import { SensorDemo } from './components/SensorDemo'

function App() {

  return (
    <>
      <div>
        <a href="https://vite.dev" target="_blank">
          <img src={viteLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Fungal Computing Playground</h1>
      <div className="card" style={{ display: 'grid', gap: 16 }}>
        <section>
          <h2>Spike Raster (Schizophyllum commune)</h2>
          <SpikeRaster url="/data/Schizophyllum_commune/spike_times_s.csv" />
        </section>
        <section>
          <h2>τ-band Timeseries</h2>
          <TauBandsChart url="/data/Schizophyllum_commune/tau_band_timeseries.csv" />
        </section>
        <section>
          <LogicGateDemo />
        </section>
        <section>
          <SensorDemo />
        </section>
      </div>
    </>
  )
}

export default App
