import { useMemo, useState } from 'react'
import { FungalComputerTS } from '../lib/fungal'

export function LogicGateDemo() {
  const computer = useMemo(() => new FungalComputerTS(), [])
  const [a, setA] = useState(false)
  const [b, setB] = useState(false)
  const result = computer.logicalAnd(a, b)
  return (
    <div style={{ border: '1px solid #ddd', padding: 12, borderRadius: 8 }}>
      <h3>Fungal Logic Gate (AND)</h3>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <label><input type="checkbox" checked={a} onChange={(e) => setA(e.target.checked)} /> Input A</label>
        <label><input type="checkbox" checked={b} onChange={(e) => setB(e.target.checked)} /> Input B</label>
        <div>
          Output: <strong>{String(result)}</strong>
        </div>
      </div>
    </div>
  )
}

