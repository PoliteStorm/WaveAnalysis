export default function FslPage() {
  const operators = [
    { group: 'Stimulus', ops: ['⊕', '⊗', '△'] },
    { group: 'Rhythm', ops: ['∿', '∿∿', '⟲'] },
    { group: 'Membrane', ops: ['∇', '∫', 'τ'] },
    { group: 'Spike', ops: ['⚡', '⟨⟩', '⌈⌉'] },
    { group: 'Network', ops: ['⟷', '⟶', '⊙', '∑'] },
    { group: 'Logic', ops: ['∧', '∨', '¬', '⊻'] },
    { group: 'Information', ops: ['ℋ', '𝒟', 'ℙ'] },
    { group: 'Time', ops: ['∂', '∫', '⟨ ⟩'] },
    { group: 'Validation', ops: ['✓', 'δ', 'ρ'] },
  ]

  const examples = [
    { name: 'stimulus_combination', expr: '⊕(moisture, temperature)' },
    { name: 'rhythm_generation', expr: '∿∿(fast∿, medium∿, slow∿)' },
    { name: 'spike_probability', expr: '⚡(P_spike, threshold)' },
    { name: 'network_connection', expr: '⟷(neuron_i, neuron_j, strength)' },
    { name: 'logic_and', expr: '∧(input_a, input_b)' },
    { name: 'entropy_calculation', expr: 'ℋ(spike_train)' },
    { name: 'effect_size', expr: 'δ(pre_stats, post_stats)' },
  ]

  return (
    <div style={{ padding: 16 }}>
      <h1>Fungal Symbolic Language (FSL)</h1>
      <h3>Operators</h3>
      <ul>
        {operators.map((g) => (
          <li key={g.group}>
            <b>{g.group}:</b> {g.ops.join(' ')}
          </li>
        ))}
      </ul>

      <h3>Examples</h3>
      <ul>
        {examples.map((e) => (
          <li key={e.name}>
            <code>{e.name}</code>: <span style={{ fontSize: 20 }}>{e.expr}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

