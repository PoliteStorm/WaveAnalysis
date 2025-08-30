# Fungal Computing Frontend

A lightweight React + TypeScript frontend to explore fungal computing datasets and run simple simulations derived from this repo's models.

## Features

- Species Explorer for dataset metrics (reads `public/data/manifest.json`)
- SNR Ablation chart comparing sqrt-transform configs vs STFT
- Interactive Fungal Network Simulator (toy JS port of `FungalNetwork` logic)
- FSL page listing symbolic operators and examples

## Getting started

```bash
cd fungal-react
npm install
npm run dev
```

Open the app and navigate:
- `/` datasets and metrics
- `/snr` SNR ablation
- `/sim` network simulator
- `/fsl` symbolic language

To add more datasets, copy result JSONs under `public/data/<Species>/<Timestamp>/` and add an entry to `public/data/manifest.json`.
