# Fungal Computing and Sensor Simulation

This directory hosts runnable demos and documentation for fungal computing and sensor simulation, built on the analysis pipelines and validated parameters in this repo.

Contents:
- computing/simulator/: network-level computing demos (logic, pattern recognition, synchrony)
- computing/sensors/: hardware-oriented sensor prototypes and distributed monitoring
- computing/examples/: quickstart scripts showing how to use outputs under results/ to drive simulations

Run:
- python3 computing/examples/run_simulator_demo.py
- python3 computing/examples/run_sensor_demo.py

All examples assume results/ contains recent, timestamped analyses (metrics.json, tau CSV, etc.).
