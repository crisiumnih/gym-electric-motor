# Simulation model scope

All existing GEM experiments use their recorded plant configurations. Their
parameters are assumed legacy simulation values, not measured specifications of
the user's 2805 140KV motor. In particular, the 44.4 V bus, 21 pole pairs, 85 mΩ,
50 µH and 4 A termination threshold are not verified motor ratings.

Hardware inventory, identification procedures and integration work were removed
at the user's request. Current work is simulation-only: qualify RL against PI
before considering any change in scope. Preserve original configurations and raw
simulation results; do not retrospectively relabel them as real-motor evidence.
