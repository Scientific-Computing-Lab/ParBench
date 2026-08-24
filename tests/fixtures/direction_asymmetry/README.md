# Direction-asymmetry fixture (Task 6, gate E-01)

Hand-authored synthetic corpus: one model directory, five direction pairs,
seven kernels, three stochastic L0 samples per (model, suite, kernel,
direction) cell (42 records; every record `temperature: 0.7`).

The pass patterns are chosen so that any-of-three cell success (the corrected
statistical unit) and last-sample-wins (the submitted overwrite semantics)
disagree on six cells.
For the `cuda-to-omp` / `omp-to-cuda` pair the corrected rule yields the 2x2
table `{both_pass: 0, discordant: 2, both_fail: 0}` while the overwrite rule
yields `{both_pass: 0, discordant: 0, both_fail: 2}`.
`tests/test_direction_asymmetry.py` pins the corrected table, so it fails on
the pre-Task-6 code and passes after it.

Spec IDs reuse real corpus names (rodinia-bfs-cuda, ...) so suite/kernel
parsing matches production, but the records themselves are synthetic and never
touch `results/evaluation`.
