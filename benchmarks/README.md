# Phase 2 reference benchmark

This directory contains the reproducible benchmark generated on October 22,
2026 with:

```bash
PYTHONPATH=src python -m pathpilot.evaluation.benchmark \
  --scenarios 10 \
  --repetitions 3 \
  --width 30 \
  --height 20 \
  --density 0.20 \
  --dynamic-obstacles 2 \
  --seed 42 \
  --output benchmarks
```

The ten seeded maps alternate between static and dynamic scenarios. Each map is
created once, then independently recreated for both strategies. Dynamic runs
receive the same obstacle events at the same movement steps.

| Metric | Always Replan | Cached Path |
| --- | ---: | ---: |
| Runs | 30 | 30 |
| Success rate | 100% | 100% |
| Mean planning latency | 2.382 ms | 0.181 ms |
| Median planning latency | 2.400 ms | 0.186 ms |
| Mean nodes explored | 1,359 | 100 |
| Mean path length | 48 | 48 |
| Mean planning calls | 48 | 2 |
| Mean replans | 47 | 1 |

In this run, Cached Path reduced cumulative planning latency by **92.4%**,
nodes explored by **92.6%**, and planning calls by **95.8%**, while preserving
the same success rate and path length.

Latency depends on hardware and system load. Re-run the command on the target
machine instead of treating these exact timing values as universal. The raw
paired observations are in `results.csv`; aggregate values are in
`summary.json`.
