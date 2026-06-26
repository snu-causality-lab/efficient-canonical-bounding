# paper_stress_scaling — Table 5

Stress test of the LP-solve speedup as the lifted dimension grows. We tile
the reduced (c_tilde, M_tilde) into larger LPs of dim `fiber × 8`:

| fiber | dim_lifted | paper speedup |
|------:|-----------:|--------------:|
| 16    | 128        | 12.0× |
| 64    | 512        | 44.8× |
| 256   | 2048       | 218.0× |

**Inflation construction**: `c_new = repeat(c, fiber) / fiber`, same for `M`.
Aggregating each fiber-group recovers the original LP; the optimum is
unchanged, but ECOS's solve time scales (roughly linearly per basis pivot)
with dim.

**Output**: `table5_stress_scaling.csv`.

The reduced-side solve time stays ~constant (~0.15ms for NDE L_{2.5}), while
lifted scales ~12×→45×→218×.
