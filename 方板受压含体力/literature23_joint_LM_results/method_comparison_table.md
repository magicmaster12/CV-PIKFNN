| Method | alpha0 | alpha_final | L2(ux) | L2(uy) | L2(sxx) | L2(syy) | L2(sxy) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed KM-PIKFNN | 1.1500 | 1.1500 | 6.269e-07 | 1.604e-06 | 7.997e-06 | 2.344e-05 | 5.989e-05 | 4.07 |
| Adaptive KM-PIKFNN | 1.1500 | 1.1540 | 1.946e-06 | 4.136e-06 | 2.104e-05 | 7.692e-05 | 1.685e-04 | 8.90 |
| Fixed KM-PIKFNN | 1.3000 | 1.3000 | 5.879e-07 | 1.481e-06 | 4.174e-06 | 1.419e-05 | 4.290e-05 | 6.75 |
| Adaptive KM-PIKFNN | 1.3000 | 1.3051 | 8.817e-07 | 2.219e-06 | 5.980e-06 | 2.022e-05 | 6.165e-05 | 14.46 |
| Fixed KM-PIKFNN | 1.4000 | 1.4000 | 1.167e-06 | 3.136e-06 | 6.386e-06 | 1.682e-05 | 5.750e-05 | 3.71 |
| Adaptive KM-PIKFNN | 1.4000 | 1.4051 | 1.377e-06 | 3.809e-06 | 7.131e-06 | 2.002e-05 | 6.788e-05 | 8.08 |
| Fixed KM-PIKFNN | 1.5000 | 1.5000 | 1.785e-06 | 5.183e-06 | 7.062e-06 | 2.220e-05 | 6.829e-05 | 3.70 |
| Adaptive KM-PIKFNN | 1.5000 | 1.5000 | 2.535e-06 | 7.082e-06 | 1.062e-05 | 2.942e-05 | 9.832e-05 | 8.19 |
| PINN | - | - | 6.994e-03 | 8.115e-03 | 1.719e-03 | 3.572e-03 | 1.186e-02 | 162.16 |

Relative L2 errors are computed componentwise on the same evaluation grid. Time is per run and includes initialization, training, and in-training diagnostics; field export and plotting are excluded. KM times include basis assembly and rank-revealing QR; all coefficients start at zero. Four-start total search costs are reported separately in search_costs.csv. See CSV device and parameter columns before comparing timings.
