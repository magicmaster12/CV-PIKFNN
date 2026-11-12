| Method | alpha0 | alpha_final | L2(ux) | L2(uy) | L2(sxx) | L2(syy) | L2(sxy) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed KM-PIKFNN | 1.1000 | 1.1000 | 1.074e-04 | 1.749e-04 | 1.005e-03 | 1.293e-03 | 6.352e-03 | 5.31 |
| Adaptive KM-PIKFNN | 1.1000 | 1.1000 | 1.074e-04 | 1.747e-04 | 1.005e-03 | 1.293e-03 | 6.350e-03 | 66.13 |
| Fixed KM-PIKFNN | 1.1500 | 1.1500 | 4.196e-07 | 9.344e-07 | 4.841e-06 | 1.336e-05 | 5.556e-05 | 10.30 |
| Adaptive KM-PIKFNN | 1.1500 | 1.1500 | 4.196e-07 | 9.344e-07 | 4.841e-06 | 1.336e-05 | 5.556e-05 | 4.72 |
| Fixed KM-PIKFNN | 1.2000 | 1.2000 | 5.386e-07 | 1.269e-06 | 6.671e-06 | 2.585e-05 | 5.719e-05 | 4.63 |
| Adaptive KM-PIKFNN | 1.2000 | 1.2000 | 5.386e-07 | 1.269e-06 | 6.671e-06 | 2.585e-05 | 5.719e-05 | 9.62 |
| Fixed KM-PIKFNN | 1.2500 | 1.2500 | 3.864e-07 | 7.633e-07 | 4.423e-06 | 8.730e-06 | 3.200e-05 | 11.03 |
| Adaptive KM-PIKFNN | 1.2500 | 1.2500 | 3.864e-07 | 7.633e-07 | 4.423e-06 | 8.730e-06 | 3.200e-05 | 11.54 |
| Fixed KM-PIKFNN | 1.3000 | 1.3000 | 1.096e-06 | 2.082e-06 | 1.085e-05 | 2.102e-05 | 6.193e-05 | 11.12 |
| Adaptive KM-PIKFNN | 1.3000 | 1.3000 | 1.096e-06 | 2.082e-06 | 1.085e-05 | 2.102e-05 | 6.193e-05 | 11.49 |
| Fixed KM-PIKFNN | 1.4000 | 1.4000 | 9.834e-07 | 2.726e-06 | 7.370e-06 | 2.088e-05 | 6.445e-05 | 11.08 |
| Adaptive KM-PIKFNN | 1.4000 | 1.4000 | 9.834e-07 | 2.726e-06 | 7.370e-06 | 2.088e-05 | 6.445e-05 | 11.64 |
| Fixed KM-PIKFNN | 1.5000 | 1.5000 | 5.801e-07 | 1.055e-06 | 3.858e-06 | 4.301e-06 | 1.672e-05 | 11.07 |
| Adaptive KM-PIKFNN | 1.5000 | 1.5000 | 5.801e-07 | 1.055e-06 | 3.858e-06 | 4.301e-06 | 1.672e-05 | 11.57 |
| PINN | - | - | 6.994e-03 | 8.115e-03 | 1.719e-03 | 3.572e-03 | 1.186e-02 | 162.16 |

Relative L2 errors are computed componentwise on the same evaluation grid. Time is per run and includes initialization, training, and in-training diagnostics; field export and plotting are excluded. KM times include basis assembly and rank-revealing QR; all coefficients start at zero. Seven-start total search costs are reported separately in search_costs.csv. See CSV device and parameter columns before comparing timings.
