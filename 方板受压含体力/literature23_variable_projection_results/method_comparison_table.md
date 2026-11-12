| Method | alpha0 | alpha_final | L2(ux) | L2(uy) | L2(sxx) | L2(syy) | L2(sxy) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed KM-PIKFNN | 1.1500 | 1.1500 | 4.940e-07 | 5.360e-07 | 6.561e-06 | 4.024e-06 | 2.594e-05 | 0.27 |
| Adaptive KM-PIKFNN | 1.1500 | 1.4175 | 6.754e-11 | 1.756e-10 | 6.847e-10 | 2.061e-09 | 7.955e-09 | 4.45 |
| Fixed KM-PIKFNN | 1.3000 | 1.3000 | 2.539e-11 | 7.387e-11 | 3.339e-10 | 1.824e-09 | 2.793e-09 | 0.10 |
| Adaptive KM-PIKFNN | 1.3000 | 1.4175 | 6.754e-11 | 1.756e-10 | 6.847e-10 | 2.061e-09 | 7.955e-09 | 5.51 |
| Fixed KM-PIKFNN | 1.4000 | 1.4000 | 6.470e-11 | 1.672e-10 | 6.565e-10 | 2.010e-09 | 7.857e-09 | 0.12 |
| Adaptive KM-PIKFNN | 1.4000 | 1.4175 | 6.754e-11 | 1.756e-10 | 6.847e-10 | 2.061e-09 | 7.955e-09 | 5.36 |
| Fixed KM-PIKFNN | 1.5000 | 1.5000 | 1.027e-10 | 3.009e-10 | 8.418e-10 | 2.291e-09 | 8.511e-09 | 0.09 |
| Adaptive KM-PIKFNN | 1.5000 | 1.4175 | 6.754e-11 | 1.756e-10 | 6.847e-10 | 2.061e-09 | 7.955e-09 | 4.01 |
| PINN | - | - | 6.994e-03 | 8.115e-03 | 1.719e-03 | 3.572e-03 | 1.186e-02 | 162.16 |

Relative L2 errors are computed componentwise on the same evaluation grid. Time is per run and includes initialization, training, and in-training diagnostics; field export and plotting are excluded. KM times include kernel assembly and all truncated-SVD coefficient solves. No external coefficient solution is reused. Four-start total search costs are reported separately in search_costs.csv. See CSV device and parameter columns before comparing timings.
