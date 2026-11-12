| Method | alpha0 | alpha_final | L2(ux) | L2(uy) | L2(sxx) | L2(syy) | L2(sxy) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed KM-PIKFNN | 1.1000 | 1.1000 | 1.007e-02 | 1.801e-02 | 4.402e-02 | 5.744e-02 | 2.235e-01 | 43.14 |
| Adaptive KM-PIKFNN | 1.1000 | 1.1857 | 3.480e-03 | 9.059e-03 | 1.040e-02 | 2.621e-02 | 7.296e-02 | 68.14 |
| Fixed KM-PIKFNN | 1.1500 | 1.1500 | 4.292e-04 | 1.173e-03 | 2.386e-03 | 8.081e-03 | 1.816e-02 | 38.41 |
| Adaptive KM-PIKFNN | 1.1500 | 1.1716 | 1.368e-03 | 3.279e-03 | 4.491e-03 | 1.480e-02 | 3.748e-02 | 60.09 |
| Fixed KM-PIKFNN | 1.2000 | 1.2000 | 5.796e-04 | 1.405e-03 | 2.341e-03 | 8.291e-03 | 2.119e-02 | 38.60 |
| Adaptive KM-PIKFNN | 1.2000 | 1.2593 | 5.158e-03 | 1.035e-02 | 1.376e-02 | 2.326e-02 | 6.730e-02 | 60.25 |
| Fixed KM-PIKFNN | 1.2500 | 1.2500 | 8.510e-04 | 2.182e-03 | 2.704e-03 | 9.268e-03 | 2.669e-02 | 38.31 |
| Adaptive KM-PIKFNN | 1.2500 | 1.6500 | 1.699e-03 | 4.106e-03 | 4.003e-03 | 5.884e-03 | 1.656e-02 | 79.52 |
| Fixed KM-PIKFNN | 1.3000 | 1.3000 | 9.554e-04 | 2.326e-03 | 2.840e-03 | 8.943e-03 | 2.748e-02 | 38.90 |
| Adaptive KM-PIKFNN | 1.3000 | 1.6500 | 9.018e-04 | 2.434e-03 | 2.170e-03 | 4.286e-03 | 1.182e-02 | 56.33 |
| Fixed KM-PIKFNN | 1.4000 | 1.4000 | 1.495e-03 | 3.786e-03 | 3.460e-03 | 1.055e-02 | 3.511e-02 | 38.34 |
| Adaptive KM-PIKFNN | 1.4000 | 1.6500 | 1.232e-03 | 3.263e-03 | 2.780e-03 | 4.891e-03 | 1.129e-02 | 56.68 |
| Fixed KM-PIKFNN | 1.5000 | 1.5000 | 1.736e-03 | 3.720e-03 | 3.997e-03 | 7.531e-03 | 2.654e-02 | 51.42 |
| Adaptive KM-PIKFNN | 1.5000 | 1.6500 | 1.132e-03 | 3.284e-03 | 2.610e-03 | 4.947e-03 | 1.104e-02 | 99.77 |
| PINN | - | - | 6.994e-03 | 8.115e-03 | 1.719e-03 | 3.572e-03 | 1.186e-02 | 162.16 |

Relative L2 errors are computed componentwise on the same evaluation grid. Time is per run and includes initialization, training, and in-training diagnostics; field export and plotting are excluded. KM times include column-scaling matrix assembly and any enabled initialization; cold starts do not run SVD. Seven-start total search costs are reported separately in search_costs.csv. See CSV device and parameter columns before comparing timings.
