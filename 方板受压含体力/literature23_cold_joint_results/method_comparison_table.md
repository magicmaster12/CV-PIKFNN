| Method | alpha0 | alpha_final | L2(ux) | L2(uy) | L2(sxx) | L2(syy) | L2(sxy) | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed KM-PIKFNN | 1.1000 | 1.1000 | 2.069e-04 | 4.954e-04 | 1.774e-03 | 4.595e-03 | 1.123e-02 | 295.91 |
| Adaptive KM-PIKFNN | 1.1000 | 1.1257 | 1.248e-04 | 3.544e-04 | 7.812e-04 | 2.779e-03 | 6.937e-03 | 351.26 |
| Fixed KM-PIKFNN | 1.1500 | 1.1500 | 2.126e-04 | 5.126e-04 | 1.168e-03 | 4.413e-03 | 1.032e-02 | 371.59 |
| Adaptive KM-PIKFNN | 1.1500 | 1.1615 | 2.507e-04 | 6.534e-04 | 1.114e-03 | 4.109e-03 | 9.755e-03 | 226.50 |
| Fixed KM-PIKFNN | 1.2000 | 1.2000 | 3.392e-04 | 8.800e-04 | 1.319e-03 | 5.111e-03 | 1.350e-02 | 386.93 |
| Adaptive KM-PIKFNN | 1.2000 | 1.2752 | 8.321e-04 | 1.720e-03 | 2.600e-03 | 7.137e-03 | 1.742e-02 | 1342.57 |
| Fixed KM-PIKFNN | 1.2500 | 1.2500 | 4.834e-04 | 1.146e-03 | 1.762e-03 | 5.368e-03 | 1.532e-02 | 603.39 |
| Adaptive KM-PIKFNN | 1.2500 | 1.7302 | 1.541e-03 | 3.418e-03 | 3.220e-03 | 6.529e-03 | 1.643e-02 | 590.63 |
| Fixed KM-PIKFNN | 1.3000 | 1.3000 | 6.236e-04 | 1.616e-03 | 2.023e-03 | 6.468e-03 | 2.004e-02 | 380.25 |
| Adaptive KM-PIKFNN | 1.3000 | 1.8000 | 2.229e-03 | 4.442e-03 | 4.599e-03 | 5.571e-03 | 1.842e-02 | 198.99 |
| Fixed KM-PIKFNN | 1.4000 | 1.4000 | 1.135e-03 | 3.038e-03 | 2.853e-03 | 7.417e-03 | 2.399e-02 | 393.92 |
| Adaptive KM-PIKFNN | 1.4000 | 1.8000 | 1.235e-03 | 2.410e-03 | 2.923e-03 | 3.462e-03 | 1.370e-02 | 417.22 |
| Fixed KM-PIKFNN | 1.5000 | 1.5000 | 1.009e-03 | 2.579e-03 | 2.096e-03 | 5.800e-03 | 1.560e-02 | 267.74 |
| Adaptive KM-PIKFNN | 1.5000 | 1.8000 | 1.072e-03 | 2.168e-03 | 2.515e-03 | 3.587e-03 | 1.425e-02 | 209.77 |
| PINN | - | - | 6.994e-03 | 8.115e-03 | 1.719e-03 | 3.572e-03 | 1.186e-02 | 162.16 |

Relative L2 errors are computed componentwise on the same evaluation grid. Time is per run and includes initialization, training, and in-training diagnostics; field export and plotting are excluded. KM times include basis assembly and rank-revealing QR; all coefficients start at zero. Seven-start total search costs are reported separately in search_costs.csv. See CSV device and parameter columns before comparing timings.
