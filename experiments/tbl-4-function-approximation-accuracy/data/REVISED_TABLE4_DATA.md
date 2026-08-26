# Old Table 4

| Function | Taylor MSE | Frac-T MSE | Interp MSE | Frac-I MSE | LinearLUT MSE | NN-LUT MSE | T-LUT MSE | SCNA MSE | Taylor MAE | Frac-T MAE | Interp MAE | Frac-I MAE | LinearLUT MAE | NN-LUT MAE | T-LUT MAE | SCNA MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp | INF | 1.23e-8 | 3.60e-2 | 1.07e-9 | 4.41e-6 | 1.24e-6 | 1.24e-6 | 1.91e-8 | INF | 2.47e-6 | 5.85e-2 | 2.47e-6 | 1.40e-4 | 2.25e-4 | 1.95e-3 | 1.91e-5 |
| Exp2 | INF | 1.75e-8 | 3.37e-2 | 1.55e-9 | 2.73e-6 | 1.28e-6 | 1.24e-6 | 1.01e-8 | INF | 8.48e-6 | 5.69e-2 | 3.57e-6 | 1.35e-4 | 2.00e-4 | 1.95e-3 | 1.82e-5 |
| Sigmoid | 1.77e-3 | N/A | 1.25e-4 | N/A | 5.59e-8 | 2.36e-6 | 1.23e-6 | 4.01e-8 | 2.61e-2 | N/A | 8.26e-3 | N/A | 1.23e-4 | 1.20e-3 | 1.95e-3 | 1.61e-4 |
| Softsign | N/A | N/A | 5.24e-2 | N/A | 3.58e-6 | 3.45e-4 | 1.27e-6 | 1.93e-7 | N/A | N/A | 1.02e-1 | N/A | 2.50e-4 | 1.39e-2 | 1.95e-3 | 2.83e-4 |
| Softplus | 5.98e-2 | N/A | 1.75e-4 | N/A | 7.35e-7 | 3.89e-6 | 1.26e-6 | 1.12e-7 | 1.45e-1 | N/A | 1.02e-2 | N/A | 5.02e-4 | 1.63e-3 | 1.95e-3 | 2.69e-4 |
| Tanh | 4.64e-2 | N/A | 4.55e-5 | N/A | 6.50e-8 | 9.68e-6 | 6.52e-7 | 1.80e-7 | 1.52e-1 | N/A | 4.57e-3 | N/A | 1.21e-4 | 2.58e-3 | 1.95e-3 | 3.27e-4 |
| Arctan | N/A | N/A | 1.40e-1 | N/A | 8.41e-6 | 2.25e-4 | 1.26e-6 | 1.03e-7 | N/A | N/A | 1.66e-1 | N/A | 3.92e-4 | 7.85e-3 | 1.95e-3 | 2.00e-4 |
| Erf | N/A | N/A | 3.74e-4 | N/A | 1.85e-7 | 8.38e-6 | 4.32e-7 | 1.12e-7 | N/A | N/A | 1.47e-2 | N/A | 2.46e-4 | 2.30e-3 | 1.95e-3 | 2.68e-4 |
| Sin | 5.23e-4 | N/A | 9.74e-5 | N/A | 1.30e-7 | 2.99e-3 | N/A | 3.00e-7 | 1.27e-2 | N/A | 8.15e-3 | N/A | 2.46e-4 | 5.47e-2 | N/A | 4.50e-4 |
| Rsqrt | N/A | N/A | 3.79e-1 | N/A | 8.05e-5 | 2.10e-2 | N/A | 8.96e-7 | N/A | N/A | 1.90e-1 | N/A | 5.40e-4 | 1.10e-1 | N/A | 5.57e-4 |
| GeLU | 1.65e-1 | N/A | 3.27e-5 | N/A | 6.10e-9 | 6.67e-6 | 1.19e-6 | 9.95e-8 | 2.41e-1 | N/A | 3.41e-3 | N/A | 3.63e-5 | 2.08e-3 | 1.95e-3 | 2.56e-4 |
| Geomean | 1.2e5x | N/A | 1.9e4x | N/A | 7.61x | 431x | 14.9x | 1.00x | 371x | N/A | 127x | N/A | 1.02x | 22.6x | 14.1x | 1.00x |

# New Table 4 (SCNA-16)

The revised SCNA-16 and SCNA-32 Rsqrt entries use `1 / sqrt(x)` on `(0.1, 1024)`, trained after reflection on `[-1024, -0.1]` with `lambda_bound=0.10` and evaluated on 1,000 uniformly spaced points.

| Function | Taylor MSE | Frac-T MSE | Interp MSE | Frac-I MSE | LinearLUT MSE | NN-LUT MSE | T-LUT MSE | SCNA MSE | Taylor MAE | Frac-T MAE | Interp MAE | Frac-I MAE | LinearLUT MAE | NN-LUT MAE | T-LUT MAE | SCNA MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp | INF | 1.23e-8 | 3.60e-2 | 1.07e-9 | 4.41e-6 | 1.24e-6 | 1.24e-6 | 5.02e-9 | INF | 2.47e-6 | 5.85e-2 | 2.47e-6 | 1.40e-4 | 2.25e-4 | 1.95e-3 | 1.12e-5 |
| Exp2 | INF | 1.75e-8 | 3.37e-2 | 1.55e-9 | 2.73e-6 | 1.28e-6 | 1.24e-6 | 1.70e-8 | INF | 8.48e-6 | 5.69e-2 | 3.57e-6 | 1.35e-4 | 2.00e-4 | 1.95e-3 | 2.38e-5 |
| Sigmoid | 1.77e-3 | N/A | 1.25e-4 | N/A | 5.59e-8 | 2.36e-6 | 1.23e-6 | 6.78e-8 | 2.61e-2 | N/A | 8.26e-3 | N/A | 1.23e-4 | 1.20e-3 | 1.95e-3 | 2.18e-4 |
| Softsign | N/A | N/A | 5.24e-2 | N/A | 3.58e-6 | 3.45e-4 | 1.27e-6 | 2.11e-7 | N/A | N/A | 1.02e-1 | N/A | 2.50e-4 | 1.39e-2 | 1.95e-3 | 2.97e-4 |
| Softplus | 5.98e-2 | N/A | 1.75e-4 | N/A | 7.35e-7 | 3.89e-6 | 1.26e-6 | 5.93e-8 | 1.45e-1 | N/A | 1.02e-2 | N/A | 5.02e-4 | 1.63e-3 | 1.95e-3 | 1.55e-4 |
| Tanh | 4.64e-2 | N/A | 4.55e-5 | N/A | 6.50e-8 | 9.68e-6 | 6.52e-7 | 1.61e-7 | 1.52e-1 | N/A | 4.57e-3 | N/A | 1.21e-4 | 2.58e-3 | 1.95e-3 | 3.34e-4 |
| Arctan | N/A | N/A | 1.40e-1 | N/A | 8.41e-6 | 2.25e-4 | 1.26e-6 | 3.04e-7 | N/A | N/A | 1.66e-1 | N/A | 3.92e-4 | 7.85e-3 | 1.95e-3 | 3.67e-4 |
| Erf | N/A | N/A | 3.74e-4 | N/A | 1.85e-7 | 8.38e-6 | 4.32e-7 | 1.30e-7 | N/A | N/A | 1.47e-2 | N/A | 2.46e-4 | 2.30e-3 | 1.95e-3 | 3.03e-4 |
| Sin | 5.23e-4 | N/A | 9.74e-5 | N/A | 1.30e-7 | 2.99e-3 | N/A | 6.47e-8 | 1.27e-2 | N/A | 8.15e-3 | N/A | 2.46e-4 | 5.47e-2 | N/A | 2.16e-4 |
| Rsqrt | N/A | N/A | 3.79e-1 | N/A | 8.05e-5 | 2.10e-2 | N/A | 1.42e-6 | N/A | N/A | 1.90e-1 | N/A | 5.40e-4 | 1.10e-1 | N/A | 7.35e-4 |
| GeLU | 1.65e-1 | N/A | 3.27e-5 | N/A | 6.10e-9 | 6.67e-6 | 1.19e-6 | 7.84e-8 | 2.41e-1 | N/A | 3.41e-3 | N/A | 3.63e-5 | 2.08e-3 | 1.95e-3 | 1.67e-4 |
| Geomean | 1.67e5x | N/A | 2.35e4x | N/A | 7.36x | 356x | 14.9x | 1.00x | 337x | N/A | 146x | N/A | 1.18x | 20.5x | 14.3x | 1.00x |

# SCNA-32 Reference

| Function | Taylor MSE | Frac-T MSE | Interp MSE | Frac-I MSE | LinearLUT MSE | NN-LUT MSE | T-LUT MSE | SCNA MSE | Taylor MAE | Frac-T MAE | Interp MAE | Frac-I MAE | LinearLUT MAE | NN-LUT MAE | T-LUT MAE | SCNA MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp | INF | 1.23e-8 | 3.60e-2 | 1.07e-9 | 4.41e-6 | 1.24e-6 | 1.24e-6 | 1.35e-8 | INF | 2.47e-6 | 5.85e-2 | 2.47e-6 | 1.40e-4 | 2.25e-4 | 1.95e-3 | 1.20e-5 |
| Exp2 | INF | 1.75e-8 | 3.37e-2 | 1.55e-9 | 2.73e-6 | 1.28e-6 | 1.24e-6 | 1.69e-8 | INF | 8.48e-6 | 5.69e-2 | 3.57e-6 | 1.35e-4 | 2.00e-4 | 1.95e-3 | 1.66e-5 |
| Sigmoid | 1.77e-3 | N/A | 1.25e-4 | N/A | 5.59e-8 | 2.36e-6 | 1.23e-6 | 5.07e-8 | 2.61e-2 | N/A | 8.26e-3 | N/A | 1.23e-4 | 1.20e-3 | 1.95e-3 | 1.89e-4 |
| Softsign | N/A | N/A | 5.24e-2 | N/A | 3.58e-6 | 3.45e-4 | 1.27e-6 | 1.77e-8 | N/A | N/A | 1.02e-1 | N/A | 2.50e-4 | 1.39e-2 | 1.95e-3 | 9.44e-5 |
| Softplus | 5.98e-2 | N/A | 1.75e-4 | N/A | 7.35e-7 | 3.89e-6 | 1.26e-6 | 2.46e-8 | 1.45e-1 | N/A | 1.02e-2 | N/A | 5.02e-4 | 1.63e-3 | 1.95e-3 | 1.00e-4 |
| Tanh | 4.64e-2 | N/A | 4.55e-5 | N/A | 6.50e-8 | 9.68e-6 | 6.52e-7 | 6.21e-8 | 1.52e-1 | N/A | 4.57e-3 | N/A | 1.21e-4 | 2.58e-3 | 1.95e-3 | 2.08e-4 |
| Arctan | N/A | N/A | 1.40e-1 | N/A | 8.41e-6 | 2.25e-4 | 1.26e-6 | 5.67e-8 | N/A | N/A | 1.66e-1 | N/A | 3.92e-4 | 7.85e-3 | 1.95e-3 | 1.58e-4 |
| Erf | N/A | N/A | 3.74e-4 | N/A | 1.85e-7 | 8.38e-6 | 4.32e-7 | 5.02e-8 | N/A | N/A | 1.47e-2 | N/A | 2.46e-4 | 2.30e-3 | 1.95e-3 | 1.88e-4 |
| Sin | 5.23e-4 | N/A | 9.74e-5 | N/A | 1.30e-7 | 2.99e-3 | N/A | 3.85e-8 | 1.27e-2 | N/A | 8.15e-3 | N/A | 2.46e-4 | 5.47e-2 | N/A | 1.66e-4 |
| Rsqrt | N/A | N/A | 3.79e-1 | N/A | 8.05e-5 | 2.10e-2 | N/A | 6.83e-8 | N/A | N/A | 1.90e-1 | N/A | 5.40e-4 | 1.10e-1 | N/A | 1.77e-4 |
| GeLU | 1.65e-1 | N/A | 3.27e-5 | N/A | 6.10e-9 | 6.67e-6 | 1.19e-6 | 4.74e-8 | 2.41e-1 | N/A | 3.41e-3 | N/A | 3.63e-5 | 2.08e-3 | 1.95e-3 | 1.30e-4 |
| Geomean | 3.13e5x | N/A | 6.00e4x | N/A | 18.8x | 908x | 31.5x | 1.00x | 460x | N/A | 246x | N/A | 1.99x | 34.6x | 22.4x | 1.00x |
