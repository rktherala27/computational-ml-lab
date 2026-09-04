## Results

### Clean data

SINDy recovered the exact Duffing system:

x' = v

v' = -1.0x - 0.3v - 0.5x^3

The discovered model reproduced an unseen trajectory with near-zero relative error.

### Noisy data

Adding 1% Gaussian measurement noise changed the recovered coefficients and could
introduce incorrect terms. Applying `SmoothedFiniteDifference` before derivative
estimation improved recovery by reducing noise amplification during differentiation.