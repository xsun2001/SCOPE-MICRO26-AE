# Design Template

## What the difference vs classical SA

- Additional hardware units/components
- Additional supported operations
- Additional dataflow/data stream
- Estimate area/power overhead vs classical SA

## Description of 2D PE

- ASCII illustration of microarch and signal connection
- Detailed description of how it does for each operation

## Description of 1D PE/Accumulator

- ASCII illustration of microarch and signal connection
- Detailed description of how it does for each operation

## Summary of register usage

A table of each kind of PEs for each register and their type, description, and update method with each operation.

## Dataflow of a full flashattention tile

$$
S = QK^T \\
m_{\text{new}} = \max(m, \mathrm{rowmax}(S)) \\
P = \exp(S - m_{\text{new}}) \\
\alpha = \exp(m - m_{\text{new}}) \\
l_{\text{new}} = \alpha \cdot l + \mathrm{rowsum}(P) \\
O_{\text{new}} = \alpha \cdot O + PV \\
m = m_{\text{new}}
$$

How does this modified SA handles it? Give a step-by-step description with cycle-level annotation. If there're overlapping pipeline, note them. Finally, calculate the cycles needed for one tile and for full size FlashAttention kernel.