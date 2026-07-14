# Filtered native synthesis reports

`reports/` contains the 112 paper-matching synthesis result sets used for Figures 18 and 19. Paths are intentionally limited to experiment identity:

```text
reports/<baseline|scope|onesa|fusemax|fsa>/<datatype>/<nXX>/
    area.rpt
    power.rpt
    timing.rpt
```

Project IDs, job IDs, service timestamps, status exports, failed jobs, stale sweeps, and duplicate report copies are not included. The reports themselves are unedited native Synopsys Design Compiler V-2023.12 outputs using TSMC 28 nm libraries.

Figure 18's `reproduce.py` extracts and fits these reports. Figure 19 consumes that verified fit for its 32x32 incremental-overhead calculation. Corresponding synthesis-time generated RTL is stored in `../rtl/paper-mesh-snapshot/`.
