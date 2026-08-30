# EXP-155 Unified Efficiency Audit

Compares F0, F3, F4, F16 and F35 under the frozen current three-anchor mode-template decoder used by EXP-154. It reports parameter partitions, operator-accounted FLOPs/MACs, sparse-graph and temporal-social score-element counts, peak GPU memory, and interleaved GPU latency for batch sizes 1 and 64.

The FLOP value is the sum of operations that PyTorch Profiler can account for; unsupported custom operations are not silently estimated. F16/F35 attention score counts are therefore reported separately.
