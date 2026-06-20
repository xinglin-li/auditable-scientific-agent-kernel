---
name: rolling-backtest
description: Use this skill when evaluating forecasting designs, comparing time-series models (like ARIMA vs ETS), or verifying that chronological simulation rows do not leak future information.
allowed_tools: ["run_approved_script"]
---
# Rolling Origin Backtest Standard Operating Procedure

## Core Execution Steps
1. Define the clear forecasting horizon (h).
2. Establish the fixed initial training window size.
3. Validate that chronological row splitting does not introduce future-information leakage.
4. Compare target predictions against a naive baseline model.