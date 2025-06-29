# 🔍 Codebase Architecture Analysis Report
Generated on: 2025-06-29 07:52:25

## 📊 Summary
- **Total Files Analyzed**: 39
- **Critical Files**: 8
- **Normal Files**: 29
- **Archival Candidates**: 2

## 🎯 Critical Files (Keep These!)
These files are essential to your project architecture:

### `utils/config.py`
**Score**: 27.0
**Reasons**: Configuration file, High incoming dependencies (10), Large file (527 lines)
**Stats**: 527 lines, 10 dependents

### `main.py`
**Score**: 10.0
**Reasons**: Main application file, Large file (184 lines)
**Stats**: 184 lines, 0 dependents

### `models/bayesian_lstm.py`
**Score**: 10.0
**Reasons**: High incoming dependencies (4), Large file (694 lines)
**Stats**: 694 lines, 4 dependents

### `data/storage.py`
**Score**: 10.0
**Reasons**: High incoming dependencies (4), Large file (1102 lines)
**Stats**: 1102 lines, 4 dependents

### `monitoring/alerts.py`
**Score**: 8.0
**Reasons**: High incoming dependencies (3), Large file (638 lines)
**Stats**: 638 lines, 3 dependents

### `trading/adaptive_parameters.py`
**Score**: 8.0
**Reasons**: High incoming dependencies (3), Large file (420 lines)
**Stats**: 420 lines, 3 dependents

### `trading/engine.py`
**Score**: 8.0
**Reasons**: High incoming dependencies (3), Large file (541 lines)
**Stats**: 541 lines, 3 dependents

### `utils/logging.py`
**Score**: 7.0
**Reasons**: High incoming dependencies (3)
**Stats**: 90 lines, 3 dependents

## 📦 Archival Candidates
These files might be safely archived or removed:

### `telegram/handlers.py`
**Score**: -2.0
**Reasons**: No incoming dependencies

### `trading/backtester.py`
**Score**: -2.0
**Reasons**: No incoming dependencies
