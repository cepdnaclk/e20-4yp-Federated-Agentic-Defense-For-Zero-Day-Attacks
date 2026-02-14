# Quick Start Guide - Detection Metrics Analysis

## What You Have Now

✅ **Working Scripts**: `detect_metrics.py` (cross-platform Python script)  
✅ **Analysis Results**: Complete performance metrics in `detection_metrics/`  
✅ **Detailed Report**: Comprehensive findings in `ANALYSIS_REPORT.md`  

## How to Run the Analysis

### Simple Command (Recommended)
```bash
python detect_metrics.py
```

### With Custom Paths
```bash
python detect_metrics.py --system-metrics "path/to/system_metrics.jsonl" \
                        --unsw-dataset "path/to/UNSW_NB15_training-set.csv" \
                        --output-dir "my_results"
```

## What the Results Tell You

### 🔴 **Critical Issue Found**
Your agent has an **80.7% false positive rate** - it's flagging normal traffic as attacks!

### Key Numbers:
- **Accuracy**: 19.3% (needs improvement)  
- **False Positives**: 67/83 flows (way too high)
- **All flows tested were benign** (no attacks in this segment)

## Quick Fixes to Try

### 1. **Adjust the Threshold** (Immediate)
Your current anomaly threshold is too low. In your code, look for the threshold value and increase it by 10-100x.

### 2. **Whitelist Common Traffic** 
Add exceptions for:
- DNS queries (port 53)
- ARP traffic  
- Standard network protocols

### 3. **Test with Real Attacks**
Run analysis on UNSW flows with actual attacks to see if you can detect them.

## Files Generated

| File | Purpose |
|------|---------|
| `detection_metrics/detailed_results.csv` | Flow-by-flow comparison |
| `ANALYSIS_REPORT.md` | Complete findings and recommendations |
| `detect_metrics.py` | Reusable analysis script |
| `METRICS_README.md` | Detailed documentation |

## Next Steps

1. **Read the full analysis**: Check `ANALYSIS_REPORT.md`
2. **Adjust your thresholds**: Reduce false positives  
3. **Test with attacks**: Use UNSW flows that contain actual attacks
4. **Re-run analysis**: Use the same script after changes

## Need Help?

- **Script not working?** Check Python version (Python 3.6+)
- **File not found?** Adjust paths in the command
- **Want more features?** Check other scripts: `compare_detection_metrics.sh`, `simple_metrics_comparison.sh`

---
**Your agent shows promise but needs calibration!** 🎯