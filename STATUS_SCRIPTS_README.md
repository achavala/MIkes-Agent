# 📊 Training Status Scripts

A collection of scripts to monitor and check training progress.

## Available Scripts

### 1. `quick_status.sh` ⚡
**Fast overview of training status**

```bash
./quick_status.sh
```

**Output:**
- Training process status (running/stopped)
- Current progress percentage
- Timesteps completed/remaining
- Estimated time remaining

**Use when:** You want a quick check without details

---

### 2. `detailed_status.sh` 📊
**Comprehensive training analysis**

```bash
./detailed_status.sh
```

**Output:**
- Process status (CPU, memory, runtime)
- Sleep prevention status
- Power status
- Checkpoint analysis
- Log file analysis
- Final model status
- Time estimates

**Use when:** You need detailed information about training

---

### 3. `monitor_training.sh` 📺
**Real-time monitoring with auto-refresh**

```bash
./monitor_training.sh          # Default: 10 second refresh
./monitor_training.sh 5        # 5 second refresh
./monitor_training.sh 30       # 30 second refresh
```

**Features:**
- Auto-refreshing display
- Progress bar visualization
- Latest metrics
- Real-time updates

**Use when:** You want to watch training progress in real-time

**Stop:** Press `Ctrl+C`

---

### 4. `progress_report.sh` 📈
**Formatted progress report**

```bash
./progress_report.sh
```

**Output:**
- Clean, formatted report
- Progress milestones
- Status summary
- Time estimates

**Use when:** You want a clean, readable report

---

### 5. `check_all_status.sh` 🔍
**All-in-one comprehensive check**

```bash
./check_all_status.sh
```

**Features:**
- Runs all status checks
- Quick status
- Progress report
- Python analysis (if available)

**Use when:** You want everything at once

---

### 6. `check_training_status.sh` (Original)
**Original status check script**

```bash
./check_training_status.sh
```

**Features:**
- Basic status check
- Checkpoint info
- Log file tail

**Use when:** You prefer the original script

---

## Quick Reference

| Script | Speed | Detail Level | Auto-Refresh |
|--------|-------|-------------|--------------|
| `quick_status.sh` | ⚡ Fast | Basic | ❌ |
| `detailed_status.sh` | 🐢 Slow | High | ❌ |
| `monitor_training.sh` | ⚡ Fast | Medium | ✅ |
| `progress_report.sh` | ⚡ Fast | Medium | ❌ |
| `check_all_status.sh` | 🐢 Slow | High | ❌ |

## Examples

### Quick Check
```bash
./quick_status.sh
```

### Detailed Analysis
```bash
./detailed_status.sh
```

### Watch Training (5 second updates)
```bash
./monitor_training.sh 5
```

### Generate Report
```bash
./progress_report.sh > training_report.txt
```

### All-in-One
```bash
./check_all_status.sh
```

## Output Examples

### Quick Status
```
✅ Training: RUNNING
   PID: 17704 | CPU: 93.8% | Memory: 0.4% | Runtime: 02:11:12

📊 Progress: 63.98%
   Timesteps: 3,198,976 / 5,000,000
   Remaining: 1,801,024
   ⏱️  Estimated remaining: ~1.25 hours
```

### Monitor (with progress bar)
```
✅ Training: RUNNING (PID: 17704)
   CPU: 93.8% | Memory: 0.4% | Runtime: 02:11:12

📊 Progress: 63.98%
   [████████████████████████████████████████░░░░░░░░░░░░░░░░]
   Timesteps: 3,198,976 / 5,000,000
   Remaining: 1,801,024
   ⏱️  Rate: 400.2 tps | Remaining: ~1.2h 15m
```

## Tips

1. **Quick checks:** Use `quick_status.sh` for fast updates
2. **Detailed analysis:** Use `detailed_status.sh` when you need full info
3. **Real-time monitoring:** Use `monitor_training.sh` to watch progress
4. **Reports:** Use `progress_report.sh` to generate text reports
5. **Everything:** Use `check_all_status.sh` for comprehensive overview

## Troubleshooting

### Scripts not executable
```bash
chmod +x *.sh
```

### Scripts not found
```bash
# Make sure you're in the project directory
cd /Users/chavala/Mike-agent-project
```

### No training running
```
❌ Training: NOT RUNNING
```
- Check if training was started
- Check PID file: `.training.pid`
- Check logs: `training_*.log`

### No checkpoints
```
⚠️  No checkpoints found yet
```
- Training may have just started
- Check if checkpoint directory exists: `models/checkpoints/`
- Wait for first checkpoint (saved every 50,000 timesteps)

## Notes

- All scripts require `bc` for calculations (usually pre-installed on Mac)
- Scripts automatically detect the latest log file
- Progress calculations are based on checkpoint filenames
- Time estimates are approximate and may vary

---

**Last Updated:** December 9, 2025

