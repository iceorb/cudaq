# GPU-Aware Distributed Job Dispatcher

A Python-based job dispatcher that intelligently assigns compute tasks to GPUs based on real-time memory availability, with persistent job tracking and recovery capabilities.

## 🚀 Features

- 🎯 **GPU-aware job scheduling** based on free memory
- 📝 **Persistent job tracking** using JSONL files
- 🔄 **Automatic state recovery** after crashes/restarts
- 📊 **Real-time job status monitoring**
- 📁 **Per-job log files** with stdout/stderr capture
- ⚙️ **Configurable thresholds and GPU selection**

---

## 📦 Installation

1. Install prerequisites:
   ```bash
   pip install pynvml pyyaml
   ```

2. Clone/download the dispatcher script:
   ```bash
   git clone [your-repo-url-here]
   ```

---

## ⚡ Quick Start

1. Create a commands file (`commands.txt`):
   ```txt
   python train.py --batch-size 128
   python inference.py --input-dir ./data
   ```

2. Run the dispatcher:
   ```bash
   python cudaq.py run --commands-file commands.txt --min-free-mem-mb 8000
   ```

3. Check status:
   ```bash
   python cudaq.py status
   ```

---

## ⚙️ Configuration

Create `config.yaml` for persistent settings:
```yaml
gpu_ids: [0, 1, 2]          # Which GPUs to use
min_free_mem_mb: 10000      # Minimum free memory required
poll_interval: 30           # Check interval in seconds
commands_file: jobs.txt     # Job commands source
log_dir: ./job_logs         # Log storage location
jobs_file: ./queue.jsonl    # Job tracking file
```

Start dispatcher with config:
```bash
python cudaq.py run --config config.yaml
```

---

## 📋 Job Management

### Command File Format

Plain text file with one command per line:
```txt
# comments start with #
python train_resnet.py
python process_data.py --workers 4
```

### Job Lifecycle

- **Pending**: Waiting for GPU resources  
- **Running**: Actively executing on GPU  
- **Completed**: Finished successfully  
- **Failed**: Exited with error/crash  

---

## 💾 Persistent Tracking

Jobs are tracked in JSONL format with:
- PID and start time
- Assigned GPU ID
- Status history
- Log file path
- Full command string

---

## 📁 Log Files

- Automatically created in `log_dir`
- Format: `job_YYYYMMDD_HHMMSS.log`
- Contains full stdout/stderr output
- Path stored in job tracking file

---

## 🔍 Status Monitoring

View current job states:
```bash
python cudaq.py status
```

Sample output:
```
[→] python train.py     → GPU 0 → Running (PID 1234)
[✓] python infer.py     → GPU 1 → Completed
[ ]  python eval.py     → GPU N/A → Pending
```

---

## 📄 License

MIT License – see header in source code for details.
```
